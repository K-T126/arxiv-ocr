#!/usr/bin/env python3
"""
query.py - 質問回答（GraphRAG 検索）スクリプト

README.md Step 3 の仕様に従い、以下のパイプラインを実行します:
1. KùzuDB (arxiv_graph.kuzu) からモデルの派生関係やベンチマーク評価などのサブグラフを取得
2. OpenRouter (OpenAI SDK) + Qwen/Qwen2.5-72B-Instruct にグラフ情報をコンテキストとして埋め込み、
   ユーザーの質問に日本語で多段階推論した回答を生成
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

from dotenv import load_dotenv
import kuzu
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def retrieve_graph_context(conn: kuzu.Connection, question: Optional[str] = None) -> str:
    """
    KùzuDB からモデルのノード、派生関係 (DERIVED_FROM)、
    ベンチマーク評価 (EVALUATED_ON) を取得し、LLM 向けテキストコンテキストに整形する。
    """
    lines = []
    lines.append("=== KùzuDB Knowledge Graph Context ===")

    # 1. すべての Model ノードとその派生親 (DERIVED_FROM)
    model_query = """
    MATCH (m:Model)
    OPTIONAL MATCH (m)-[:DERIVED_FROM]->(parent:Model)
    RETURN m.name, m.method_type, m.summary, parent.name
    """
    try:
        res = conn.execute(model_query)
        models_info = []
        while res.has_next():
            row = res.get_next()
            m_name, method_type, summary, parent_name = row[0], row[1], row[2], row[3]
            line = f"- Model: {m_name} (Method: {method_type or 'Unknown'})"
            if parent_name:
                line += f" -> Derived From: {parent_name}"
            if summary:
                line += f" | Summary: {summary}"
            models_info.append(line)

        if models_info:
            lines.append("\n[Models and Derivations (Lineage)]:")
            lines.extend(models_info)
        else:
            lines.append("\n[Models and Derivations]: No models found in database.")
    except Exception as e:
        logger.warning("Failed to query models: %s", e)

    # 2. ベンチマーク評価 (EVALUATED_ON)
    eval_query = """
    MATCH (m:Model)-[r:EVALUATED_ON]->(b:Benchmark)
    RETURN m.name, b.name, r.score, r.metric
    """
    try:
        res = conn.execute(eval_query)
        evals_info = []
        while res.has_next():
            row = res.get_next()
            m_name, b_name, score, metric = row[0], row[1], row[2], row[3]
            evals_info.append(
                f"- {m_name} on {b_name}: Score = {score or 'N/A'} (Metric: {metric or 'N/A'})"
            )

        if evals_info:
            lines.append("\n[Benchmark Evaluations]:")
            lines.extend(evals_info)
        else:
            lines.append("\n[Benchmark Evaluations]: No evaluation records found in database.")
    except Exception as e:
        logger.warning("Failed to query evaluations: %s", e)

    return "\n".join(lines)


def answer_question(
    client: OpenAI,
    model: str,
    conn: kuzu.Connection,
    question: str,
    show_context: bool = False,
    no_llm: bool = False,
) -> str:
    """知識グラフからコンテキストを取得し、LLM で回答を生成する"""
    graph_context = retrieve_graph_context(conn, question=question)

    if show_context:
        print(f"\n{graph_context}\n")

    if no_llm:
        return f"[No-LLM Mode: ナレッジグラフ検索結果]\n{graph_context}"

    system_prompt = (
        "あなたは学術論文のナレッジグラフを活用する高度な AI リサーチアシスタントです。\n"
        "提供される【ナレッジグラフ情報】（LLM の派生関係・系統樹、ベンチマークスコア等）に基づいて、"
        "ユーザーの質問に正確かつ論理的に日本語で回答してください。\n\n"
        "回答の指針:\n"
        "1. ナレッジグラフに記載されている派生元・派生先モデルの関係や、ベンチマークスコア（MMLU, GSM8K等）を根拠として明確に提示してください。\n"
        "2. もしナレッジグラフに直接の情報が存在しない場合は、グラフ上にデータがない旨を断った上で、あなたの持っている一般知識を補足として回答してください。\n"
        "3. 丁寧で分かりやすい構成（箇条書きなどを活用）で回答してください。"
    )

    user_prompt = (
        f"【ナレッジグラフ情報】\n"
        f"{graph_context}\n\n"
        f"【質問】\n"
        f"{question}\n\n"
        f"上記の情報をもとに日本語で回答してください。"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("LLM API Call failed (%s): %s", model, e)
        return (
            f"⚠️ OpenRouter API 呼び出しでエラーが発生しました: {e}\n\n"
            f"【取得されたナレッジグラフ情報（ローカル検索結果）】:\n"
            f"{graph_context}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about LLM lineage and benchmarks using GraphRAG"
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="Question to ask (if omitted, interactive mode starts)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./arxiv_graph.kuzu",
        help="Path to KùzuDB directory (default: ./arxiv_graph.kuzu)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID for OpenRouter (default: env OPENROUTER_MODEL or qwen/qwen-2.5-72b-instruct)",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Display retrieved graph context before generating answer",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation and output retrieved graph context only",
    )

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not set in environment or .env file.")
        sys.exit(1)

    model_name = args.model or os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")

    if not os.path.exists(args.db_path):
        logger.error(
            "Database path does not exist: %s. Please run ingest.py first to create and populate the graph.",
            args.db_path,
        )
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    db = kuzu.Database(args.db_path)
    conn = kuzu.Connection(db)

    if args.question:
        # 単発実行モード
        print(f"\n[質問]: {args.question}")
        answer = answer_question(
            client,
            model_name,
            conn,
            args.question,
            show_context=args.show_context,
            no_llm=args.no_llm,
        )
        print(f"\n[回答]:\n{answer}\n")
    else:
        # 対話型モード
        print("=== GraphRAG 質問対話システム (終了するには 'exit' または 'quit' と入力) ===")
        while True:
            try:
                q = input("\n質問を入力してください > ").strip()
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "q"):
                    print("終了します。")
                    break

                print("\n検索 & 回答生成中...")
                answer = answer_question(
                    client,
                    model_name,
                    conn,
                    q,
                    show_context=args.show_context,
                    no_llm=args.no_llm,
                )
                print(f"\n[回答]:\n{answer}\n")
            except (KeyboardInterrupt, EOFError):
                print("\n終了します。")
                break


if __name__ == "__main__":
    main()
