#!/usr/bin/env python3
"""
ingest.py - arXiv検索・PDF自動取得・要約抽出・グラフDB登録スクリプト

README.md Step 2 の仕様に従い、以下のパイプラインを実行します:
1. arXiv API で cs.CL かつ LLM 関連論文を検索し、PDF を data/ にダウンロード
2. PyMuPDF (fitz) で論文本文テキストを抽出
3. OpenRouter (OpenAI SDK) + Qwen/Qwen2.5-72B-Instruct で Pydantic スキーマに従い要点抽出
4. KùzuDB (arxiv_graph.kuzu) にノード (Model, Benchmark) とリレーション (DERIVED_FROM, EVALUATED_ON) を登録
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import List, Optional

import arxiv
from dotenv import load_dotenv
import kuzu
from openai import OpenAI
from pydantic import BaseModel, Field
import pymupdf
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Pydantic スキーマ定義 ---

class Evaluation(BaseModel):
    benchmark: str = Field(
        ...,
        description="Name of the evaluation benchmark or dataset (e.g., 'MMLU', 'GSM8K', 'HumanEval')",
    )
    score: Optional[str] = Field(
        None,
        description="Score achieved on this benchmark (e.g., '65.4', '78.2%')",
    )
    metric: Optional[str] = Field(
        None,
        description="Metric used for evaluation (e.g., 'accuracy', 'pass@1', 'BLEU')",
    )


class ModelInfo(BaseModel):
    model_name: str = Field(
        ...,
        description="Name of the model or method introduced, evaluated, or fine-tuned (e.g., 'Alpaca', 'Vicuna-13B')",
    )
    base_model: Optional[str] = Field(
        None,
        description="Parent or base model this model is derived from (e.g., 'LLaMA-7B', 'Mistral-7B'). None if foundational.",
    )
    method_type: Optional[str] = Field(
        None,
        description="Type of training, adaptation, or method (e.g., 'Instruction Tuning', 'RLHF', 'DPO', 'Pre-training')",
    )
    evaluations: List[Evaluation] = Field(
        default_factory=list,
        description="Benchmark evaluations for this model",
    )
    summary: Optional[str] = Field(
        None,
        description="Short summary of the model and its main contributions",
    )


class PaperAnalysis(BaseModel):
    paper_title: str = Field(..., description="Title of the paper")
    models: List[ModelInfo] = Field(
        default_factory=list,
        description="List of LLM models or methods analyzed or proposed in this paper",
    )


# --- データベース初期化 ---

def init_kuzu_db(db_path: str):
    """KùzuDB データベースとスキーマ（ノード・リレーション）を初期化する"""
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    # Model ノード
    conn.execute(
        """
        CREATE NODE TABLE IF NOT EXISTS Model(
            name STRING,
            method_type STRING,
            summary STRING,
            PRIMARY KEY (name)
        )
        """
    )

    # Benchmark ノード
    conn.execute(
        """
        CREATE NODE TABLE IF NOT EXISTS Benchmark(
            name STRING,
            PRIMARY KEY (name)
        )
        """
    )

    # DERIVED_FROM リレーション: (:Model)-[:DERIVED_FROM]->(:Model)
    conn.execute(
        """
        CREATE REL TABLE IF NOT EXISTS DERIVED_FROM(
            FROM Model TO Model
        )
        """
    )

    # EVALUATED_ON リレーション: (:Model)-[:EVALUATED_ON {score, metric}]->(:Benchmark)
    conn.execute(
        """
        CREATE REL TABLE IF NOT EXISTS EVALUATED_ON(
            FROM Model TO Benchmark,
            score STRING,
            metric STRING
        )
        """
    )

    return db, conn


# --- PDF テキスト抽出 ---

def extract_text_from_pdf(pdf_path: str, max_chars: int = 250000) -> str:
    """PyMuPDF を用いて PDF から本文テキストを抽出する"""
    logger.info("Extracting text from PDF: %s", pdf_path)
    doc = pymupdf.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text.append(page.get_text())

    extracted = "\n\n".join(full_text)
    doc.close()

    # 長すぎる場合は上限でカット（128kトークンコンテキスト内に安全に収める）
    if len(extracted) > max_chars:
        logger.warning(
            "Text truncated from %d to %d characters to respect token limits.",
            len(extracted),
            max_chars,
        )
        extracted = extracted[:max_chars]

    return extracted


# --- OpenRouter + Qwen2.5-72B による要点抽出 ---

def analyze_paper_with_llm(
    client: OpenAI,
    model: str,
    paper_title: str,
    paper_text: str,
) -> PaperAnalysis:
    """論文テキストからモデル派生関係とベンチマーク結果を抽出し、Pydanticモデルに変換する"""
    logger.info("Analyzing paper with LLM (%s): %s", model, paper_title)

    system_prompt = (
        "You are an expert AI research assistant specializing in Large Language Models (LLMs).\n"
        "Your task is to analyze academic papers and extract structured information about models, "
        "their derivation lineage (e.g. Alpaca derived from LLaMA), and benchmark evaluation scores.\n\n"
        "Output MUST be strict, valid JSON conforming to this schema:\n"
        "{\n"
        '  "paper_title": "string",\n'
        '  "models": [\n'
        "    {\n"
        '      "model_name": "string (e.g., Alpaca-7B)",\n'
        '      "base_model": "string or null (e.g., LLaMA-7B if fine-tuned from it, null if trained from scratch)",\n'
        '      "method_type": "string or null (e.g., Instruction Tuning, RLHF, DPO, Pre-training, Quantization)",\n'
        '      "evaluations": [\n'
        "        {\n"
        '          "benchmark": "string (e.g., MMLU, GSM8K, HumanEval, ARC, HellaSwag)",\n'
        '          "score": "string or null (e.g., 65.4, 78.2%)",\n'
        '          "metric": "string or null (e.g., accuracy, pass@1)"\n'
        "        }\n"
        "      ],\n"
        '      "summary": "string or null (brief description of contributions)"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Respond ONLY with the JSON object. Do not include introductory text or markdown formatting."
    )

    user_prompt = (
        f"Paper Title: {paper_title}\n\n"
        f"--- Paper Text Content ---\n"
        f"{paper_text}\n\n"
        f"Extract all primary models/methods, their base models (if derived/fine-tuned from an existing LLM), "
        f"and benchmark evaluations as specified in the schema."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content or ""

    # JSON 文字列の抽出（Markdown code block が含まれている場合の対処）
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = content.strip()

    try:
        data = json.loads(json_str)
        analysis = PaperAnalysis(**data)
        logger.info(
            "Successfully extracted %d models from '%s'",
            len(analysis.models),
            paper_title,
        )
        return analysis
    except Exception as e:
        logger.error("Failed to parse LLM response into schema: %s\nRaw output: %s", e, content)
        # フォールバックとして空のモデルリストを返す
        return PaperAnalysis(paper_title=paper_title, models=[])


# --- KùzuDB への保存 ---

def save_to_kuzu(conn: kuzu.Connection, analysis: PaperAnalysis):
    """抽出したデータを KùzuDB に書き込む"""
    for model in analysis.models:
        model_name = model.model_name.strip()
        method_type = model.method_type or "Unknown"
        summary = model.summary or ""

        # 1. Model ノードを登録 (MERGE)
        conn.execute(
            """
            MERGE (m:Model {name: $name})
            ON CREATE SET m.method_type = $method_type, m.summary = $summary
            ON MATCH SET m.method_type = $method_type, m.summary = $summary
            """,
            parameters={
                "name": model_name,
                "method_type": method_type,
                "summary": summary,
            },
        )
        logger.info("Saved Model node: %s", model_name)

        # 2. ベースモデルが存在する場合、親ノード作成と DERIVED_FROM リレーション登録
        if model.base_model:
            base_model_name = model.base_model.strip()
            if base_model_name and base_model_name.lower() != model_name.lower():
                # 親ノードを存在確認/作成
                conn.execute(
                    """
                    MERGE (b:Model {name: $name})
                    ON CREATE SET b.method_type = 'Base Model', b.summary = 'Referenced as base model'
                    """,
                    parameters={"name": base_model_name},
                )

                # リレーション (:Model)-[:DERIVED_FROM]->(:Model)
                conn.execute(
                    """
                    MATCH (m:Model {name: $child}), (b:Model {name: $parent})
                    MERGE (m)-[:DERIVED_FROM]->(b)
                    """,
                    parameters={"child": model_name, "parent": base_model_name},
                )
                logger.info("Created relationship: (:Model %s)-[:DERIVED_FROM]->(:Model %s)", model_name, base_model_name)

        # 3. 評価ベンチマークの登録と EVALUATED_ON リレーション作成
        for ev in model.evaluations:
            b_name = ev.benchmark.strip()
            if not b_name:
                continue

            score = ev.score or "N/A"
            metric = ev.metric or "N/A"

            # Benchmark ノード
            conn.execute(
                """
                MERGE (b:Benchmark {name: $name})
                """,
                parameters={"name": b_name},
            )

            # リレーション (:Model)-[:EVALUATED_ON]->(:Benchmark)
            conn.execute(
                """
                MATCH (m:Model {name: $mname}), (b:Benchmark {name: $bname})
                MERGE (m)-[r:EVALUATED_ON]->(b)
                ON CREATE SET r.score = $score, r.metric = $metric
                ON MATCH SET r.score = $score, r.metric = $metric
                """,
                parameters={
                    "mname": model_name,
                    "bname": b_name,
                    "score": score,
                    "metric": metric,
                },
            )
            logger.info("Created evaluation: (%s)-[:EVALUATED_ON {score: %s}]->(%s)", model_name, score, b_name)


def _save_pdf_from_url(url: str, target_path: str):
    """URL から PDF をダウンロードしてファイルに保存する"""
    headers = {"User-Agent": "arxiv-ocr-pipeline/1.0 (mailto:researcher@example.com)"}
    resp = requests.get(url, headers=headers, stream=True, timeout=30)
    resp.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


# --- arXiv 検索 & PDF ダウンロード ---

def download_arxiv_papers(
    query: str = 'cat:cs.CL AND (LLM OR "Large Language Model")',
    max_results: int = 5,
    download_dir: str = "./data",
    arxiv_id: Optional[str] = None,
) -> List[tuple]:
    """
    arXiv API から論文を検索し PDF をローカルにダウンロードする。
    戻り値: [(arxiv.Result, local_pdf_path), ...]
    """
    os.makedirs(download_dir, exist_ok=True)
    if max_results <= 0 and not arxiv_id:
        logger.info("max_results is 0 and no arxiv_id provided. Skipping download.")
        return []

    client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=3)

    if arxiv_id:
        clean_id = arxiv_id.strip()
        if "arxiv.org" in clean_id:
            clean_id = clean_id.split("/")[-1].replace(".pdf", "")
        logger.info("Fetching specific arXiv paper by ID: %s", clean_id)
        search = arxiv.Search(id_list=[clean_id])
    else:
        logger.info("Searching arXiv: query='%s', max_results=%d", query, max_results)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

    results = []
    for paper in client.results(search):
        paper_id = paper.get_short_id()
        safe_filename = f"{paper_id}.pdf"
        target_path = os.path.join(download_dir, safe_filename)

        if os.path.exists(target_path):
            logger.info("PDF already exists locally: %s", target_path)
        else:
            logger.info("Downloading PDF: %s -> %s", paper.title, target_path)
            try:
                # arxiv 2.x 互換
                if hasattr(paper, "download_pdf"):
                    paper.download_pdf(dirpath=download_dir, filename=safe_filename)
                else:
                    # arxiv 4.x / 直接ダウンロード
                    _save_pdf_from_url(paper.pdf_url, target_path)
            except Exception as e:
                logger.error("Failed to download PDF for %s: %s", paper.title, e)
                continue

        results.append((paper, target_path))

    return results


# --- メイン処理 ---

def main():
    parser = argparse.ArgumentParser(
        description="Ingest arXiv papers into KùzuDB Knowledge Graph"
    )
    parser.add_argument(
        "--query",
        type=str,
        default='cat:cs.CL AND (LLM OR "Large Language Model")',
        help='arXiv search query (default: cat:cs.CL AND (LLM OR "Large Language Model"))',
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Number of papers to fetch (default: 5)",
    )
    parser.add_argument(
        "--arxiv-id",
        type=str,
        default=None,
        help="Specific arXiv paper ID or URL to process (e.g., 2302.13971)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to an existing local PDF file to process directly",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./arxiv_graph.kuzu",
        help="Path to KùzuDB directory (default: ./arxiv_graph.kuzu)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to save downloaded PDFs (default: ./data)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID for OpenRouter (default: env OPENROUTER_MODEL or qwen/qwen-2.5-72b-instruct)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download PDF and extract text only, without calling OpenRouter LLM",
    )
    parser.add_argument(
        "--seed-sample-data",
        action="store_true",
        help="Seed sample LLM lineage and evaluation data into KùzuDB for testing (LLaMA, Alpaca, Vicuna)",
    )

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not set in environment or .env file.")
        sys.exit(1)

    model_name = args.model or os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # KùzuDB 初期化
    logger.info("Initializing KùzuDB at: %s", args.db_path)
    db, conn = init_kuzu_db(args.db_path)

    if args.seed_sample_data:
        logger.info("Seeding sample LLM lineage and evaluation data into KùzuDB...")
        sample_analysis = PaperAnalysis(
            paper_title="Sample LLM Family Benchmarks",
            models=[
                ModelInfo(
                    model_name="LLaMA-7B",
                    base_model=None,
                    method_type="Pre-training",
                    summary="Foundational open-source LLM released by Meta AI.",
                    evaluations=[
                        Evaluation(benchmark="MMLU", score="35.1", metric="accuracy"),
                        Evaluation(benchmark="GSM8K", score="11.0", metric="accuracy"),
                    ],
                ),
                ModelInfo(
                    model_name="Alpaca-7B",
                    base_model="LLaMA-7B",
                    method_type="Instruction Tuning",
                    summary="Stanford Alpaca, fine-tuned from LLaMA-7B using 52k self-instruct demonstrations.",
                    evaluations=[
                        Evaluation(benchmark="MMLU", score="40.5", metric="accuracy"),
                        Evaluation(benchmark="GSM8K", score="14.2", metric="accuracy"),
                    ],
                ),
                ModelInfo(
                    model_name="Vicuna-13B",
                    base_model="LLaMA-13B",
                    method_type="Instruction Tuning",
                    summary="Chat assistant model fine-tuned on ShareGPT conversations from LLaMA.",
                    evaluations=[
                        Evaluation(benchmark="MMLU", score="52.1", metric="accuracy"),
                        Evaluation(benchmark="GSM8K", score="27.6", metric="accuracy"),
                    ],
                ),
            ],
        )
        save_to_kuzu(conn, sample_analysis)
        logger.info("Sample data successfully seeded.")

    targets = []

    if args.pdf:
        # 直接指定されたローカル PDF を処理
        if not os.path.exists(args.pdf):
            logger.error("Specified PDF file does not exist: %s", args.pdf)
            sys.exit(1)
        title = os.path.splitext(os.path.basename(args.pdf))[0]
        targets.append((title, args.pdf))
    else:
        # arXiv から取得
        papers = download_arxiv_papers(
            query=args.query,
            max_results=args.max_results,
            download_dir=args.data_dir,
            arxiv_id=args.arxiv_id,
        )
        for p, pdf_path in papers:
            targets.append((p.title, pdf_path))

    if not targets:
        logger.warning("No papers found or specified to process.")
        return

    logger.info("Found %d paper(s) to process.", len(targets))

    # 各論文の処理
    for title, pdf_path in targets:
        logger.info("==========================================")
        logger.info("Processing: %s", title)
        try:
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                logger.warning("No text could be extracted from %s. Skipping.", pdf_path)
                continue

            if args.dry_run:
                logger.info(
                    "[Dry-Run] Extracted %d characters from PDF. Skipping OpenRouter LLM call and DB write.",
                    len(text),
                )
                continue

            analysis = analyze_paper_with_llm(
                client=client,
                model=model_name,
                paper_title=title,
                paper_text=text,
            )

            save_to_kuzu(conn, analysis)
            logger.info("Successfully ingested paper: %s", title)

        except Exception as e:
            logger.error("Error processing paper %s (%s): %s", title, pdf_path, e, exc_info=True)

    logger.info("All ingestion tasks completed.")


if __name__ == "__main__":
    main()
