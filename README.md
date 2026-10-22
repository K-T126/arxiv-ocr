# arxiv-ocr (LLM GraphRAG)

## プロジェクト概要

arXiv の学術論文の中から **LLM（大規模言語モデル）分野の論文** を対象に、「モデルや手法同士の派生関係（例: LLaMA → Alpaca → Vicuna）」や「どのベンチマークデータセット（MMLU, GSM8K 等）でどんな性能を出したか」という関係性を抽出し、多段階推論が可能なグラフ RAG（GraphRAG）を構築する。

---

## 論文の対象選定方針（アプローチ: arXiv API フィルタリング）

ノイズとなる他分野（画像認識、物理、ロボティクス等）の論文を除外し、**LLM 関連論文のみを確実に収集** するため、arXiv API によるカテゴリおよびキーワード絞り込みを採用する。

* **指定カテゴリ**: `cs.CL`（Computation and Language: 自然言語処理分野）
* **検索キーワード**: `LLM` OR `"Large Language Model"`
* **仕組み**: 
  * Python の `arxiv` クライアント経由で arXiv API を叩き、条件に完全合致する論文のみを自動検索して PDF をダウンロードする。
  * これにより、論文全文を解析する前にシステム外部で完全に LLM 関連論文のみに限定できる。

---

## 使用モデル & APIプロバイダー

* **モデル**: `Qwen/Qwen2.5-72B-Instruct`
  * 根拠論文: https://arxiv.org/pdf/2509.25922
  * テキスト読み取り・情報抽出において軽量かつ高精度。
* **APIプロバイダー**: **OpenRouter**（または OpenAI 互換 API プロバイダー）
  * **採用理由（長文トークン制限の解決）**: Hugging Face 無料サーバーレス API の制限を回避し、Qwen2.5 本来の **128k トークン（長文コンテキスト）** を活用して論文全文を丸ごと安定して処理するため。
  * クライアントには標準の `openai` SDK を使用。

---

## ディレクトリ構成

```text
arxiv-ocr/
├── data/                    # ダウンロードされた論文PDFの格納先（自動保存）
│   ├── 2302.13971v1.pdf     # 例: LLaMA論文
│   └── ...
├── arxiv_graph.kuzu/        # KùzuDB（グラフデータベースの格納フォルダ / 自動生成）
├── .env                     # APIキー等の環境変数
├── .gitignore
├── requirements.txt         # 依存ライブラリ一覧
├── ingest.py                # arXiv検索・PDF自動取得・要約抽出・グラフDB登録
├── query.py                 # 質問回答（グラフRAG検索）スクリプト
├── visualize.py             # グラフ可視化スクリプト
└── README.md
```

---

## プロジェクトの実装ステップ

### Step 1: 環境準備

1. **Python 仮想環境の作成**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **ライブラリのインストール**:
   * `arxiv`: arXiv API を経由して `cs.CL` かつ LLM 関連の論文を検索・PDF 自動取得。
   * `openai`: OpenRouter 経由で Qwen2.5-72B (128k) を呼び出すクライアント。
   * `pymupdf` (`fitz`): 論文 PDF から高速・省メモリでテキストを抽出。
   * `pydantic`: 抽出データのスキーマ（型定義）バリデーション。
   * `kuzu`: ローカル組み込み型グラフデータベース（KùzuDB）。
   * `pyvis`: インタラクティブなネットワーク図（HTML）の出力。
   * `python-dotenv`: `.env` から環境変数を読み込み。
3. **環境変数の設定 (`.env`)**:
   ```env
   OPENROUTER_API_KEY="your_openrouter_api_key_here"
   ```

---

### Step 2: 論文の取り込みスクリプト (`ingest.py`)

LLM 論文を自動検索（または URL/ID 指定）で取得し、要点を抽出して KùzuDB に保存します。

1. **arXiv API 検索 & PDF ダウンロード (`arxiv`)**:
   * 検索クエリ `cat:cs.CL AND (LLM OR "Large Language Model")` に基づき、指定件数（例: 最新 5 件）の論文メタデータと PDF を `data/` に自動ダウンロード。
   * （URL や arXiv ID を直接指定してダウンロード・取り込みすることも可能）
2. **PyMuPDF (`fitz`)**:
   * ダウンロードした PDF から本文テキストを抽出。
3. **Pydantic スキーマ定義 (LLM 特化)**:
   * 手法/モデル名（`model_name`: 例: "Alpaca"）
   * ベースモデル・親手法（`base_model` / `derived_from`: 例: "LLaMA-7B"）
   * チューニング/手法分類（`method_type`: 例: "Instruction Tuning", "RLHF", "DPO"）
   * 評価ベンチマークとスコア（`evaluations`: `benchmark`（例: "MMLU", "GSM8K"）, `score`）
   * 概要（`summary`）
4. **OpenRouter (`openai` SDK) + `Qwen/Qwen2.5-72B-Instruct`**:
   * 128k コンテキストで論文テキストを送信し、上記の Pydantic スキーマに沿った JSON を構造化出力。
5. **KùzuDB (`kuzu`)**:
   * 以下のノードとリレーションをローカル DB に書き込み。
     * **ノード**: `Model`（LLM名）, `Benchmark`（評価データセット）
     * **リレーション**: 
       * `(:Model)-[:DERIVED_FROM]->(:Model)`
       * `(:Model)-[:EVALUATED_ON {score, metric}]->(:Benchmark)`

---

### Step 3: 質問回答スクリプト (`query.py`)

「LLaMA から派生した指示チューニングモデルと、それぞれの GSM8K のスコアを教えて」といった質問に答えます。

1. **KùzuDB (`kuzu`)**:
   * 質問キーワード（例: LLaMA）から Cypher クエリを実行し、派生ツリーやベンチマークスコアのサブグラフを取得。
2. **OpenRouter (`openai` SDK) + `Qwen/Qwen2.5-72B-Instruct`**:
   * 取得したグラフ構造データをプロンプトにコンテキストとして埋め込み、日本語でわかりやすく回答を生成。

---

### Step 4: グラフ可視化スクリプト (`visualize.py`)

蓄積された LLM の系譜図（モデルの家系図とベンチマーク比較）をブラウザ上でインタラクティブに確認します。

1. **KùzuDB (`kuzu`)**:
   * 保存されているすべての `Model` ノードと `DERIVED_FROM`, `EVALUATED_ON` などのリレーションを取得。
2. **PyVis (`pyvis`)**:
   * インタラクティブな HTML 図（`graph.html`）を生成して保存。ドラッグ＆ドロップでモデル同士の派生関係やつながりを視覚的に探索可能。

---

## 主な使い方

仮想環境を有効化した状態で実行します:

```bash
source .venv/bin/activate
```

### 1. 論文の自動取り込み (`ingest.py`)

`cs.CL` カテゴリかつ LLM 関連の最新論文を自動検索・ダウンロードし、Qwen2.5-72B で要点・ベンチマークを抽出して KùzuDB に格納します。

```bash
# 最新の LLM 論文を 5 件検索して自動取得・DB 登録
python3 ingest.py --max-results 5

# 特定の arXiv ID またはローカル PDF を取り込む場合
python3 ingest.py --arxiv-id 2302.13971
python3 ingest.py --pdf ./data/example.pdf

# （API を呼ばずに PDF 取得とテキスト抽出のみ動作確認したい場合）
python3 ingest.py --dry-run --max-results 1
```

### 2. ナレッジグラフへの質問回答 (`query.py`)

KùzuDB からモデルの派生関係やベンチマーク結果を読み出し、Qwen2.5-72B を使って質問に回答します。

```bash
# コマンドラインから直接質問（単発実行）
python3 query.py "LLaMAから派生したモデルとそれぞれのGSM8Kのスコアを教えて"

# 対話モードの起動
python3 query.py

# （LLM を呼ばずにローカルのグラフ検索結果のみを確認したい場合）
python3 query.py "LLaMA" --no-llm
```

### 3. グラフの可視化 (`visualize.py`)

蓄積されたナレッジグラフからインタラクティブな HTML ネットワーク図を出力します。

```bash
python3 visualize.py
# -> カレントディレクトリに graph.html が生成され、ブラウザで探索可能です
```
