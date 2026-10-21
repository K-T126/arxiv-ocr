# arxiv-ocr

## プロジェクト概要

arXiv の学術論文から「論文同士の手法の派生関係」や「どのデータセットでどの手法が使われ、どんな性能だったか」といった関係性を抽出し、多段階推論が可能なグラフ RAG（GraphRAG）を構築する。

## 使用モデル & APIプロバイダー

* **モデル**: `Qwen/Qwen2.5-72B-Instruct`
  * 根拠論文: https://arxiv.org/pdf/2509.25922
  * テキスト読み取り・情報抽出において軽量かつ高精度。
* **APIプロバイダー**: **OpenRouter**（または OpenAI 互換 API プロバイダー）
  * **採用理由（長文トークン制限の解決）**: Hugging Face の無料サーバーレス API におけるトークン上限・タイムアウト制限を回避し、Qwen2.5 本来の **128k トークン（長文コンテキスト）** をそのまま活用して論文全文を安定して処理するため。
  * クライアントには標準の `openai` SDK を使用。

---

## ディレクトリ構成

```text
arxiv-ocr/
├── data/                    # 論文PDFの格納先（ここに直接PDFを配置）
│   ├── sample_resnet.pdf
│   └── ...
├── arxiv_graph.kuzu/        # KùzuDB（グラフデータベースの格納フォルダ / 自動生成）
├── .env                     # APIキー等の環境変数
├── .gitignore
├── requirements.txt         # 依存ライブラリ一覧
├── ingest.py                # 論文取り込み & グラフDB登録スクリプト
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
   * `openai`: OpenRouter / OpenAI 互換 API 経由で Qwen2.5-72B を呼び出すクライアント。
   * `pymupdf` (`fitz`): arXiv 論文 PDF から高速・省メモリでテキストを抽出。
   * `pydantic`: 抽出データのスキーマ（型定義）バリデーション。
   * `kuzu`: ローカル組み込み型グラフデータベース（KùzuDB）。
   * `pyvis`: インタラクティブなネットワーク図（HTML）の出力。
   * `python-dotenv`: `.env` から環境変数を安全に読み込み。
3. **環境変数の設定 (`.env`)**:
   ```env
   OPENROUTER_API_KEY="your_openrouter_api_key_here"
   ```

---

### Step 2: 論文の取り込みスクリプト (`ingest.py`)

`data/` 配下に配置された論文 PDF を読み込み、要点を抽出して KùzuDB に保存します。

1. **PyMuPDF (`fitz`)**:
   * `data/*.pdf` から本文テキストを抽出。
   * （必要に応じて References 以降をカットし、入力効率を最大化）
2. **Pydantic**:
   * AI に出力させる JSON 構造のスキーマを定義。
     * 手法名（`method_name`）
     * 派生元の親手法（`derived_from`）
     * 評価データセットとスコア一覧（`evaluations`: `dataset`, `metric`, `score`）
     * 論文の要約（`summary`）
3. **OpenRouter (`openai` SDK) + `Qwen/Qwen2.5-72B-Instruct`**:
   * 128k コンテキストを活かして論文テキストを丸ごと送信し、Pydantic スキーマに沿った JSON を構造化出力。
4. **KùzuDB (`kuzu`)**:
   * グラフ DB に以下のノードとリレーションを作成・書き込み。
     * **ノード**: `Method`（手法）、`Dataset`（データセット）
     * **リレーション**: `(:Method)-[:DERIVED_FROM]->(:Method)`, `(:Method)-[:EVALUATED_ON {metric, score}]->(:Dataset)`

---

### Step 3: 質問回答スクリプト (`query.py`)

「ResNet から派生した手法と、その性能を教えて」といった自然言語の質問に対して、グラフデータを活用して回答します。

1. **KùzuDB (`kuzu`)**:
   * 質問に含まれるキーワードをもとに Cypher クエリを実行し、該当手法から伸びる派生関係や評価スコアのサブグラフ（関連データ）を取り出す。
2. **OpenRouter (`openai` SDK) + `Qwen/Qwen2.5-72B-Instruct`**:
   * KùzuDB から取り出した「つながりのデータ」をプロンプトにコンテキストとして埋め込み、「このデータをもとに、ユーザーの質問にわかりやすく日本語で答えてください」と指示して回答を生成。

---

### Step 4: グラフ可視化スクリプト (`visualize.py`)

蓄積された論文・手法の家系図をブラウザでインタラクティブに確認します。

1. **KùzuDB (`kuzu`)**:
   * 保存されているすべての `Method` ノードと `DERIVED_FROM` などのリレーションを全件取得。
2. **PyVis (`pyvis`)**:
   * 取得したノードとエッジを、ドラッグ＆ドロップで動かせるインタラクティブな HTML 図（`graph.html`）に変換して出力。
