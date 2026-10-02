# arxiv-ocr

## プロジェクト概要

arxivの学術論文を「論文同士の手法の派生関係」「どのデータセットでどの手法が使 
  われ、どんな性能だったか」といった関係性の多段階推論が可能なRAGを構築する


## 使用モデル（テキスト読み取り）

テキスト読み取りモデルはQwenが軽量で精度が良いため採用（Hugging FaceのAPIで使用可）
ただし、MediumとHardで比較すると、Hardの落ち方が激しく要注意である
根拠論文：https://arxiv.org/pdf/2509.25922

 ## 4. プロジェクトの実装ステップ                                                                              
                                                                                                                
  以下のフェーズ順に進めていくことを提案します：                                                                
                                                                                                                
  1. Phase 1: 環境セットアップ & 基盤構築                                                                       
      • 必要な依存関係（kuzu, llama-cpp-python, transformers, pypdf 等）の整理                                  
      • 軽量LLMの選定・ロードテスト（4-bit GGUFのダウンロードとメモリ確認）
  2. Phase 2: 論文オントロジー抽出パイプラインの実装
      • サンプル論文（arXivのPDF 2〜3本）を用意
      • LLMに論文テキストから「Paper - Method - Task - Dataset - Metric」をJSON抽出させるプロンプト設計         
  3. Phase 3: KùzuDBへのグラフ格納
      • グラフスキーマ定義
      • 抽出したJSONからノード・リレーションを自動作成・挿入
  4. Phase 4: Graph-RAG 推論エンジンの構築
      • ユーザーの質問からCypherクエリを生成、または関連サブグラフを取得
      • グラフの文脈をLLMに渡して回答を合成
  5. Phase 5: Docker化 & CLI / WebUI整備
      • コンテナ内でワンコマンドで論文追加＆質問ができるように整備