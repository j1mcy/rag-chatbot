# test_retriever.py
import os
from dotenv import load_dotenv
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever

load_dotenv()

# 既存のベクトルストアを読み込み
api_key = os.getenv("GOOGLE_API_KEY")
embedding_gen = EmbeddingGenerator(api_key=api_key)
embedding_function = embedding_gen.get_embedding_function()

vector_store = VectorStore(embedding_function)
if not vector_store.load_existing():
    print("エラー: インデックスが見つかりません。Phase 1を先に実行してください。")
    exit(1)

# Retrieverの初期化
retriever = Retriever(vector_store)

# 検索テスト
query = "有給休暇の取得方法を教えてください"
print(f"質問: {query}\n")

results = retriever.search(query, top_k=3)
formatted_results = retriever.format_results(results)

print(f"検索時間: {results['performance']['search_time']:.3f}秒")
print(f"取得件数: {results['performance']['num_results']}件\n")

for result in formatted_results:
    print(f"--- 結果 {result['rank']} ---")
    print(f"ソース: {result['metadata'].get('source')}")
    print(f"スコア: {result['score']:.4f}")
    print(f"内容:\n{result['content'][:200]}...\n")
