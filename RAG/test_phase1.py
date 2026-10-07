"""
Phase 1の統合テスト
"""

import os
from dotenv import load_dotenv
from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
 
# 環境変数の読み込み
load_dotenv()
 
def main():
    print("=== Phase 1: インデックス作成 ===\n")
   
    # 1. ドキュメント読み込み
    print("1. ドキュメントを読み込んでいます...")
    loader = DocumentLoader()
    documents = loader.load_directory('./data/raw')
    print(f"   読み込み完了: {len(documents)}件のドキュメント\n")
   
    # 2. テキスト分割
    print("2. テキストを分割しています...")
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"   分割完了: {len(chunks)}個のチャンク\n")
   
    # 3. Embedding生成とベクトルストア作成
    print("3. Embeddingを生成してインデックスを作成しています...")
    print("   （この処理には数分かかる場合があります）")
   
    api_key = os.getenv("GOOGLE_API_KEY")
    embedding_gen = EmbeddingGenerator(api_key=api_key)
    embedding_function = embedding_gen.get_embedding_function()
   
    vector_store = VectorStore(embedding_function)
    vector_store.create_from_documents(chunks)
   
    print(f"   インデックス作成完了\n")
   
    # 4. 確認
    doc_count = vector_store.get_document_count()
    print(f"4. インデックス化されたドキュメント数: {doc_count}件")
   
    print("\n=== Phase 1 完了 ===")
 
if __name__ == "__main__":
    main()
