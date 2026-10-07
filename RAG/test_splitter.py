# test_splitter.py
from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter

# ドキュメント読み込み
loader = DocumentLoader()
documents = loader.load_directory('./data/raw')

print(f"元のドキュメント数: {len(documents)}")

# テキスト分割
splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"分割後のチャンク数: {len(chunks)}")
print(f"\n設定: {splitter.get_config()}")

# 最初のチャンクを表示
print(f"\n--- チャンク例 ---")
print(f"ソース: {chunks[0].metadata.get('source')}")
print(f"チャンクID: {chunks[0].metadata.get('chunk_id')}")
print(f"テキスト長: {len(chunks[0].page_content)}文字")
print(f"内容:\n{chunks[0].page_content[:200]}...")