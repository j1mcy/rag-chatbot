# test_loader.py
from src.document_loader import DocumentLoader

loader = DocumentLoader()

# ディレクトリから読み込み
documents = loader.load_directory('./data/raw')

print(f"読み込んだドキュメント数: {len(documents)}")
for i, doc in enumerate(documents[:3]):
    print(f"\n--- ドキュメント {i+1} ---")
    print(f"ソース: {doc.metadata.get('source', 'Unknown')}")
    print(f"テキスト（最初の100文字）: {doc.page_content[:100]}...")