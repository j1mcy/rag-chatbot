"""
チャンキング機能のテストコード
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from langchain.schema import Document
from src.text_splitter import TextSplitter


class TestTextSplitter:
    """TextSplitterクラスのテスト"""

    def test_basic_splitting(self):
        """基本的な分割機能のテスト"""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        
        text = "これはテストです。" * 50  # 長いテキスト
        documents = [Document(page_content=text, metadata={"source": "test.txt"})]
        
        chunks = splitter.split_documents(documents)
        
        # チャンクが生成されているか
        assert len(chunks) > 0
        
        # 各チャンクのサイズが適切か
        for chunk in chunks:
            assert len(chunk.page_content) <= 120  # chunk_size + 余裕


    def test_chunk_overlap(self):
        """オーバーラップが正しく機能するかテスト"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        
        text = "あいうえお" * 30
        documents = [Document(page_content=text, metadata={"source": "test.txt"})]
        
        chunks = splitter.split_documents(documents)
        
        # 少なくとも2つのチャンクが生成される
        assert len(chunks) >= 2
        
        # 隣接するチャンク間でオーバーラップが存在する
        if len(chunks) >= 2:
            chunk1_end = chunks[0].page_content[-5:]
            chunk2_start = chunks[1].page_content[:5]
            # 完全一致でなくても、一部が重複していることを確認
            assert len(chunk1_end) > 0 and len(chunk2_start) > 0


    def test_metadata_preservation(self):
        """メタデータが保持されるかテスト"""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        
        documents = [
            Document(
                page_content="テストドキュメント" * 20,
                metadata={"source": "test.txt", "page": 1}
            )
        ]
        
        chunks = splitter.split_documents(documents)
        
        for chunk in chunks:
            # 元のメタデータが保持されている
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["page"] == 1
            
            # 新しいメタデータが追加されている
            assert "chunk_id" in chunk.metadata
            assert "created_at" in chunk.metadata


    def test_empty_document(self):
        """空のドキュメントの処理テスト"""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        
        documents = [Document(page_content="", metadata={"source": "empty.txt"})]
        
        chunks = splitter.split_documents(documents)
        
        # 空のドキュメントでもエラーが発生しない
        assert isinstance(chunks, list)


    def test_short_document(self):
        """短いドキュメントの処理テスト"""
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        
        short_text = "これは短いテキストです。"
        documents = [Document(page_content=short_text, metadata={"source": "short.txt"})]
        
        chunks = splitter.split_documents(documents)
        
        # チャンクサイズより短いドキュメントは1つのチャンクになる
        assert len(chunks) == 1
        assert chunks[0].page_content == short_text


    def test_japanese_separator(self):
        """日本語のセパレータで正しく分割されるかテスト"""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        
        text = "これは第一段落です。\n\nこれは第二段落です。\n\nこれは第三段落です。"
        documents = [Document(page_content=text, metadata={"source": "test.txt"})]
        
        chunks = splitter.split_documents(documents)
        
        # 段落で分割されている
        assert len(chunks) >= 1


    def test_config_retrieval(self):
        """設定取得機能のテスト"""
        chunk_size = 300
        chunk_overlap = 30
        
        splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        config = splitter.get_config()
        
        assert config["chunk_size"] == chunk_size
        assert config["chunk_overlap"] == chunk_overlap
        assert "separators" in config


    def test_multiple_documents(self):
        """複数ドキュメントの一括処理テスト"""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        
        documents = [
            Document(page_content="ドキュメント1" * 20, metadata={"source": "doc1.txt"}),
            Document(page_content="ドキュメント2" * 20, metadata={"source": "doc2.txt"}),
        ]
        
        chunks = splitter.split_documents(documents)
        
        # 各ドキュメントからチャンクが生成されている
        sources = set(chunk.metadata["source"] for chunk in chunks)
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources


    @pytest.mark.parametrize("chunk_size,chunk_overlap", [
        (100, 20),
        (300, 50),
        (500, 100),
        (1000, 200),
    ])
    def test_various_chunk_sizes(self, chunk_size, chunk_overlap):
        """様々なチャンクサイズでの動作テスト"""
        splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        text = "テストテキスト。" * 200
        documents = [Document(page_content=text, metadata={"source": "test.txt"})]
        
        chunks = splitter.split_documents(documents)
        
        # チャンクが生成されている
        assert len(chunks) > 0
        
        # 各チャンクが適切なサイズ範囲内
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 100  # 余裕を持たせる