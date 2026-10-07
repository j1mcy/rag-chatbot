"""
テキスト分割モジュール
ドキュメントを適切なサイズに分割する機能
"""

from typing import List
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import CHUNKING_CONFIG


class TextSplitter:
    """テキストを適切なサイズに分割するクラス"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ):
        """
        Args:
            chunk_size: チャンクの最大文字数
            chunk_overlap: チャンク間の重複文字数
            separators: 分割に使用するセパレータのリスト
        """
        self.chunk_size = chunk_size or CHUNKING_CONFIG["chunk_size"]["default"]
        self.chunk_overlap = chunk_overlap or CHUNKING_CONFIG["chunk_overlap"]["default"]
        self.separators = separators or CHUNKING_CONFIG["separators"]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        ドキュメントを適切なサイズに分割

        Args:
            documents: 分割するドキュメントのリスト

        Returns:
            分割されたドキュメント（チャンク）のリスト
        """
        # ドキュメントを分割
        chunks = self.text_splitter.split_documents(documents)

        # メタデータを追加
        current_time = datetime.now().isoformat()
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': f"{chunk.metadata.get('source', 'unknown')}_{i}",
                'created_at': current_time,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            })

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        テキスト文字列を直接分割

        Args:
            text: 分割するテキスト

        Returns:
            分割されたテキストのリスト
        """
        return self.text_splitter.split_text(text)

    def get_config(self) -> dict:
        """
        現在の設定を取得

        Returns:
            設定の辞書
        """
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'separators': self.separators
        }
