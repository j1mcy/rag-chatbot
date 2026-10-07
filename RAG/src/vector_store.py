"""
ベクトルDBモジュール
ChromaDBを使用したベクトルストアの管理
"""

import os
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import CHROMA_DB_PATH, COLLECTION_NAME


class VectorStore:
    """ChromaDBを使用したベクトルストア管理クラス"""

    def __init__(self, embedding_function, persist_directory: str = None, collection_name: str = None):
        """
        Args:
            embedding_function: Embedding関数
            persist_directory: ChromaDBの永続化ディレクトリ
            collection_name: コレクション名
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory or CHROMA_DB_PATH
        self.collection_name = collection_name or COLLECTION_NAME
        self.vector_store = None

        # ディレクトリが存在しない場合は作成
        os.makedirs(self.persist_directory, exist_ok=True)

    def create_from_documents(self, documents: List[Document]) -> None:
        """
        ドキュメントからベクトルストアを作成

        Args:
            documents: インデックス化するドキュメントのリスト
        """
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

    def load_existing(self) -> bool:
        """
        既存のベクトルストアを読み込む

        Returns:
            読み込みに成功した場合True、失敗した場合False
        """
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name
            )

            # コレクションが空でないか確認
            collection = self.vector_store._collection
            count = collection.count()

            return count > 0

        except Exception:
            return False

    def add_documents(self, documents: List[Document]) -> None:
        """
        既存のベクトルストアにドキュメントを追加

        Args:
            documents: 追加するドキュメントのリスト
        """
        if self.vector_store is None:
            raise ValueError("ベクトルストアが初期化されていません")

        self.vector_store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        類似度検索を実行

        Args:
            query: 検索クエリ
            k: 取得するドキュメント数
            score_threshold: 類似度スコアの閾値。Noneの場合は閾値なし

        Returns:
            検索結果のドキュメントリスト
        """
        if self.vector_store is None:
            raise ValueError("ベクトルストアが初期化されていません")

        if score_threshold is not None:
            # スコア付きで検索
            results = self.vector_store.similarity_search_with_score(query, k=k)
            # 閾値でフィルタリング
            filtered_results = [
                doc for doc, score in results if score >= score_threshold
            ]
            return filtered_results
        else:
            return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple]:
        """
        類似度スコア付きで検索を実行

        Args:
            query: 検索クエリ
            k: 取得するドキュメント数

        Returns:
            (ドキュメント, スコア)のタプルのリスト
        """
        if self.vector_store is None:
            raise ValueError("ベクトルストアが初期化されていません")

        return self.vector_store.similarity_search_with_score(query, k=k)

    def delete_collection(self) -> None:
        """コレクションを削除"""
        if self.vector_store is not None:
            self.vector_store.delete_collection()
            self.vector_store = None

    def get_vector_store(self):
        """ベクトルストアのインスタンスを取得"""
        return self.vector_store

    def get_document_count(self) -> int:
        """
        ベクトルストア内のドキュメント数を取得

        Returns:
            ドキュメント数
        """
        if self.vector_store is None:
            return 0

        try:
            collection = self.vector_store._collection
            return collection.count()
        except Exception:
            return 0
