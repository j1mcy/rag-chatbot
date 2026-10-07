"""
検索機能モジュール
ベクトルDBからのセマンティック検索を実装
"""

from typing import List, Dict, Any
import time
from langchain_core.documents import Document
from src.vector_store import VectorStore
from config.settings import RETRIEVAL_CONFIG


class Retriever:
    """セマンティック検索を実行するクラス"""

    def __init__(self, vector_store: VectorStore):
        """
        Args:
            vector_store: VectorStoreのインスタンス
        """
        self.vector_store = vector_store
        self.default_top_k = RETRIEVAL_CONFIG["top_k"]["default"]
        self.default_score_threshold = RETRIEVAL_CONFIG["score_threshold"]["default"]

    def search(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
        return_scores: bool = True
    ) -> Dict[str, Any]:
        """
        クエリに基づいてドキュメントを検索

        Args:
            query: 検索クエリ
            top_k: 取得するドキュメント数
            score_threshold: 類似度スコアの閾値
            return_scores: スコアを返すかどうか

        Returns:
            検索結果（documents、scores、performance）
        """
        k = top_k or self.default_top_k
        threshold = score_threshold if score_threshold is not None else self.default_score_threshold

        # 検索時間の計測
        start_time = time.time()

        try:
            if return_scores:
                # スコア付きで検索
                results = self.vector_store.similarity_search_with_score(query, k=k)

                # 閾値でフィルタリング（ChromaDBは距離なので小さいほど類似）
                if threshold > 0:
                    filtered_results = [
                        (doc, score) for doc, score in results if score <= threshold
                    ]
                else:
                    filtered_results = results

                documents = [doc for doc, _ in filtered_results]
                scores = [score for _, score in filtered_results]
            else:
                documents = self.vector_store.similarity_search(
                    query, k=k, score_threshold=threshold
                )
                scores = None

            search_time = time.time() - start_time

            return {
                'documents': documents,
                'scores': scores,
                'performance': {
                    'search_time': search_time,
                    'num_results': len(documents)
                }
            }

        except Exception as e:
            raise Exception(f"検索エラー: {str(e)}")

    def format_results(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        検索結果を整形して返す

        Args:
            search_results: searchメソッドの戻り値

        Returns:
            整形された検索結果
        """
        formatted_results = []
        documents = search_results['documents']
        scores = search_results.get('scores', [None] * len(documents))

        for i, (doc, score) in enumerate(zip(documents, scores)):
            result = {
                'rank': i + 1,
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            }
            formatted_results.append(result)

        return formatted_results

    def get_context_for_llm(self, documents: List[Document]) -> str:
        """
        LLMに渡すためのコンテキスト文字列を生成

        Args:
            documents: 検索結果のドキュメント

        Returns:
            整形されたコンテキスト文字列
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page', '')

            context_part = f"[ソース {i}: {source}"
            if page:
                context_part += f" (ページ {page})"
            context_part += f"]\n{doc.page_content}\n"

            context_parts.append(context_part)

        return "\n".join(context_parts)
