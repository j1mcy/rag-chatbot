"""
QAチェーンモジュール
LangChainとGeminiを使用した質問応答システム
"""

import os
import time
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
from src.retriever import Retriever
from config.settings import (
    LLM_CONFIG,
    QA_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    API_RETRY_ATTEMPTS,
    API_RETRY_MIN_WAIT,
    API_RETRY_MAX_WAIT
)


class QAChain:
    """質問応答チェーンクラス"""

    def __init__(
        self,
        retriever: Retriever,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_output_tokens: int = None
    ):
        self.retriever = retriever
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Google API Keyが設定されていません")

        # 属性を先に宣言（エディタの警告を消す）
        self.model = None
        self.temperature = None
        self.max_output_tokens = None
        self.llm = None
        self.prompt_template = None
        self.output_parser = None
        self.chain = None

        # 設定を適用
        self.update_config(model, temperature, max_output_tokens)


    def update_config(self, model=None, temperature=None, max_output_tokens=None):
        """設定値を更新し、LLMとチェーンを再構築する"""

        # 設定値を決定（None の場合はデフォルト）
        self.model = model if model is not None else LLM_CONFIG["model"]["default"]
        self.temperature = temperature if temperature is not None else LLM_CONFIG["temperature"]["default"]
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else LLM_CONFIG["max_output_tokens"]["default"]

        # LLMの初期化
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        # プロンプトテンプレート
        self.prompt_template = PromptTemplate(
            template=SYSTEM_PROMPT + "\n\n" + QA_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )


        # 出力パーサー
        self.output_parser = StrOutputParser()

        # チェーン構築
        self.chain = self.prompt_template | self.llm | self.output_parser


    @retry(
        stop=stop_after_attempt(API_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=1,
            min=API_RETRY_MIN_WAIT,
            max=API_RETRY_MAX_WAIT
        )
    )
    def answer_question(
        self,
        question: str,
        top_k: int = None,
        score_threshold: float = None
    ) -> Dict[str, Any]:
        """
        質問に対して回答を生成（パフォーマンス計測付き）
        """
        try:
            # 1. 関連ドキュメントを検索
            search_start = time.time()
            search_results = self.retriever.search(
                query=question,
                top_k=top_k,
                score_threshold=score_threshold,
                return_scores=True
            )
            search_time = time.time() - search_start

            documents = search_results['documents']

            if not documents:
                return {
                    'answer': '関連する情報が見つかりませんでした',
                    'sources': [],
                    'performance': {
                        'search_time': search_time,
                        'generation_time': 0.0,
                        'total_time': search_time
                    }
                }

            # 2. コンテキストを生成
            context = self.retriever.get_context_for_llm(documents)

            # 3. LLMで回答を生成
            generation_start = time.time()
            response = self.chain.invoke({
                'context': context,
                'question': question
            })
            generation_time = time.time() - generation_start

            # 4. 結果を整形
            formatted_sources = self.retriever.format_results(search_results)
            total_time = search_time + generation_time

            return {
                'answer': response,  # ★ StrOutputParser は str を返すため修正済み
                'sources': formatted_sources,
                'performance': {
                    'search_time': search_time,
                    'generation_time': generation_time,
                    'total_time': total_time,
                    'num_results': len(documents)
                }
            }

        except Exception as e:
            raise RuntimeError(f"回答生成エラー: {e}") from e


