"""
RAGチャットボット - Streamlitメインアプリケーション
"""

import streamlit as st
import os
from dotenv import load_dotenv

from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.qa_chain import QAChain
from utils.file_handler import list_files_in_directory, get_file_info
from utils.performance import format_performance_report
from config.settings import (
    CHUNKING_CONFIG,
    RETRIEVAL_CONFIG,
    LLM_CONFIG,
    CHROMA_DB_PATH
)

# 環境変数の読み込み
load_dotenv()

# ページ設定
st.set_page_config(
    page_title="RAG チャットボット",
    page_icon="🤖",
    layout="wide"
)

# セッション状態の初期化
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'indexed_files' not in st.session_state:
    st.session_state.indexed_files = []


def initialize_system():
    """システムの初期化"""
    try:
        # API Keyの確認
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            st.error("⚠️ Google API Keyが設定されていません。.envファイルを確認してください")
            st.info("Google AI Studioでアカウントを作成しAPI Keyを取得してください: https://aistudio.google.com/")
            return False

        # Embeddingジェネレータの初期化
        embedding_gen = EmbeddingGenerator(api_key=api_key)
        embedding_function = embedding_gen.get_embedding_function()

        # ベクトルストアの初期化
        vector_store = VectorStore(embedding_function)

        # 既存のベクトルストアを読み込み
        if vector_store.load_existing():
            st.session_state.vector_store = vector_store
            st.session_state.indexed_files = ["既存のインデックスを読み込みました"]

            # Retrieverの初期化
            retriever = Retriever(vector_store)

            # QAチェーンの初期化
            qa_chain = QAChain(retriever, api_key=api_key)
            st.session_state.qa_chain = qa_chain

            return True
        else:
            st.session_state.vector_store = vector_store
            return True

    except Exception as e:
        st.error(f"エラー: {str(e)}")
        return False


def build_index(data_directory: str, chunk_size: int, chunk_overlap: int):
    """インデックスの構築"""
    try:
        with st.spinner("ドキュメントを読み込んでいます..."):
            # ドキュメントローダー
            loader = DocumentLoader()
            documents = loader.load_directory(data_directory)

            if not documents:
                st.warning("ドキュメントが見つかりませんでした")
                return False

            st.success(f"✓ {len(documents)}件のドキュメントを読み込みました")

        with st.spinner("テキストを分割しています..."):
            # テキスト分割
            splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(documents)
            st.success(f"✓ {len(chunks)}個のチャンクに分割しました")

        with st.spinner("Embeddingを生成してインデックスを作成しています..."):
            # ベクトルストア作成
            st.session_state.vector_store.create_from_documents(chunks)
            st.success("✓ インデックスの作成が完了しました")

        # Retrieverの初期化
        retriever = Retriever(st.session_state.vector_store)

        # QAチェーンの初期化
        api_key = os.getenv("GOOGLE_API_KEY")
        qa_chain = QAChain(retriever, api_key=api_key)
        st.session_state.qa_chain = qa_chain

        # インデックス化ファイルの記録
        files = list_files_in_directory(data_directory)
        st.session_state.indexed_files = [os.path.basename(f) for f in files]

        return True

    except Exception as e:
        st.error(f"インデックス作成エラー: {str(e)}")
        return False


def main():
    """メイン関数"""

    st.title("🤖 RAG チャットボット")
    st.markdown("📚社内文書を使って質問に回答します")

    # サイドバー
with st.sidebar:
    st.header("⚙️ 設定")

    # インデックス構築セクション
    st.subheader("📁 インデックス構築")

    data_dir = st.text_input(
        "ドキュメントディレクトリ",
        value="./data/raw",
        help="読み込むドキュメントが格納されているディレクトリ"
    )

    st.write("**チャンキング設定**")
    chunk_size = st.slider(
        "チャンクサイズ",
        min_value=CHUNKING_CONFIG["chunk_size"]["min"],
        max_value=CHUNKING_CONFIG["chunk_size"]["max"],
        value=CHUNKING_CONFIG["chunk_size"]["default"],
        step=CHUNKING_CONFIG["chunk_size"]["step"],
        help=CHUNKING_CONFIG["chunk_size"]["description"]
    )

    chunk_overlap = st.slider(
        "チャンクオーバーラップ",
        min_value=CHUNKING_CONFIG["chunk_overlap"]["min"],
        max_value=CHUNKING_CONFIG["chunk_overlap"]["max"],
        value=CHUNKING_CONFIG["chunk_overlap"]["default"],
        step=CHUNKING_CONFIG["chunk_overlap"]["step"],
        help=CHUNKING_CONFIG["chunk_overlap"]["description"]
    )

    if st.button("🔨 インデックス作成", type="primary"):
        build_index(data_dir, chunk_size, chunk_overlap)

    # 検索設定
    st.subheader("🔍 検索設定")
    top_k = st.slider(
        "取得ドキュメント数",
        min_value=RETRIEVAL_CONFIG["top_k"]["min"],
        max_value=RETRIEVAL_CONFIG["top_k"]["max"],
        value=RETRIEVAL_CONFIG["top_k"]["default"],
        help=RETRIEVAL_CONFIG["top_k"]["description"]
    )

    # LLM設定
    st.subheader("🤖 LLM設定")
    temperature = st.slider(
        "Temperature",
        min_value=LLM_CONFIG["temperature"]["min"],
        max_value=LLM_CONFIG["temperature"]["max"],
        value=LLM_CONFIG["temperature"]["default"],
        step=LLM_CONFIG["temperature"]["step"],
        help=LLM_CONFIG["temperature"]["description"]
    )

    # インデックス情報
    st.subheader("📊 インデックス情報")
    if st.session_state.vector_store:
        doc_count = st.session_state.vector_store.get_document_count()
        st.info(f"インデックス化済: {doc_count}件")
    else:
        st.info("インデックスは作成されていません")


    # メイン画面
    # システム初期化
    if st.session_state.vector_store is None:
        initialize_system()

    # チャットインターフェース
    st.subheader("💬 チャット")

    # チャット履歴の表示
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 参照元"):
                    for source in message["sources"]:
                        st.markdown(f"**[{source['rank']}] {source['metadata'].get('source', '')}** (スコア: {source['score']:.4f})")
                        st.text(source['content'][:200] + "...")
                        st.divider()

                if "performance" in message:
                    with st.expander("⏱️ パフォーマンス"):
                        st.markdown(format_performance_report(message["performance"]))

    # 質問入力
if question := st.chat_input("質問を入力してください"):
    # QAチェーンが初期化されているか確認
    if st.session_state.qa_chain is None:
        st.warning("⚠️ まずインデックスを作成してください")
    else:
        # ユーザーメッセージの表示
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        # 回答生成
        with st.chat_message("assistant"):
            with st.spinner("回答を生成しています..."):
                try:
                    # LLM設定の更新
                    st.session_state.qa_chain.update_config(temperature=temperature)

                    # 質問に対する回答
                    result = st.session_state.qa_chain.answer_question(
                        question=question,
                        top_k=top_k
                    )

                    # 回答の表示
                    st.markdown(result['answer'])

                    # ソース情報の表示
                    if result['sources']:
                        with st.expander("📚 参照元"):
                            for source in result['sources']:
                                st.markdown(
                                    f"**[{source['rank']}] {source['metadata'].get('source', '')}** "
                                    f"(スコア: {source['score']:.4f})"
                                )
                                st.text(source['content'][:200] + "...")
                                st.divider()

                    # パフォーマンス情報の表示
                    with st.expander("⏱️ パフォーマンス"):
                        st.markdown(format_performance_report(result['performance']))

                    # チャット履歴に追加
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result['sources'],
                        "performance": result['performance']
                    })

                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")


if __name__ == "__main__":
    main()
