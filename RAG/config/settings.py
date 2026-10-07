"""
設定ファイル - チャンキング、検索、LLM、Embeddingの設定
"""

# -----------------------------
# チャンキング設定
# -----------------------------
CHUNKING_CONFIG = {
    "chunk_size": {
        "default": 500,
        "min": 100,
        "max": 2000,
        "step": 100,
        "description": "テキストを分割する文字数"
    },
    "chunk_overlap": {
        "default": 50,
        "min": 0,
        "max": 500,
        "step": 10,
        "description": "チャンク間の重複文字数"
    },
    "separators": ["\n\n", "\n", "。", "、", " ", ""]
}

# -----------------------------
# 検索設定（Retriever 用）
# -----------------------------
RETRIEVAL_CONFIG = {
    "top_k": {
        "default": 5,
        "min": 1,
        "max": 20,
        "description": "取得するドキュメント数"
    },
    "score_threshold": {
        "default": 0.0,  # 0.0 = フィルタなし（距離ベース）
        "min": 0.0,
        "max": 1.0,
        "step": 0.1,
        "description": "距離スコアの閾値（Chromaは距離なので小さいほど類似）"
    }
}

# -----------------------------
# LLM設定（Gemini 最新版）
# -----------------------------
LLM_CONFIG = {
    "model": {
        "default": "gemini-2.5-flash",  # ★ 最新モデルに統一
        "options": [
            "gemini-2.5-flash",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
    },
    "temperature": {
        "default": 0.0,
        "min": 0.0,
        "max": 1.0,
        "step": 0.1,
        "description": "回答の創造性"
    },
    "max_output_tokens": {
        "default": 2048,
        "min": 100,
        "max": 8192
    }
}

# -----------------------------
# Embedding設定（Gemini 最新版）
# -----------------------------
EMBEDDING_CONFIG = {
    "model": "gemini-embedding-2",  # 最新の embedding モデル
    "task_type": "retrieval_document",
    "dimension": 768
}

# -----------------------------
# プロンプトテンプレート
# -----------------------------
QA_PROMPT_TEMPLATE = """以下のコンテキストを使用して質問に回答してください。
コンテキストに情報がない場合は、「提供された情報からは回答できません」と答えてください。

コンテキスト:
{context}

質問: {question}

回答:"""

SYSTEM_PROMPT = """あなたは社内文書検索アシスタントです。
提供されたコンテキストに基づいて、正確で簡潔な回答を提供してください。
不確実な情報については、その旨を明示してください。"""

# -----------------------------
# ChromaDB設定
# -----------------------------
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "company_documents"

# -----------------------------
# ファイル処理設定
# -----------------------------
SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt", "md"]
MAX_FILE_SIZE_MB = 10

# -----------------------------
# パフォーマンス設定
# -----------------------------
BATCH_SIZE = 100
API_RETRY_ATTEMPTS = 3
API_RETRY_MIN_WAIT = 4
API_RETRY_MAX_WAIT = 10
