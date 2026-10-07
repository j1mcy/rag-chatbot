from google.genai import Client
from google.genai.types import Content
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

# ============================
# 1. Google Gemini クライアント
# ============================
client = Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

# ============================
# 2. SentenceTransformer（埋め込み）
# ============================
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ============================
# 3. ChromaDB（ベクトルストア）
# ============================
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="rag_collection",
    embedding_function=None  # 自前で埋め込みを生成する
)

# ============================
# 4. 埋め込み生成
# ============================
def embed_text(text: str):
    return embedder.encode(text).tolist()

# ============================
# 5. ドキュメント追加
# ============================
def add_document(doc_id: str, text: str):
    embedding = embed_text(text)
    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding]
    )

# 例として追加
add_document("doc1", "有給休暇は労働者が取得できる休暇制度で、事前申請が必要です。")
add_document("doc2", "会社の就業規則に従って有給休暇を申請してください。")

# ============================
# 6. RAG 検索
# ============================
def retrieve(query: str, top_k: int = 3):
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]

# ============================
# 追加 パフォーマンス表示
# ============================

def format_rag_output(question, answer, docs, scores, search_time, gen_time):
    total_time = search_time + gen_time

    # 参照元の整形
    refs = ""
    for i, (doc, score) in enumerate(zip(docs, scores)):
        refs += f"[{i+1}] {doc[:30]}... (スコア: {score:.4f})\n"

    return f"""
質問: {question}

=== 回答 ===
{answer}

=== 参照元 ===
{refs}

=== パフォーマンス ===
検索時間: {search_time:.3f}秒
生成時間: {gen_time:.3f}秒
総処理時間: {total_time:.3f}秒
"""

# ============================
# 7. LLM による回答生成
# ============================
import time

def generate_answer(question: str):
    # 計測開始
    t0 = time.time()

    # RAG 検索
    docs = retrieve(question)
    search_time = time.time() - t0

    context = "\n".join(docs)

    prompt = f"""
以下のコンテキストに基づいて質問に答えてください。

【コンテキスト】
{context}

【質問】
{question}

日本語で丁寧に答えてください。
"""

    # LLM生成
    t1 = time.time()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[Content(role="user", parts=[{"text": prompt}])]
    )
    gen_time = time.time() - t1

    answer = response.text

    # スコア取得
    scores = collection.query(
        query_embeddings=[embed_text(question)],
        n_results=3
    )["distances"][0]

    # 整形して返す
    return format_rag_output(question, answer, docs, scores, search_time, gen_time)


# ============================
# 8. 実行例
# ============================
if __name__ == "__main__":
    question = "有給休暇を取得するにはどうすればいいですか？"
    answer = generate_answer(question)
    print("質問:", question)
    print("回答:", answer)
