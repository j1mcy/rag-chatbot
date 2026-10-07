import os
from dotenv import load_dotenv
from google import genai

# 環境変数の読み込み
load_dotenv()

# APIキーの取得
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("エラー: GOOGLE_API_KEY が設定されていません")
    exit(1)

# クライアントの初期化
client = genai.Client(api_key=api_key)

# モデルに問い合わせ
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="こんにちは。簡単に自己紹介してください。"
)

print("APIの接続に成功しました！")
print("\nGeminiからの応答:")
print(response.text)
