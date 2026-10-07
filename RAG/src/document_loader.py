"""
ドキュメントローダーモジュール
PDF、DOCX、TXT、MDファイルのドキュメント読み込み
"""

import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_core.documents import Document


class DocumentLoader:
    """様々な形式のドキュメントを読み込むクラス"""

    def __init__(self):
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader
        }

    def load_file(self, file_path: str) -> List[Document]:
        """
        単一のファイルを読み込む

        Args:
            file_path: 読み込むファイルのパス

        Returns:
            読み込まれたドキュメントのリスト

        Raises:
            ValueError: サポートされていないファイル形式の場合
            FileNotFoundError: ファイルが存在しない場合
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext not in self.supported_extensions:
            raise ValueError(f"サポートされていないファイル形式: {ext}")

        try:
            loader_class = self.supported_extensions[ext]
            loader = loader_class(file_path)
            documents = loader.load()

            # メタデータにファイル名を追加
            for doc in documents:
                doc.metadata['source'] = os.path.basename(file_path)

            return documents
        except Exception as e:
            raise Exception(f"ファイル読み込みエラー: {str(e)}")

    def load_directory(self, directory_path: str) -> List[Document]:
        """
        ディレクトリ内の全てのサポートされているファイルを読み込む

        Args:
            directory_path: 読み込むディレクトリのパス

        Returns:
            読み込まれた全てのドキュメントのリスト
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"ディレクトリが見つかりません: {directory_path}")

        all_documents = []

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # ディレクトリはスキップ
            if os.path.isdir(file_path):
                continue

            _, ext = os.path.splitext(filename)
            if ext.lower() in self.supported_extensions:
                try:
                    documents = self.load_file(file_path)
                    all_documents.extend(documents)
                except Exception:
                    continue  # 失敗したファイルはスキップ

        return all_documents

    def load_files(self, file_paths: List[str]) -> List[Document]:
        """
        複数のファイルを読み込む

        Args:
            file_paths: 読み込むファイルパスのリスト

        Returns:
            読み込まれた全てのドキュメントのリスト
        """
        all_documents = []

        for file_path in file_paths:
            try:
                documents = self.load_file(file_path)
                all_documents.extend(documents)
            except Exception:
                continue  # 失敗したファイルはスキップ

        return all_documents
    
        def load_documents(self, directory_path: str = "data") -> List[Document]:
        
            return self.load_directory(directory_path)
"""
        Qiita テンプレート互換のメソッド
        data フォルダ内の全ファイルを読み込む
        """
