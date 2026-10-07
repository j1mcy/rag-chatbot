"""
ファイル処理ユーティリティ
ファイルのバリデーションと情報取得機能
"""

import os
from typing import List, Tuple
from config.settings import SUPPORTED_FILE_TYPES, MAX_FILE_SIZE_MB


def validate_file(file_path: str) -> Tuple[bool, str]:
    """
    ファイルが有効かどうかを検証

    Args:
        file_path: 検証するファイルのパス

    Returns:
        (有効かどうか, エラーメッセージ)のタプル
    """
    # ファイルの存在確認
    if not os.path.exists(file_path):
        return False, "ファイルが存在しません"

    # ファイルサイズの確認
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"ファイルサイズが{MAX_FILE_SIZE_MB}MBを超えています（{file_size_mb:.2f}MB）"

    # 拡張子の確認
    _, ext = os.path.splitext(file_path)
    ext = ext.lower().lstrip('.')

    if ext not in SUPPORTED_FILE_TYPES:
        return False, f"サポートされていないファイル形式です。対応形式: {', '.join(SUPPORTED_FILE_TYPES)}"

    return True, ""


def get_file_info(file_path: str) -> dict:
    """
    ファイル情報を取得

    Args:
        file_path: ファイルのパス

    Returns:
        ファイル情報の辞書
    """
    if not os.path.exists(file_path):
        return None

    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_name)

    return {
        'name': file_name,
        'path': file_path,
        'size_bytes': file_size,
        'size_mb': file_size / (1024 * 1024),
        'extension': ext.lower().lstrip('.')
    }


def list_files_in_directory(directory_path: str, extensions: List[str] = None) -> List[str]:
    """
    ディレクトリ内のファイル一覧を取得

    Args:
        directory_path: ディレクトリのパス
        extensions: フィルタリングする拡張子のリスト。Noneの場合は全ファイル

    Returns:
        ファイルパスのリスト
    """
    if not os.path.exists(directory_path):
        return []

    file_paths = []
    extensions = extensions or SUPPORTED_FILE_TYPES

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # ディレクトリはスキップ
        if os.path.isdir(file_path):
            continue

        _, ext = os.path.splitext(filename)
        ext = ext.lower().lstrip('.')

        if ext in extensions:
            file_paths.append(file_path)

    return sorted(file_paths)


def format_file_size(size_bytes: int) -> str:
    """
    ファイルサイズをバイト数から人間が読みやすい形式に変換

    Args:
        size_bytes: バイト単位のファイルサイズ

    Returns:
        フォーマットされたファイルサイズ文字列
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"
