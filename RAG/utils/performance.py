"""
パフォーマンス計測ユーティリティ
処理時間の計測とレポート生成機能
"""

import time
from typing import Dict, Any
from contextlib import contextmanager


class PerformanceTracker:
    """パフォーマンス計測クラス"""

    def __init__(self):
        self.metrics = {}

    @contextmanager
    def track(self, metric_name: str):
        """
        コンテキストマネージャーで処理時間を計測

        使用例:
            tracker = PerformanceTracker()
            with tracker.track("search"):
                # 処理
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            self.add_metric(metric_name, elapsed_time)

    def add_metric(self, name: str, value: float):
        """メトリクスを追加"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_average(self, name: str) -> float:
        """メトリクスの平均値を取得"""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])

    def format_time(self, seconds: float) -> str:
        """秒数を人間が読みやすい形式に変換"""
        if seconds < 0.001:
            return f"{seconds * 1000000:.2f} μs"
        elif seconds < 1:
            return f"{seconds * 1000:.2f} ms"
        else:
            return f"{seconds:.2f} s"


def format_performance_report(performance_data: Dict[str, Any]) -> str:
    """
    パフォーマンスデータを整形してレポート文字列を生成
    """
    tracker = PerformanceTracker()
    report_lines = ["## パフォーマンスレポート\n"]

    if 'search_time' in performance_data:
        search_time_str = tracker.format_time(performance_data['search_time'])
        report_lines.append(f"- 検索時間: {search_time_str}")

    if 'generation_time' in performance_data:
        gen_time_str = tracker.format_time(performance_data['generation_time'])
        report_lines.append(f"- 回答生成時間: {gen_time_str}")

    if 'total_time' in performance_data:
        total_time_str = tracker.format_time(performance_data['total_time'])
        report_lines.append(f"- 総処理時間: {total_time_str}")

    if 'num_results' in performance_data:
        report_lines.append(f"- 検索結果数: {performance_data['num_results']}件")

    return "\n".join(report_lines)