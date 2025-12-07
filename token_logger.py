#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Token计数器和性能日志记录器

用于跟踪后处理过程中的token使用情况和性能指标。
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional


class TokenLogger:
    """
    Token计数器和性能日志记录器

    记录每次模型调用的token使用情况和性能指标，并保存到日志文件。
    """

    def __init__(self, output_dir: Path):
        """
        初始化TokenLogger

        Args:
            output_dir: 输出目录路径，token.log将保存在此目录中
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "token.log"
        self.lock = Lock()  # 线程安全，防止并发写入冲突
        self.fieldnames = [
            "timestamp", "request_id", "step_name", "page_name",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "end_time", "duration_sec", "tokens_per_sec"
        ]

        # 初始化日志文件
        self._init_log_file()

    def _init_log_file(self):
        """初始化日志文件，写入表头；如存在旧版JSON日志则转换为CSV"""
        with self.lock:
            if not self.log_file.exists() or self.log_file.stat().st_size == 0:
                with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                    self._write_header(f)
                return

            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            first_data_line = next(
                (line for line in lines
                 if line.strip() and not line.startswith('#')),
                ""
            )

            if first_data_line.startswith('{'):
                self._convert_json_log_to_csv(lines)
            elif not first_data_line:
                with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                    self._write_header(f)

    def _write_header(self, file_obj):
        """写入CSV表头及字段说明"""
        file_obj.write("# Token使用日志（CSV格式）\n")
        file_obj.write("# 字段: timestamp, request_id, step_name, page_name, "
                       "prompt_tokens, completion_tokens, total_tokens, end_time, duration_sec, tokens_per_sec\n")
        writer = csv.writer(file_obj)
        writer.writerow(self.fieldnames)

    def _convert_json_log_to_csv(self, lines: List[str]):
        """将旧版JSON行日志转换为CSV并保留原有记录"""
        records: List[Dict] = []
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            try:
                entry = json.loads(line)
                records.append(entry)
            except json.JSONDecodeError:
                continue

        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            self._write_header(f)
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            for entry in records:
                writer.writerow({
                    "timestamp": entry.get("timestamp"),
                    "request_id": entry.get("request_id"),
                    "step_name": entry.get("step_name"),
                    "page_name": entry.get("page_name"),
                    "prompt_tokens": entry.get("prompt_tokens"),
                    "completion_tokens": entry.get("completion_tokens"),
                    "total_tokens": entry.get("total_tokens"),
                    "end_time": entry.get("end_time"),
                    "duration_sec": entry.get("duration_sec"),
                    "tokens_per_sec": entry.get("tokens_per_sec")
                })

    def log_request(self,
                   request_id: str,
                   step_name: str,
                   page_name: str,
                   prompt_tokens: int,
                   completion_tokens: int,
                   duration_sec: float):
        """
        记录一次模型请求的token使用情况

        Args:
            request_id: 请求的唯一标识符
            step_name: 处理步骤名称（如"grammar", "merge"）
            page_name: 页面名称
            prompt_tokens: 输入token数量
            completion_tokens: 输出token数量
            duration_sec: 处理耗时（秒）
        """
        start_time = datetime.now()
        end_time = start_time.timestamp() + duration_sec

        total_tokens = prompt_tokens + completion_tokens
        tokens_per_sec = completion_tokens / duration_sec if duration_sec > 0 else 0

        log_entry = {
            "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "request_id": request_id,
            "step_name": step_name,
            "page_name": page_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "end_time": datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "duration_sec": round(duration_sec, 3),
            "tokens_per_sec": round(tokens_per_sec, 2)
        }

        # 增量写入日志文件
        with self.lock:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    log_entry["timestamp"],
                    log_entry["request_id"],
                    log_entry["step_name"],
                    log_entry["page_name"],
                    log_entry["prompt_tokens"],
                    log_entry["completion_tokens"],
                    log_entry["total_tokens"],
                    log_entry["end_time"],
                    log_entry["duration_sec"],
                    log_entry["tokens_per_sec"]
                ])
                f.flush()

        # 同时输出到控制台（可选）
        print(f"[TokenLogger] {step_name} | {page_name} | "
              f"输入:{prompt_tokens} 输出:{completion_tokens} | "
              f"耗时:{duration_sec:.1f}s 速率:{tokens_per_sec:.1f} tokens/s")

    def get_summary(self) -> Dict:
        """
        获取token使用统计摘要

        Returns:
            Dict: 包含总token数、平均速率等统计信息
        """
        if not self.log_file.exists():
            return {}

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_duration = 0
        request_count = 0

        with open(self.log_file, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip() and not line.startswith('#')]

        if not lines:
            return {}

        reader = csv.DictReader(lines)
        for entry in reader:
            try:
                total_prompt_tokens += int(float(entry.get('prompt_tokens', 0)))
                total_completion_tokens += int(float(entry.get('completion_tokens', 0)))
                total_duration += float(entry.get('duration_sec', 0) or 0)
                request_count += 1
            except (ValueError, TypeError):
                continue

        if request_count == 0:
            return {}

        return {
            "total_requests": request_count,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "total_duration_sec": round(total_duration, 3),
            "avg_tokens_per_sec": round(total_completion_tokens / total_duration, 2) if total_duration > 0 else 0,
            "avg_request_duration_sec": round(total_duration / request_count, 3)
        }

    def save_summary(self):
        """保存统计摘要到文件"""
        summary = self.get_summary()
        if summary:
            summary_file = self.output_dir / "token_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n[TokenLogger] 统计摘要已保存到: {summary_file}")
            print(f"[TokenLogger] 总请求数: {summary['total_requests']}")
            print(f"[TokenLogger] 总token数: {summary['total_tokens']}")
            print(f"[TokenLogger] 平均速率: {summary['avg_tokens_per_sec']} tokens/s")


# 全局token logger实例（延迟初始化）
_global_token_logger: Optional[TokenLogger] = None


def get_token_logger() -> Optional[TokenLogger]:
    """获取全局token logger实例"""
    return _global_token_logger


def init_token_logger(output_dir: Path):
    """初始化全局token logger"""
    global _global_token_logger
    _global_token_logger = TokenLogger(output_dir)


def log_token_usage(request_id: str,
                   step_name: str,
                   page_name: str,
                   prompt_tokens: int,
                   completion_tokens: int,
                   duration_sec: float):
    """
    便捷函数：记录token使用情况

    如果token logger未初始化，则忽略调用。
    """
    logger = get_token_logger()
    if logger:
        logger.log_request(request_id, step_name, page_name,
                          prompt_tokens, completion_tokens, duration_sec)
