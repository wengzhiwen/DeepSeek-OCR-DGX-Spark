#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NGram 重复抑制处理器

用于阻止模型生成重复的 N-gram 序列，有效抑制幻觉输出。
来源: DeepSeek-OCR 官方仓库
"""

import torch
from transformers import LogitsProcessor
from typing import Set


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    阻止 N-gram 重复的 LogitsProcessor。
    
    在生成过程中，如果检测到即将生成的 token 会导致 N-gram 重复，
    则将该 token 的 logit 设置为 -inf，从而阻止其生成。
    
    Args:
        ngram_size: N-gram 的大小（例如 30 表示检测 30 个 token 的重复）
        window_size: 检测窗口大小（只在最近的 window_size 个 token 中检测重复）
        whitelist_token_ids: 白名单 token IDs，这些 token 允许重复（如表格标签 <td>, </td>）
    """

    def __init__(self,
                 ngram_size: int,
                 window_size: int = 100,
                 whitelist_token_ids: set = None):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
            )
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(
                f"`window_size` has to be a strictly positive integer, but is {window_size}"
            )
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()

    def __call__(self, input_ids,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        处理 logits，阻止会导致 N-gram 重复的 token。
        
        Args:
            input_ids: 当前已生成的 token IDs（可以是 List[int] 或 torch.Tensor）
            scores: 当前步骤的 logits
            
        Returns:
            处理后的 logits
        """
        # 兼容 List[int] 和 torch.Tensor
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
            if isinstance(input_ids[0], list):
                # 批量处理时取第一个序列
                input_ids = input_ids[0]

        if len(input_ids) < self.ngram_size:
            return scores

        # 获取当前的 N-gram 前缀
        current_prefix = tuple(input_ids[-(self.ngram_size - 1):])

        # 在窗口内搜索重复的 N-gram
        search_start = max(0, len(input_ids) - self.window_size)
        search_end = len(input_ids) - self.ngram_size + 1

        banned_tokens = set()
        for i in range(search_start, search_end):
            ngram = tuple(input_ids[i:i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned_tokens.add(ngram[-1])

        # 移除白名单中的 token
        banned_tokens = banned_tokens - self.whitelist_token_ids

        # 将被禁止的 token 的 logit 设置为 -inf
        if banned_tokens:
            scores = scores.clone()
            for token in banned_tokens:
                if scores.dim() == 1:
                    scores[token] = -float("inf")
                else:
                    scores[:, token] = -float("inf")

        return scores
