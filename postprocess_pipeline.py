#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
后处理管道模块

提供可扩展的后处理管道架构，支持多步骤处理。
"""

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

from postprocess_vllm import (PostProcessVLLM, estimate_tokens,
                              list_ocr_md_files, load_ocr_pages, timestamp)
# 导入token logger
from token_logger import init_token_logger, get_token_logger


class PostProcessStep(ABC):
    """
    后处理步骤抽象基类。

    所有后处理步骤都应该继承此类并实现 process 方法。
    """

    @abstractmethod
    async def process(self, pages: List[Tuple[str, str]],
                     engine: PostProcessVLLM) -> List[Tuple[str, str]]:
        """
        处理页面列表。

        Args:
            pages: (页面名称, 页面内容) 的列表
            engine: PostProcessVLLM 引擎实例

        Returns:
            List[Tuple[str, str]]: 处理后的页面列表
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        返回步骤名称。

        Returns:
            str: 步骤名称
        """
        pass


class GrammarCorrectionStep(PostProcessStep):
    """
    步骤1：日语语法错误修正。

    逐页检查并修正明显的日语语法错误。
    """

    def __init__(self, show_progress: bool = True):
        """
        初始化语法修正步骤。

        Args:
            show_progress: 是否显示进度
        """
        self.show_progress = show_progress

    async def process(self, pages: List[Tuple[str, str]],
                      engine: PostProcessVLLM) -> List[Tuple[str, str]]:
        """
        逐页修正语法错误。

        Args:
            pages: (页面名称, 页面内容) 的列表
            engine: PostProcessVLLM 引擎实例

        Returns:
            List[Tuple[str, str]]: 修正后的页面列表
        """
        print("\n" + "=" * 60)
        print(f"{timestamp()} 开始步骤1: 日语语法错误修正")
        print("=" * 60)

        corrected_pages = []
        total = len(pages)

        for i, (page_name, content) in enumerate(pages, 1):
            print(f"\n{timestamp()} 处理页面 {i}/{total}: {page_name}")
            try:
                corrected_content = await engine.correct_grammar_async(
                    content, show_progress=self.show_progress)
                corrected_pages.append((page_name, corrected_content))
                print(f"{timestamp()} ✓ 完成: {page_name}")
            except Exception as e:
                print(f"{timestamp()} ✗ 错误: {page_name} - {e}")
                # 如果出错，保留原内容
                corrected_pages.append((page_name, content))

        print(f"\n{timestamp()} 步骤1完成: 处理了 {len(corrected_pages)} 个页面")
        return corrected_pages

    def get_name(self) -> str:
        return "语法错误修正"


class ContextAwareMergeStep(PostProcessStep):
    """
    步骤2：上下文理解合并。

    使用滑动窗口方式，每页提供前后n页的上下文，进行上下文理解并合并。
    """

    def __init__(self,
                 context_before: int = 2,
                 context_after: int = 2,
                 max_context_tokens: int = 100000,
                 show_progress: bool = True):
        """
        初始化上下文合并步骤。

        Args:
            context_before: 每页前面提供的上下文页数
            context_after: 每页后面提供的上下文页数
            max_context_tokens: 最大上下文token数（避免超出模型窗口）
            show_progress: 是否显示进度
        """
        self.context_before = context_before
        self.context_after = context_after
        self.max_context_tokens = max_context_tokens
        self.show_progress = show_progress

    def _get_context_pages(self, pages: List[Tuple[str, str]], index: int,
                           n_before: int, n_after: int) -> Tuple[str, str]:
        """
        获取指定页面的前后上下文。

        Args:
            pages: 所有页面列表
            index: 当前页面索引
            n_before: 前面页数
            n_after: 后面页数

        Returns:
            Tuple[str, str]: (前面上下文, 后面上下文)
        """
        start_idx = max(0, index - n_before)
        end_idx = min(len(pages), index + 1 + n_after)

        context_before_pages = pages[start_idx:index]
        context_after_pages = pages[index + 1:end_idx]

        context_before = "\n\n---\n\n".join(
            [f"## {name}\n\n{content}" for name, content in context_before_pages])
        context_after = "\n\n---\n\n".join(
            [f"## {name}\n\n{content}" for name, content in context_after_pages])

        return context_before, context_after

    def _truncate_context(self, text: str, max_tokens: int) -> str:
        """
        截断上下文文本以符合token限制。

        Args:
            text: 原始文本
            max_tokens: 最大token数

        Returns:
            str: 截断后的文本
        """
        estimated = estimate_tokens(text)
        if estimated <= max_tokens:
            return text

        # 简单截断：保留前面的部分
        ratio = max_tokens / estimated
        target_length = int(len(text) * ratio * 0.9)  # 留10%缓冲
        return text[:target_length] + "\n\n[... 上下文已截断 ...]"

    async def process(self, pages: List[Tuple[str, str]],
                      engine: PostProcessVLLM) -> List[Tuple[str, str]]:
        """
        使用上下文理解合并页面。

        Args:
            pages: (页面名称, 页面内容) 的列表
            engine: PostProcessVLLM 引擎实例

        Returns:
            List[Tuple[str, str]]: 处理后的页面列表
        """
        print("\n" + "=" * 60)
        print(f"{timestamp()} 开始步骤2: 上下文理解合并")
        print("=" * 60)
        print(
            f"{timestamp()} 上下文窗口: 前{self.context_before}页, 后{self.context_after}页"
        )

        merged_pages = []
        total = len(pages)

        for i, (page_name, content) in enumerate(pages):
            print(f"\n{timestamp()} 处理页面 {i+1}/{total}: {page_name}")

            # 获取上下文
            context_before, context_after = self._get_context_pages(
                pages, i, self.context_before, self.context_after)

            # 估算token数并截断上下文
            current_tokens = estimate_tokens(content)
            remaining_tokens = self.max_context_tokens - current_tokens - 2000  # 留2000给prompt和输出

            if remaining_tokens > 0:
                context_before_tokens = estimate_tokens(context_before)
                context_after_tokens = estimate_tokens(context_after)

                # 按比例分配剩余token
                total_context_tokens = context_before_tokens + context_after_tokens
                if total_context_tokens > remaining_tokens:
                    if context_before_tokens > 0 and context_after_tokens > 0:
                        # 按比例截断
                        before_ratio = context_before_tokens / total_context_tokens
                        before_max = int(remaining_tokens * before_ratio)
                        after_max = remaining_tokens - before_max
                        context_before = self._truncate_context(
                            context_before, before_max)
                        context_after = self._truncate_context(
                            context_after, after_max)
                    elif context_before_tokens > 0:
                        context_before = self._truncate_context(
                            context_before, remaining_tokens)
                    elif context_after_tokens > 0:
                        context_after = self._truncate_context(
                            context_after, remaining_tokens)

            try:
                merged_content = await engine.merge_with_context_async(
                    content,
                    context_before,
                    context_after,
                    i + 1,
                    total,
                    show_progress=self.show_progress)
                merged_pages.append((page_name, merged_content))
                print(f"{timestamp()} ✓ 完成: {page_name}")
            except Exception as e:
                print(f"{timestamp()} ✗ 错误: {page_name} - {e}")
                # 如果出错，保留原内容
                merged_pages.append((page_name, content))

        print(f"\n{timestamp()} 步骤2完成: 处理了 {len(merged_pages)} 个页面")
        return merged_pages

    def get_name(self) -> str:
        return "上下文理解合并"


class PostProcessPipeline:
    """
    后处理管道。

    支持多个后处理步骤的顺序执行。
    """

    def __init__(self, engine: PostProcessVLLM):
        """
        初始化管道。

        Args:
            engine: PostProcessVLLM 引擎实例
        """
        self.engine = engine
        self.steps: List[PostProcessStep] = []

    def add_step(self, step: PostProcessStep):
        """
        添加处理步骤。

        Args:
            step: 后处理步骤实例
        """
        self.steps.append(step)
        print(f"{timestamp()} 添加处理步骤: {step.get_name()}")

    async def run(self, pages: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        运行管道，依次执行所有步骤。

        Args:
            pages: (页面名称, 页面内容) 的列表

        Returns:
            List[Tuple[str, str]]: 最终处理后的页面列表
        """
        result = pages
        for i, step in enumerate(self.steps, 1):
            print(f"\n{timestamp()} 执行步骤 {i}/{len(self.steps)}: {step.get_name()}")
            result = await step.process(result, self.engine)
        return result

    def save_result(self,
                    pages: List[Tuple[str, str]],
                    output_path: Path,
                    filename: str = "postprocessed_result.md"):
        """
        保存处理结果到文件。

        Args:
            pages: (页面名称, 页面内容) 的列表
            output_path: 输出目录路径
            filename: 输出文件名
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 合并所有页面
        merged_content = []
        for page_name, content in pages:
            merged_content.append(f"## {page_name}\n\n{content}\n")

        final_content = "\n\n---\n\n".join(merged_content)

        output_file = output_path / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_content)

        print(f"\n{timestamp()} 结果已保存到: {output_file}")
        print(f"{timestamp()} 总字符数: {len(final_content)}")


async def process_ocr_results(ocr_result_dir: Path,
                               output_dir: Path,
                               model_path: str,  # 移除默认值，强制从CLI传入
                               cuda_device: str = '0',
                               context_before: int = 1,
                               context_after: int = 1,
                               show_progress: bool = True,
                               model_preset: str = "auto",  # 需要强制指定
                               tensor_parallel_size: Optional[int] = None,
                               load_format: Optional[str] = None,
                               quant_method: Optional[str] = None,
                               enforce_eager: Optional[bool] = None):
    """
    处理OCR结果的异步主函数。

    Args:
        ocr_result_dir: OCR结果目录路径
        output_dir: 输出目录路径
        model_path: 模型路径（必需）
        cuda_device: CUDA设备编号
        context_before: 上下文窗口前页数
        context_after: 上下文窗口后页数
        show_progress: 是否显示进度
        model_preset: 模型预设（8b/70b，必需）
        tensor_parallel_size: 张量并行大小（默认按预设选择）
        load_format: 模型加载格式（可选）
        quant_method: 量化方法（可选：bitsandbytes/gptq/awq）
        enforce_eager: 强制使用eager模式（可选，会降低性能）
    """
    print("\n" + "=" * 60)
    print(f"{timestamp()} OCR后处理流程开始")
    print("=" * 60)
    print(f"{timestamp()} OCR结果目录: {ocr_result_dir}")
    print(f"{timestamp()} 输出目录: {output_dir}")

    # 初始化Token Logger
    init_token_logger(output_dir)
    token_logger = get_token_logger()
    if token_logger:
        print(f"{timestamp()} Token日志将保存到: {output_dir}/token.log")

    # 仅列出文件，避免一次性加载所有内容
    md_files = list_ocr_md_files(ocr_result_dir)
    total_pages = len(md_files)
    if total_pages == 0:
        raise ValueError(f"在 {ocr_result_dir} 中未找到任何OCR结果文件")

    # 初始化引擎
    engine = PostProcessVLLM(model_path=model_path,
                             cuda_device=cuda_device,
                             model_preset=model_preset,
                             tensor_parallel_size=tensor_parallel_size,
                             load_format=load_format,
                             quant_method=quant_method,
                             enforce_eager=enforce_eager)
    engine.initialize()

    # 输出文件（增量写入）
    output_file = Path(output_dir) / "postprocessed_result.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    f = output_file.open("w", encoding="utf-8")
    report_entries = []

    print(f"{timestamp()} 总页数: {total_pages}")
    print(f"{timestamp()} 增量写入: {output_file}")

    # 预先语法修正后面的 m 页，保持前后窗口
    def format_pages(pages: List[Tuple]) -> str:
        """兼容 (name, content) 或附带状态的 tuple，只取前两个元素。"""
        normalized = []
        for item in pages:
            if isinstance(item, tuple) and len(item) >= 2:
                name, content = item[0], item[1]
                normalized.append((name, content))
        return "\n\n---\n\n".join([f"## {n}\n\n{c}" for n, c in normalized])

    def truncate_context(text: str, max_tokens: int) -> str:
        estimated = estimate_tokens(text)
        if estimated <= max_tokens:
            return text
        ratio = max_tokens / estimated
        target_length = int(len(text) * ratio * 0.9)
        return text[:target_length] + "\n\n[... 上下文已截断 ...]"

    async def grammar_correct(path: Path) -> Tuple[str, str, str, str]:
        name = path.stem
        with open(path, "r", encoding="utf-8") as rf:
            content = rf.read()
        last_reason = ""
        for attempt in range(1, 3):
            try:
                print(
                    f"{timestamp()} [语法修正] {name} 尝试 {attempt}/2 (长度: {len(content)} 字符)"
                )
                corrected = await engine.correct_grammar_async(
                    content,
                    show_progress=show_progress,
                    page_name=name,
                )
                return name, corrected, "success", ""
            except Exception as e:
                last_reason = f"error: {e}"
                print(
                    f"{timestamp()} ⚠ 语法修正异常 {name} attempt {attempt}/2: {e}")
        print(
            f"{timestamp()} ✗ 语法修正失败/放弃 {name}，原因: {last_reason}，使用原文")
        return name, content, "fallback_original", last_reason

    # 队列：当前页 + 后续 context_after 页的“语法修正后”内容
    lookahead: Deque[Tuple[str, str, str, str]] = deque()
    file_iter = iter(md_files)

    # 预填充
    for _ in range(context_after + 1):
        try:
            path = next(file_iter)
        except StopIteration:
            break
        lookahead.append(await grammar_correct(path))

    history: Deque[Tuple[str, str]] = deque(maxlen=context_before)
    processed_count = 0

    while lookahead:
        page_name, page_content, grammar_status, grammar_reason = lookahead[0]
        processed_count += 1

        context_before_pages = list(history)
        context_after_pages = list(lookahead)[1:1 + context_after]

        # 估算上下文截断，基于模型窗口
        max_context_tokens = max(engine.max_model_len - 2000, 4000)
        current_tokens = estimate_tokens(page_content)
        remaining_tokens = max_context_tokens - current_tokens - 2000
        if remaining_tokens > 0:
            before_tokens = estimate_tokens(format_pages(context_before_pages))
            after_tokens = estimate_tokens(format_pages(context_after_pages))
            total_ctx = before_tokens + after_tokens
            if total_ctx > remaining_tokens:
                if before_tokens > 0 and after_tokens > 0:
                    before_max = int(remaining_tokens *
                                     (before_tokens / total_ctx))
                    after_max = remaining_tokens - before_max
                    context_before_text = truncate_context(
                        format_pages(context_before_pages), before_max)
                    context_after_text = truncate_context(
                        format_pages(context_after_pages), after_max)
                elif before_tokens > 0:
                    context_before_text = truncate_context(
                        format_pages(context_before_pages), remaining_tokens)
                    context_after_text = ""
                else:
                    context_before_text = ""
                    context_after_text = truncate_context(
                        format_pages(context_after_pages), remaining_tokens)
            else:
                context_before_text = format_pages(context_before_pages)
                context_after_text = format_pages(context_after_pages)
        else:
            context_before_text = format_pages(context_before_pages)
            context_after_text = format_pages(context_after_pages)

        print(f"\n{timestamp()} 处理页面 {processed_count}/{total_pages}: {page_name}")
        merge_status = "success"
        merge_reason = ""
        merged_content = page_content
        for attempt in range(1, 3):
            try:
                print(
                    f"{timestamp()} [合并] {page_name} 尝试 {attempt}/2 (前{len(context_before_text)}字 / 后{len(context_after_text)}字)"
                )
                merged_content = await engine.merge_with_context_async(
                    page_content,
                    context_before_text,
                    context_after_text,
                    processed_count,
                    total_pages,
                    show_progress=show_progress)
                break
            except Exception as e:
                merge_status = "error"
                merge_reason = str(e)
                print(
                    f"{timestamp()} ⚠ 合并异常 {page_name} attempt {attempt}/2: {e}")
        if merge_status != "success" and merge_reason:
            print(
                f"{timestamp()} ✗ 合并放弃 {page_name}，原因: {merge_reason}，使用当前内容"
            )
        else:
            print(f"{timestamp()} ✓ 完成: {page_name}")

        # 写入输出（增量）
        if processed_count > 1:
            f.write("\n\n---\n\n")
        f.write(f"## {page_name}\n\n{merged_content}\n")
        f.flush()

        report_entries.append({
            "page": page_name,
            "grammar_status": grammar_status,
            "grammar_reason": grammar_reason,
            "merge_status": merge_status,
            "merge_reason": merge_reason,
        })

        # 窗口推进
        history.append((page_name, merged_content))
        lookahead.popleft()
        try:
            path = next(file_iter)
            lookahead.append(await grammar_correct(path))
        except StopIteration:
            pass

    f.close()

    # 保存报告
    report_file = Path(output_dir) / "postprocess_report.md"
    with report_file.open("w", encoding="utf-8") as rf:
        rf.write("# 后处理报告\n\n")
        rf.write(f"- 总页数: {total_pages}\n")
        rf.write(f"- 输出文件: {output_file.name}\n\n")
        rf.write("| 页面 | 语法修正 | 原因 | 合并 | 原因 |\n")
        rf.write("| --- | --- | --- | --- | --- |\n")
        for entry in report_entries:
            rf.write(
                f"| {entry['page']} | {entry['grammar_status']} | {entry['grammar_reason']} | "
                f"{entry['merge_status']} | {entry['merge_reason']} |\n")

    print(f"\n{timestamp()} OCR后处理流程完成")
    print(f"{timestamp()} 结果已保存到: {output_file}")
    print(f"{timestamp()} 报告已保存到: {report_file}")

    # 保存Token使用统计摘要
    if token_logger:
        token_logger.save_summary()
