#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
后处理管道模块

提供可扩展的后处理管道架构，支持多步骤处理。
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from postprocess_vllm import (PostProcessVLLM, list_ocr_md_files, load_ocr_pages,
                              timestamp, remove_eot_tokens, remove_image_tags)
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
                corrected_pages.append((page_name, content))

        print(f"\n{timestamp()} 步骤1完成: 处理了 {len(corrected_pages)} 个页面")
        return corrected_pages

    def get_name(self) -> str:
        return "语法错误修正"


class ContextAwareMergeStep(PostProcessStep):
    """
    步骤2：合并（仅清洗，不再调用LLM）。
    """

    def __init__(self, show_progress: bool = True):
        """
        初始化合并步骤。

        Args:
            show_progress: 是否显示进度
        """
        self.show_progress = show_progress

    async def process(self, pages: List[Tuple[str, str]],
                      engine: PostProcessVLLM) -> List[Tuple[str, str]]:
        """
        合并页面内容（LLM已关闭，仅做清洗）。

        Args:
            pages: (页面名称, 页面内容) 的列表
            engine: PostProcessVLLM 引擎实例

        Returns:
            List[Tuple[str, str]]: 处理后的页面列表
        """
        print("\n" + "=" * 60)
        print(f"{timestamp()} 开始步骤2: 合并（跳过LLM，仅清洗）")
        print("=" * 60)

        merged_pages = []
        total = len(pages)

        for i, (page_name, content) in enumerate(pages):
            print(f"\n{timestamp()} 处理页面 {i+1}/{total}: {page_name}")

            merged_pages.append((page_name, content))
            print(f"{timestamp()} ✓ 完成: {page_name}")

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

        merged_content = []
        for _, content in pages:
            merged_content.append(f"{content.strip()}\n")

        final_content = "\n\n".join(merged_content)

        output_file = output_path / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_content)

        print(f"\n{timestamp()} 结果已保存到: {output_file}")
        print(f"{timestamp()} 总字符数: {len(final_content)}")


async def process_ocr_results(ocr_result_dir: Path,
                               output_dir: Path,
                               model_path: str,  # 移除默认值，强制从CLI传入
                               cuda_device: str = '0',
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

    init_token_logger(output_dir)
    token_logger = get_token_logger()
    if token_logger:
        print(f"{timestamp()} Token日志将保存到: {output_dir}/token.log")

    md_files = list_ocr_md_files(ocr_result_dir)
    total_pages = len(md_files)
    if total_pages == 0:
        raise ValueError(f"在 {ocr_result_dir} 中未找到任何OCR结果文件")

    engine = PostProcessVLLM(model_path=model_path,
                             cuda_device=cuda_device,
                             model_preset=model_preset,
                             tensor_parallel_size=tensor_parallel_size,
                             load_format=load_format,
                             quant_method=quant_method,
                             enforce_eager=enforce_eager)
    engine.initialize()

    output_file = Path(output_dir) / "postprocessed_result.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    f = output_file.open("w", encoding="utf-8")
    report_entries = []
    raw_fix_log_path = Path(output_dir) / "postprocess_fixes_raw.jsonl"
    reviewed_fix_log_path = Path(output_dir) / "postprocess_fixes.jsonl"
    raw_fix_log = raw_fix_log_path.open("w", encoding="utf-8")
    reviewed_fix_log = reviewed_fix_log_path.open("w", encoding="utf-8")

    print(f"{timestamp()} 总页数: {total_pages}")
    print(f"{timestamp()} 增量写入: {output_file}")

    REASON_ALLOWLIST = {"文字化け", "スペース誤挿入", "誤字"}

    def is_valid_space_fix(find: str, replace: str) -> bool:
        """仅允许删除/归一化空白，且包含空白前后的上下文。"""
        has_context = re.search(r'\S\s+\S', find) is not None
        same_without_space = re.sub(r'\s+', '', find) == re.sub(r'\s+', '', replace)
        return has_context and same_without_space

    def apply_line_fixes(text: str, fixes) -> Tuple[str, int, List[str], List]:
        lines = text.splitlines()
        errors = []
        applied = 0
        kept = []
        for fix in fixes:
            reason_text = getattr(fix, "reason", "")
            if not reason_text or reason_text not in REASON_ALLOWLIST:
                errors.append(f"skip_reason_not_allowed_line_{fix.line}")
                continue
            if reason_text in {"文字化け", "誤字"} and abs(len(fix.replace) - len(fix.find)) > 1:
                errors.append(f"skip_length_diff_line_{fix.line}")
                continue
            if reason_text == "スペース誤挿入" and not is_valid_space_fix(fix.find, fix.replace):
                errors.append(f"skip_space_validation_line_{fix.line}")
                continue
            if "<" in fix.find or ">" in fix.find or "<" in fix.replace or ">" in fix.replace:
                errors.append(f"skip_html_change_line_{fix.line}")
                continue
            if "http://" in fix.find or "https://" in fix.find or "http://" in fix.replace or "https://" in fix.replace:
                errors.append(f"skip_url_line_{fix.line}")
                continue
            line_idx = fix.line - 1
            if line_idx < 0 or line_idx >= len(lines):
                errors.append(f"line_out_of_range:{fix.line}")
                continue
            line = lines[line_idx]
            if fix.find not in line:
                errors.append(f"not_found_line_{fix.line}")
                continue
            lines[line_idx] = line.replace(fix.find, fix.replace, 1)
            applied += 1
            kept.append(fix)
        return "\n".join(lines), applied, errors, kept

    async def grammar_correct(path: Path) -> Tuple[str, str, str, str]:
        name = path.stem
        with open(path, "r", encoding="utf-8") as rf:
            content = rf.read()
        last_reason = ""
        for attempt in range(1, 3):
            try:
                print(f"{timestamp()} [修正提案] {name} 尝试 {attempt}/2 (长度: {len(content)} 字符)")
                fixes = await engine.propose_line_fixes_async(
                    content,
                    show_progress=show_progress,
                    page_name=name,
                )
                raw_fix_log.write(json.dumps({
                    "page": name,
                    "status": "raw",
                    "proposals": [fix.__dict__ for fix in fixes],
                }, ensure_ascii=False) + "\n")
                raw_fix_log.flush()
                prefilter_errors = []
                filtered_for_review = []
                for fix in fixes:
                    r = getattr(fix, "reason", "")
                    if not r or r not in REASON_ALLOWLIST:
                        prefilter_errors.append(f"prefilter_skip_reason_line_{fix.line}")
                        continue
                    if r in {"文字化け", "誤字"} and abs(len(fix.replace) - len(fix.find)) > 1:
                        prefilter_errors.append(f"prefilter_skip_length_line_{fix.line}")
                        continue
                    if r == "スペース誤挿入" and not is_valid_space_fix(fix.find, fix.replace):
                        prefilter_errors.append(f"prefilter_skip_space_line_{fix.line}")
                        continue
                    filtered_for_review.append(fix)

                reviewed_fixes = await engine.review_fixes_async(
                    content,
                    filtered_for_review,
                    show_progress=show_progress,
                    page_name=name,
                )
                corrected, applied, errors, kept = apply_line_fixes(
                    content, reviewed_fixes)
                errors = prefilter_errors + errors
                reviewed_fix_log.write(json.dumps({
                    "page": name,
                    "status": "reviewed",
                    "applied": applied,
                    "errors": errors,
                    "proposals": [fix.__dict__ for fix in kept],
                }, ensure_ascii=False) + "\n")
                reviewed_fix_log.flush()
                return name, corrected, "success", ""
            except Exception as e:
                last_reason = f"error: {e}"
                print(
                    f"{timestamp()} ⚠ 语法修正异常 {name} attempt {attempt}/2: {e}")
        print(
            f"{timestamp()} ✗ 语法修正失败/放弃 {name}，原因: {last_reason}，使用原文")
        raw_fix_log.write(json.dumps({
            "page": name,
            "status": "raw",
            "proposals": [],
            "errors": [last_reason],
        }, ensure_ascii=False) + "\n")
        raw_fix_log.flush()
        reviewed_fix_log.write(json.dumps({
            "page": name,
            "status": "fallback_original",
            "applied": 0,
            "errors": [last_reason],
            "proposals": [],
        }, ensure_ascii=False) + "\n")
        reviewed_fix_log.flush()
        return name, remove_eot_tokens(content), "fallback_original", last_reason

    processed_count = 0

    for path in md_files:
        page_name, page_content, grammar_status, grammar_reason = await grammar_correct(path)
        processed_count += 1

        print(f"\n{timestamp()} 处理页面 {processed_count}/{total_pages}: {page_name}")
        merged_content = remove_image_tags(remove_eot_tokens(page_content))
        print(f"{timestamp()} [合并] 跳过LLM，清理图片标签并写入纠错结果")

        if processed_count > 1:
            f.write("\n\n")
        f.write(f"{merged_content}\n")
        f.flush()

        report_entries.append({
            "page": page_name,
            "grammar_status": grammar_status,
            "grammar_reason": grammar_reason,
            "merge_status": "skipped_llm",
            "merge_reason": "merge skipped; using grammar output",
        })

    f.close()
    raw_fix_log.close()
    reviewed_fix_log.close()

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

    if token_logger:
        token_logger.save_summary()
