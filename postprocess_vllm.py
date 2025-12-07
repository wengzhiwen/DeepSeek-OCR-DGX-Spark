#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vLLM 后处理工具类

使用 vLLM 推理引擎运行 Llama-3.3-Swallow-70B-Instruct 模型对OCR结果进行后处理。
"""

import asyncio
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# vLLM 环境配置（必须在 import vllm 之前设置）
os.environ['VLLM_USE_V1'] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

# 导入token logger
from token_logger import get_token_logger

# 导入OCR模块中的幻觉检测函数
from ocr_transformers import detect_and_truncate_hallucination


# 单页生成超时时间（秒）
INFERENCE_TIMEOUT_SECONDS = 200


def timestamp():
    """返回当前时间戳字符串"""
    return datetime.now().strftime("[%H:%M:%S]")


def build_chat_prompt(system_msg: str, user_msg: str) -> str:
    """
    按 Llama3/Swallow 聊天模板构建提示，明确区分 system / user。
    """
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_msg}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_msg}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")


MODEL_CONFIGS = {
    # 8B 调试配置：中等上下文 + 较高显存利用，兼顾速度与稳定
    "8b": {
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.90,
        "tensor_parallel_size": 1,
        "block_size": 512,
        "max_num_batched_tokens": 2048,
        "swap_space": 32,
    },
    # 70B OCR后处理配置：针对单页处理优化
    "70b": {
        "max_model_len": 8192,  # 8K支持最多9页上下文（每页约800-1000 tokens）
        "gpu_memory_utilization": 0.85,  # 85%充分利用119GB GPU内存
        "tensor_parallel_size": 1,
        "block_size": 256,  # 小block size适合短文本
        "max_num_batched_tokens": 1024,  # 减少批处理，适合OCR
        "swap_space": 16,  # 适中的swap空间
        # 量化配置优化
        "quant_config": {
            "bitsandbytes": {
                "max_model_len": 8192,  # 8K for int8 quantization（保持与GPTQ/AWQ一致）
                "gpu_memory_utilization": 0.80,  # 80% for quantized model
            },
            "gptq": {
                "max_model_len": 8192,  # Full 8K for GPTQ
                "gpu_memory_utilization": 0.90,  # Higher utilization for GPTQ
            },
            "awq": {
                "max_model_len": 8192,  # Full 8K for AWQ
                "gpu_memory_utilization": 0.90,  # Higher utilization for AWQ
            },
        }
    },
}


class PostProcessVLLM:
    """
    vLLM 后处理工具类。

    使用 vLLM 异步引擎，支持对OCR结果进行多步骤后处理。
    """

    def __init__(self,
                 model_path='llama3.3-swallow-70b/models/Llama-3.3-Swallow-70B-Instruct-v0.4',
                 cuda_device='0',
                 max_model_len: Optional[int] = None,
                 gpu_memory_utilization: Optional[float] = None,
                 tensor_parallel_size: Optional[int] = None,
                 model_preset: str = "auto",
                 load_format: Optional[str] = None,
                 quant_method: Optional[str] = None,
                 enforce_eager: Optional[bool] = None):
        """
        初始化 vLLM 后处理引擎。

        Args:
            model_path: 模型路径（HuggingFace Hub 或本地路径）
            cuda_device: CUDA 设备编号
            max_model_len: 最大模型长度（tokens），默认按预设
            gpu_memory_utilization: 显存利用率（0-1）
            tensor_parallel_size: 张量并行大小（默认按预设选择）
            model_preset: 预设名称（auto/8b/70b）
            load_format: 加载格式（auto/safetensors/bitsandbytes/gptq/awq）
            quant_method: 量化方法（None/bitsandbytes/gptq/awq）
            enforce_eager: 是否使用eager模式（默认按预设）
        """
        self.model_path = model_path
        self.engine = None
        self.sampling_params = None
        self.cuda_device = cuda_device
        self.model_preset = self._resolve_preset(model_path, model_preset)
        self.load_format = load_format
        self.quant_method = quant_method
        preset_cfg = MODEL_CONFIGS[self.model_preset]
        self.max_model_len = max_model_len if max_model_len is not None else preset_cfg[
            "max_model_len"]
        self.gpu_memory_utilization = gpu_memory_utilization if gpu_memory_utilization is not None else preset_cfg[
            "gpu_memory_utilization"]
        self.tensor_parallel_size = tensor_parallel_size if tensor_parallel_size is not None else preset_cfg[
            "tensor_parallel_size"]
        # 对于量化模型，默认不使用eager模式以提升性能
        if enforce_eager is None:
            # 量化模型默认使用优化编译，如果遇到问题再手动启用eager
            self.enforce_eager = False if quant_method else preset_cfg.get("enforce_eager", False)
        else:
            self.enforce_eager = enforce_eager
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    @staticmethod
    def _resolve_preset(model_path: str, model_preset: str) -> str:
        """
        根据传入的预设或模型名称选择配置。

        优先使用显式指定的 model_preset；若为 auto，则根据模型名包含“8b”自动选择 8b，否则默认 70b。
        """
        if model_preset and model_preset.lower() in MODEL_CONFIGS:
            return model_preset.lower()

        # auto: 根据模型名猜测
        if "8b" in model_path.lower():
            return "8b"
        return "70b"

    def initialize(self):
        """
        初始化 vLLM 异步引擎。

        引擎只需要初始化一次，之后可以连续处理多个任务。
        """
        print("\n" + "=" * 60)
        print(f"{timestamp()} 正在初始化 vLLM 后处理引擎...")
        print("=" * 60)
        print(f"{timestamp()} 模型: {self.model_path}")
        print(f"{timestamp()} 最大上下文长度: {self.max_model_len} tokens")
        print(
            f"{timestamp()} 预设: {self.model_preset}, 显存利用率: {self.gpu_memory_utilization}, TP: {self.tensor_parallel_size}"
        )
        if self.load_format:
            print(f"{timestamp()} 加载格式: {self.load_format}")
        if self.quant_method:
            print(f"{timestamp()} 量化方法: {self.quant_method}")
            if self.quant_method == "bitsandbytes":
                print(f"{timestamp()} 提示: int8量化可减少约43%内存使用")
            elif self.quant_method in ["gptq", "awq"]:
                print(f"{timestamp()} 提示: 4bit量化可减少约68%内存使用")
        print(f"{timestamp()} Eager模式: {self.enforce_eager}")

        # 基础引擎参数
        preset_cfg = MODEL_CONFIGS[self.model_preset]

        # 根据量化方法调整参数
        if self.quant_method and self.model_preset == "70b":
            quant_config = preset_cfg.get("quant_config", {}).get(self.quant_method, {})
            # 使用量化特定的配置，如果没有则使用默认值
            actual_max_model_len = self.max_model_len or quant_config.get("max_model_len", preset_cfg["max_model_len"])
            actual_gpu_util = self.gpu_memory_utilization or quant_config.get("gpu_memory_utilization", preset_cfg["gpu_memory_utilization"])
        else:
            actual_max_model_len = self.max_model_len
            actual_gpu_util = self.gpu_memory_utilization

        engine_kwargs = {
            "model": self.model_path,
            "block_size": preset_cfg.get("block_size", 512),
            "max_model_len": actual_max_model_len or preset_cfg["max_model_len"],
            "enforce_eager": self.enforce_eager,
            "trust_remote_code": True,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": actual_gpu_util or preset_cfg["gpu_memory_utilization"],
            "swap_space": preset_cfg.get("swap_space", 32),
            "max_num_batched_tokens": preset_cfg.get("max_num_batched_tokens", 2048),
            "max_num_seqs": 4 if self.model_preset == "70b" else 4,  # 70B保守并发数避免OOM
            "enable_prefix_caching": False,  # 禁用prefix caching以节省内存
            "kv_cache_dtype": "fp8" if self.model_preset == "70b" else "auto",  # 使用FP8量化KV cache
            "disable_log_stats": True,  # 禁用统计日志，减少开销
            "disable_sliding_window": True,  # 禁用滑动窗口，OCR不需要
        }

        # 添加量化相关参数
        if self.load_format:
            engine_kwargs["load_format"] = self.load_format
        if self.quant_method:
            engine_kwargs["quantization"] = self.quant_method

        engine_args = AsyncEngineArgs(**engine_kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # 根据模型预设调整采样参数
        if self.model_preset == "70b":
            # 70B模型的采样参数：强化重复抑制，减少幻觉
            self.sampling_params = SamplingParams(
                temperature=0.05,  # 更低的温度，减少随机性
                top_p=0.85,  # 更集中的概率分布
                top_k=40,  # 减少候选token数量
                repetition_penalty=1.3,  # 加强重复惩罚
                frequency_penalty=1.0,  # 加强频率惩罚，避免重复词汇
                presence_penalty=0.5,  # 增加存在惩罚，鼓励新内容
                max_tokens=1200,  # 保持最大输出长度
                skip_special_tokens=False
            )
        else:
            # 8B模型的采样参数：默认配置
            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.85,
                top_k=50,
                repetition_penalty=1.2,
                frequency_penalty=0.6,
                presence_penalty=0.4,
                max_tokens=800,
                skip_special_tokens=False
            )

        print(f"{timestamp()} ✓ vLLM 引擎初始化完成！")
        print(
            f"{timestamp()} 采样参数: temp={self.sampling_params.temperature} "
            f"top_p={self.sampling_params.top_p} top_k={self.sampling_params.top_k} "
            f"rep_pen={self.sampling_params.repetition_penalty} "
            f"freq_pen={self.sampling_params.frequency_penalty} "
            f"pres_pen={self.sampling_params.presence_penalty} "
            f"max_tokens={self.sampling_params.max_tokens}")

        if self.model_preset == "70b":
            print(f"{timestamp()} 70B模型: 已启用强化重复抑制参数")

    async def _generate_async(self,
                              prompt: str,
                              show_progress: bool = True,
                              page_name: str = "",
                              step_name: str = "") -> str:
        """
        异步生成文本（内部方法）。

        Args:
            prompt: 输入提示词
            show_progress: 是否显示流式输出进度

        Returns:
            str: 生成的文本结果
        """
        if self.engine is None:
            raise RuntimeError("引擎未初始化，请先调用 initialize() 方法")

        request = {"prompt": prompt}
        request_id = f"postprocess-{uuid.uuid4().hex[:8]}"

        if show_progress:
            print(f"\n{timestamp()} 开始生成: step={step_name} page={page_name}")

        printed_length = 0
        final_output = ""
        start_time = asyncio.get_event_loop().time()

        # 用于跟踪token使用情况
        prompt_tokens = 0
        completion_tokens = 0
        last_output_token_count = 0

        try:
            async for request_output in self.engine.generate(
                    request, self.sampling_params, request_id):

                if request_output.outputs:
                    # 获取token使用信息
                    if hasattr(request_output, 'prompt_token_ids'):
                        prompt_tokens = len(request_output.prompt_token_ids)

                    output = request_output.outputs[0]
                    full_text = output.text
                    new_text = full_text[printed_length:]

                    if show_progress:
                        print(new_text, end='', flush=True)

                    printed_length = len(full_text)
                    final_output = full_text

                    # 获取输出的token数量
                    if hasattr(output, 'token_ids'):
                        completion_tokens = len(output.token_ids)

                # 超时保护
                if (asyncio.get_event_loop().time() - start_time
                    ) > INFERENCE_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        f"生成超时（>{INFERENCE_TIMEOUT_SECONDS}s）")

        except Exception as e:
            if show_progress:
                print('\n')
            raise RuntimeError(
                f"vLLM 引擎生成文本时发生错误 (request_id={request_id}): {e}") from e

        # 计算耗时
        elapsed = asyncio.get_event_loop().time() - start_time

        if show_progress:
            print(
                f"\n{timestamp()} 生成完成，耗时: {elapsed:.1f} 秒，输出: {len(final_output)} 字符"
            )

        # 记录token使用情况
        token_logger = get_token_logger()
        if token_logger:
            # 如果vLLM没有提供准确的token计数，使用估算值
            if prompt_tokens == 0:
                prompt_tokens = estimate_tokens(prompt)
            if completion_tokens == 0:
                completion_tokens = estimate_tokens(final_output)

            token_logger.log_request(
                request_id=request_id,
                step_name=step_name,
                page_name=page_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_sec=elapsed
            )

        return final_output

    async def correct_grammar_async(self,
                                     text: str,
                                     show_progress: bool = True,
                                     page_name: str = "",
                                     max_retries: int = 2) -> str:
        """
        修正日语语法错误。

        只修正明显的语法错误，如果无法确定正确内容则不进行修正。

        Args:
            text: 待修正的文本
            show_progress: 是否显示进度
            page_name: 页面名称
            max_retries: 最大重试次数

        Returns:
            str: 修正后的文本
        """
        system_msg = ("あなたは日本語の専門家です。明らかな文法エラーのみを修正し、"
                      "内容を推測・補完・書き換えせず、修正版だけを1回だけ出力します。"
                      "説明文やコードブロック( ```)・ラベルは一切書かないでください。"
                      "段落順を維持し、同じ文を繰り返さないでください。")
        user_msg = f"元テキストをそのまま修正してください。\n\n{paragraph_count_hint(text)}\n\n{text}"
        prompt = build_chat_prompt(system_msg, user_msg)

        # 重试机制
        for attempt in range(max_retries):
            result = await self._generate_async(prompt,
                                                show_progress,
                                                page_name=page_name,
                                                step_name=f"grammar_attempt_{attempt + 1}")

            # 检测幻觉
            cleaned_result, has_hallucination, pattern = detect_and_truncate_hallucination(result)

            if has_hallucination:
                if show_progress:
                    print(f"\n{timestamp()} ⚠ 语法修正检测到幻觉: {pattern}")
                    if attempt < max_retries - 1:
                        print(f"{timestamp()} 正在重试... ({attempt + 2}/{max_retries})")
                # 如果还有重试机会，继续循环
                continue
            else:
                # 没有幻觉，返回结果
                return self._sanitize_output(cleaned_result, page_name, "grammar")

        # 所有重试都失败，使用最后一次的结果（或原文）
        if show_progress:
            print(f"\n{timestamp()} ✗ 语法修正重试次数已达上限，使用原文")
        return text

    async def merge_with_context_async(self,
                                       current_page: str,
                                       context_before: str,
                                       context_after: str,
                                       page_num: int,
                                       total_pages: int,
                                       show_progress: bool = True,
                                       max_retries: int = 2) -> str:
        """
        使用上下文理解合并页面内容。

        Args:
            current_page: 当前页面内容
            context_before: 前面页面的上下文
            context_after: 后面页面的上下文
            page_num: 当前页码
            total_pages: 总页数
            show_progress: 是否显示进度
            max_retries: 最大重试次数

        Returns:
            str: 处理后的页面内容
        """
        system_msg = ("あなたは文書編集の専門家です。前後の文脈を考慮しつつ、"
                      "現在のページを整形してください。内容を推測・補完・書き換えせず、"
                      "整形した本文だけを1回だけ出力します。説明文やコードブロック( ```)・"
                      "ラベルは一切書かないでください。段落順を維持し、同じ文を繰り返さないでください。"
                      "標準的なMarkdownのみを使用してください、但し「```Markdown」マークはいらない 。")
        user_msg = (f"文書の位置: {page_num}/{total_pages} ページ目\n\n"
                    f"前のページの内容（参考）:\n{context_before if context_before else '(なし)'}\n\n"
                    f"現在のページの内容:\n{current_page}\n\n"
                    f"後のページの内容（参考）:\n{context_after if context_after else '(なし)'}")
        prompt = build_chat_prompt(system_msg, user_msg)

        page_tag = f"{page_num}_{page_name_from_content(current_page)}"

        # 重试机制
        for attempt in range(max_retries):
            result = await self._generate_async(prompt,
                                                show_progress,
                                                page_name=page_tag,
                                                step_name=f"merge_attempt_{attempt + 1}")

            # 检测幻觉
            cleaned_result, has_hallucination, pattern = detect_and_truncate_hallucination(result)

            if has_hallucination:
                if show_progress:
                    print(f"\n{timestamp()} ⚠ 合并处理检测到幻觉: {pattern}")
                    if attempt < max_retries - 1:
                        print(f"{timestamp()} 正在重试... ({attempt + 2}/{max_retries})")
                # 如果还有重试机会，继续循环
                continue
            else:
                # 没有幻觉，返回结果
                try:
                    return self._sanitize_output(cleaned_result, page_tag, "merge")
                except RuntimeError as e:
                    # 检测到代码块，也需要重试
                    if show_progress and attempt < max_retries - 1:
                        print(f"\n{timestamp()} ⚠ {e}")
                        print(f"{timestamp()} 正在重试... ({attempt + 2}/{max_retries})")
                    continue

        # 所有重试都失败，使用原文
        if show_progress:
            print(f"\n{timestamp()} ✗ 合并处理重试次数已达上限，使用原文")
        return current_page

    def _sanitize_output(self, text: str, page_name: str,
                         step_name: str) -> str:
        """
        简单校验输出：如出现代码块等不应输出的结构，抛出异常让上层重试。
        """
        cleaned = text
        fence_idx = cleaned.find("```")
        if fence_idx != -1:
            raise RuntimeError(
                f"检测到代码块输出，触发重试 (step={step_name} page={page_name})")
        return cleaned.strip()


def page_name_from_content(text: str) -> str:
    """尽量从内容生成短标签，避免空 page 名称影响日志"""
    head = re.sub(r'\s+', '', text)[:8]
    return head if head else "page"


def paragraph_count_hint(text: str) -> str:
    """给模型一个段落数量提示，减少重复/遗漏。"""
    para_count = len([p for p in text.splitlines() if p.strip()])
    return f"(段落数の目安: {para_count} 行)"


def load_ocr_pages(ocr_result_dir: Path) -> List[Tuple[str, str]]:
    """
    从OCR结果目录加载所有页面的markdown文件。

    Args:
        ocr_result_dir: OCR结果目录路径

    Returns:
        List[Tuple[str, str]]: (页面名称, 页面内容) 的列表，按文件名排序
    """
    pages = []
    ocr_result_dir = Path(ocr_result_dir)

    # 查找所有包含 .md 文件的子目录
    md_files = []
    for subdir in sorted(ocr_result_dir.iterdir()):
        if subdir.is_dir():
            # 查找该目录下的 .md 文件（排除 _ori.mmd 等）
            for md_file in sorted(subdir.glob("*.md")):
                if "_ori" not in md_file.stem and "_with_boxes" not in md_file.stem:
                    md_files.append(md_file)
                    break

    # 如果子目录中没有找到，直接在根目录查找
    if not md_files:
        md_files = sorted(ocr_result_dir.glob("*.md"))
        md_files = [
            f for f in md_files
            if "_ori" not in f.stem and "_with_boxes" not in f.stem
        ]

    # 按文件名排序（确保页面顺序正确）
    md_files.sort(key=lambda x: x.name)

    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                page_name = md_file.stem
                pages.append((page_name, content))
        except Exception as e:
            print(f"{timestamp()} ⚠ 无法读取文件 {md_file}: {e}")

    print(f"{timestamp()} 加载了 {len(pages)} 个页面")
    return pages


def list_ocr_md_files(ocr_result_dir: Path) -> List[Path]:
    """
    仅列出要处理的 markdown 文件路径（不加载内容），按文件名排序。

    Args:
        ocr_result_dir: OCR结果目录路径

    Returns:
        List[Path]: markdown 文件路径列表
    """
    ocr_result_dir = Path(ocr_result_dir)
    md_files = []

    for subdir in sorted(ocr_result_dir.iterdir()):
        if subdir.is_dir():
            if subdir.name == "postprocessed":
                continue
            for md_file in sorted(subdir.glob("*.md")):
                if "_ori" not in md_file.stem and "_with_boxes" not in md_file.stem:
                    md_files.append(md_file)
                    break

    if not md_files:
        md_files = sorted(ocr_result_dir.glob("*.md"))
        md_files = [
            f for f in md_files
            if "_ori" not in f.stem and "_with_boxes" not in f.stem
        ]

    md_files.sort(key=lambda x: x.name)
    return md_files


def estimate_tokens(text: str) -> int:
    """
    粗略估算文本的token数量（日语和中文大约1字符=1token，英文约4字符=1token）。

    Args:
        text: 文本内容

    Returns:
        int: 估算的token数量
    """
    # 简单估算：日语/中文字符数 + 英文单词数/4
    jp_ch_count = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
    en_count = len(re.findall(r'[a-zA-Z]+', text))
    return jp_ch_count + en_count // 4 + len(text) // 10  # 加上一些缓冲
