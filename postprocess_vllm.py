#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vLLM 后处理工具类

使用 vLLM 推理引擎运行 Llama-3.3-Swallow-70B-Instruct 模型对OCR结果进行后处理。
"""

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# vLLM 环境配置（必须在 import vllm 之前设置）
os.environ['VLLM_USE_V1'] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from token_logger import get_token_logger
from ocr_transformers import detect_and_truncate_hallucination

INFERENCE_TIMEOUT_SECONDS = 200


def timestamp():
    """返回当前时间戳字符串"""
    return datetime.now().strftime("[%H:%M:%S]")


@dataclass
class FixProposal:
    line: int
    find: str
    replace: str
    reason: str


def build_chat_prompt(system_msg: str, user_msg: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_msg}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_msg}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")


MODEL_CONFIGS = {
    "8b": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "tensor_parallel_size": 1,
        "block_size": 512,
        "max_num_batched_tokens": 1024,
        "swap_space": 32,
    },
    "32b": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "tensor_parallel_size": 1,
        "block_size": 256,
        "max_num_batched_tokens": 1024,
        "swap_space": 16,
    },
    "70b": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
        "block_size": 256,
        "max_num_batched_tokens": 1024,
        "swap_space": 16,
        "quant_config": {
            "bitsandbytes": {
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.80,
            },
            "gptq": {
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.90,
            },
            "awq": {
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.90,
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
        if enforce_eager is None:
            self.enforce_eager = False if quant_method else preset_cfg.get("enforce_eager", False)
        else:
            self.enforce_eager = enforce_eager
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    @staticmethod
    def _resolve_preset(model_path: str, model_preset: str) -> str:
        """
        根据传入的预设或模型名称选择配置。

        优先使用显式指定的 model_preset；若为 auto，则根据模型名包含“8b/32b”自动选择，默认 70b。
        """
        if model_preset and model_preset.lower() in MODEL_CONFIGS:
            return model_preset.lower()

        if "8b" in model_path.lower():
            return "8b"
        if "32b" in model_path.lower():
            return "32b"
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

        preset_cfg = MODEL_CONFIGS[self.model_preset]

        if self.quant_method and self.model_preset == "70b":
            quant_config = preset_cfg.get("quant_config", {}).get(self.quant_method, {})
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
            "max_num_seqs": 4,
            "enable_prefix_caching": False,
            "kv_cache_dtype": "fp8" if self.model_preset == "70b" else "auto",
            "disable_log_stats": True,
            "disable_sliding_window": True,
        }

        if self.load_format:
            engine_kwargs["load_format"] = self.load_format
        if self.quant_method:
            engine_kwargs["quantization"] = self.quant_method

        engine_args = AsyncEngineArgs(**engine_kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        if self.model_preset == "70b":
            self.sampling_params = SamplingParams(
                temperature=0.01,
                top_p=0.50,
                top_k=10,
                repetition_penalty=1.05,
                frequency_penalty=0.2,
                presence_penalty=0.0,
                max_tokens=512,
                skip_special_tokens=True
            )
        elif self.model_preset == "32b":
            self.sampling_params = SamplingParams(
                temperature=0.01,
                top_p=0.50,
                top_k=10,
                repetition_penalty=1.05,
                frequency_penalty=0.2,
                presence_penalty=0.0,
                max_tokens=512,
                skip_special_tokens=True
            )
        else:
            self.sampling_params = SamplingParams(
                temperature=0.01,
                top_p=0.50,
                top_k=10,
                repetition_penalty=1.05,
                frequency_penalty=0.2,
                presence_penalty=0.0,
                max_tokens=512,
                skip_special_tokens=True
            )

        print(f"{timestamp()} ✓ vLLM 引擎初始化完成！")
        print(
            f"{timestamp()} 采样参数: temp={self.sampling_params.temperature} "
            f"top_p={self.sampling_params.top_p} top_k={self.sampling_params.top_k} "
            f"rep_pen={self.sampling_params.repetition_penalty} "
            f"freq_pen={self.sampling_params.frequency_penalty} "
            f"pres_pen={self.sampling_params.presence_penalty} "
            f"max_tokens={self.sampling_params.max_tokens}")

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

        prompt_tokens = 0
        completion_tokens = 0
        last_output_token_count = 0

        try:
            async for request_output in self.engine.generate(
                    request, self.sampling_params, request_id):

                if request_output.outputs:
                    if hasattr(request_output, 'prompt_token_ids'):
                        prompt_tokens = len(request_output.prompt_token_ids)

                    output = request_output.outputs[0]
                    full_text = output.text
                    new_text = full_text[printed_length:]

                    if show_progress:
                        print(new_text, end='', flush=True)

                    printed_length = len(full_text)
                    final_output = full_text

                    if hasattr(output, 'token_ids'):
                        completion_tokens = len(output.token_ids)

                if (asyncio.get_event_loop().time() - start_time
                    ) > INFERENCE_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        f"生成超时（>{INFERENCE_TIMEOUT_SECONDS}s）")

        except Exception as e:
            if show_progress:
                print('\n')
            raise RuntimeError(
                f"vLLM 引擎生成文本时发生错误 (request_id={request_id}): {e}") from e

        elapsed = asyncio.get_event_loop().time() - start_time

        if show_progress:
            print(
                f"\n{timestamp()} 生成完成，耗时: {elapsed:.1f} 秒，输出: {len(final_output)} 字符"
            )

        token_logger = get_token_logger()
        if token_logger:
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

    async def propose_line_fixes_async(self,
                                       text: str,
                                       show_progress: bool = True,
                                       page_name: str = "",
                                       max_retries: int = 2) -> List[FixProposal]:
        """
        生成行级修正提案（JSON），不返回全文。

        Args:
            text: 待修正的文本
            show_progress: 是否显示进度
            page_name: 页面名称
            max_retries: 最大重试次数

        Returns:
            List[FixProposal]: 修正提案列表
        """
        system_msg = (
            "あなたは日本語の校正者です。以下の指示を厳守してください：\\n"
            "1. 対象は「文字化け」「スペース誤挿入」「誤字」の3種類のみ。その他の修正提案は全部PASS！\\n"
            "2. 文字化け/誤字では find と replace に完全の単語を含めて提案してくれ（replaceミスを回避のために）。\\n"
            "3. スペース誤挿入では不要な空白だけを削除・統一する。空白の前後1〜2文字を含めて find を書き（例: \"プ  レゼン\"→\"プレゼン\"）、文字自体は変えない。\\n"
            "4. 出力形式: JSON配列のみを返す。先頭/末尾に余計な文字や説明を付けない。\\n"
            "5. 配列要素の形: {\"line\": 行番号(1から), \"find\": \"誤り片段\", \"replace\": \"修正片段\", \"reason\": 理由}。\\n"
            "ご注意：\\n"
            "1. reason は「文字化け」「スペース誤挿入」「誤字」のいずれかに限定する。\\n"
            "2. 行数を変えない（改行追加・削除禁止）。改行や空白だけの変更単独は出さない。\\n"
            "3. 確信のある提案のみ、最大20件まで。確信がない場合は空配列を返す。"
        )
        user_msg = f"OCRテキスト（行番号付き）:\\n\\n{self._with_line_numbers(text)}"
        prompt = build_chat_prompt(system_msg, user_msg)

        for attempt in range(max_retries):
            result = await self._generate_async(prompt,
                                                show_progress,
                                                page_name=page_name,
                                                step_name=f"grammar_plan_{attempt + 1}")
            try:
                proposals = self._parse_fix_json(result)
                return proposals
            except Exception as e:
                if show_progress:
                    print(f"\n{timestamp()} ⚠ 修正提案解析失败: {e}")
                    if attempt < max_retries - 1:
                        print(f"{timestamp()} 正在重试... ({attempt + 2}/{max_retries})")
                continue

        if show_progress:
            print(f"\n{timestamp()} ✗ 修正提案生成失败，返回空列表")
        return []

    async def review_fixes_async(self,
                                 text: str,
                                 proposals: List[FixProposal],
                                 show_progress: bool = True,
                                 page_name: str = "",
                                 max_retries: int = 1) -> List[FixProposal]:
        """
        二次复核修正提案：仅保留符合规则的提案，输出过滤后的JSON数组。
        """
        if not proposals:
            return []

        system_msg = (
            "あなたは校正提案のレビュアーです。入力のJSON配列から、規則に反する提案を削除し、"
            "残りをそのままJSON配列で返してください。前後に余計な文字を付けないこと。\n"
            "削除すべき提案ルール:\n"
            "- reason が「文字化け」「スペース誤挿入」「誤字」以外。\n"
            "- 文字化け/誤字で find と replace の文字数差が1を超えるもの。\n"
            "- スペース誤挿入なのに空白を含まない、空白以外の文字が変わる、または空白の前後の文字が find に含まれていないもの。\n"
            "その他の提案を全部PASSしてください。\n"
            "出力はフィルタ後のJSON配列（同じキー: line, find, replace, reason）のみ。"
        )

        proposals_json = json.dumps([p.__dict__ for p in proposals], ensure_ascii=False)
        user_msg = (f"OCRテキスト（参照用）:\n{self._with_line_numbers(text)}\n\n"
                    f"初回提案(JSON配列):\n{proposals_json}")
        prompt = build_chat_prompt(system_msg, user_msg)

        last_error = ""
        for attempt in range(max_retries):
            result = await self._generate_async(prompt,
                                                show_progress,
                                                page_name=page_name,
                                                step_name=f"review_plan_{attempt + 1}")
            try:
                reviewed = self._parse_fix_json(result)
                return reviewed
            except Exception as e:
                last_error = str(e)
                if show_progress:
                    print(f"\n{timestamp()} ⚠ 修正提案レビューパース失敗: {e}")
        if show_progress:
            print(f"{timestamp()} ✗ 修正提案レビューフォールバック（使用初回提案）: {last_error}")
        return proposals

    def _sanitize_output(self, text: str, page_name: str,
                         step_name: str) -> str:
        """
        简单校验输出：如出现代码块等不应输出的结构，抛出异常让上层重试。
        """
        fence_idx = text.find("```")
        if fence_idx != -1:
            raise RuntimeError(
                f"检测到代码块输出，触发重试 (step={step_name} page={page_name})")
        return text.strip()

    def _with_line_numbers(self, text: str) -> str:
        """为提示添加行号，便于模型输出定位。"""
        lines = text.splitlines()
        return "\n".join([f"{idx+1}: {line}" for idx, line in enumerate(lines)])

    def _parse_fix_json(self, raw: str) -> List[FixProposal]:
        """解析模型返回的JSON提案列表。"""
        cleaned = remove_eot_tokens(raw)
        data = json.loads(cleaned)
        proposals: List[FixProposal] = []
        if not isinstance(data, list):
            raise ValueError("期望JSON数组")
        for item in data:
            if not isinstance(item, dict):
                continue
            line = int(item.get("line", 0))
            find = str(item.get("find", "") or "")
            replace = str(item.get("replace", "") or "")
            reason = str(item.get("reason", "") or "")
            if find.strip() == "" or find.strip() == "\\n":
                continue
            if replace.strip() == "" and find.strip() in {"", "\\n"}:
                continue
            if line > 0 and find:
                proposals.append(FixProposal(line=line, find=find, replace=replace, reason=reason))
        return proposals


def page_name_from_content(text: str) -> str:
    """尽量从内容生成短标签，避免空 page 名称影响日志"""
    head = re.sub(r'\s+', '', text)[:8]
    return head if head else "page"

def remove_eot_tokens(text: str) -> str:
    """移除生成中残留的特殊标记（含缺失 '>' 变体），并顺带清理其周围的空白行。"""
    return re.sub(r'\s*<\|eot_id\|?>\s*', '', text)

def remove_image_tags(text: str) -> str:
    """移除形如 ![](images/xxx.jpg) 的图片标签。"""
    return re.sub(r'!\[[^\]]*?\]\(images/[^)]+\)', '', text)


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

    md_files = []
    for subdir in sorted(ocr_result_dir.iterdir()):
        if subdir.is_dir():
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
    jp_ch_count = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
    en_count = len(re.findall(r'[a-zA-Z]+', text))
    return jp_ch_count + en_count // 4 + len(text) // 10  # 加上一些缓冲
