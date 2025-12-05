#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vLLM OCR 工具类

使用 vLLM 推理引擎运行 DeepSeek-OCR 模型。
vLLM 默认使用 Gundam 模式（base_size=1024, image_size=640, crop_mode=True）。
"""

import asyncio
import os
import re
import uuid
import zlib
from datetime import datetime
from pathlib import Path

# vLLM 环境配置（必须在 import vllm 之前设置）
os.environ['VLLM_USE_V1'] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

from ocr_utils import load_image, save_ocr_results

# ============================================================================
# 幻觉检测配置
# ============================================================================
STREAM_CHECK_INTERVAL = 500  # 每生成多少字符检查一次幻觉
STREAM_CHECK_START = 2000  # 至少生成多少字符后才开始检查
MAX_OUTPUT_LENGTH = 50000  # 最大输出字符数
INFERENCE_TIMEOUT_SECONDS = 300  # 单张图片推理超时时间（秒）

# NGram 重复抑制参数（通过 extra_args 传递）
NGRAM_SIZE = 30
WINDOW_SIZE = 90
WHITELIST_TOKEN_IDS = [128821, 128822]


def timestamp():
    """返回当前时间戳字符串"""
    return datetime.now().strftime("[%H:%M:%S]")


def detect_hallucination_in_stream(text):
    """
    在流式输出中检测幻觉模式（通用算法，不依赖特定标签）。
    
    核心原理：幻觉的本质是"异常重复"。使用以下通用方法检测：
    1. 压缩比检测：重复内容压缩后体积极小
    2. 滑动窗口 N-gram：检测任意长度的高频重复子串
    3. 连续重复块：检测 ABCABCABC 模式
    4. 长度限制：兜底保护
    
    Args:
        text: 当前已生成的文本
        
    Returns:
        tuple: (是否检测到幻觉, 幻觉模式描述)
    """
    # 如果文本太短，不检测
    if len(text) < STREAM_CHECK_START:
        return False, None

    # ========== 方法1: 压缩比检测（最通用）==========
    # 正常文本压缩比约 0.3-0.6，重复文本压缩比 < 0.1
    text_bytes = text.encode('utf-8')
    if len(text_bytes) > 2000:
        compressed = zlib.compress(text_bytes, level=1)  # 快速压缩
        compression_ratio = len(compressed) / len(text_bytes)

        # 文本越长，对压缩比要求越严格
        if len(text) > 10000 and compression_ratio < 0.12:
            return True, f"压缩比异常低: {compression_ratio:.3f}"
        if len(text) > 5000 and compression_ratio < 0.08:
            return True, f"压缩比极低: {compression_ratio:.3f}"

    # ========== 方法2: 滑动窗口 N-gram 高频检测 ==========
    # 检测最近文本中是否有任意子串异常高频出现
    if len(text) > 3000:
        recent = text[-3000:]

        # 检测多种长度的重复模式
        for n in [12, 20, 30, 50]:
            if len(recent) < n * 8:
                continue

            ngrams = {}
            for i in range(len(recent) - n):
                gram = recent[i:i + n]
                ngrams[gram] = ngrams.get(gram, 0) + 1

            if ngrams:
                max_gram, max_count = max(ngrams.items(), key=lambda x: x[1])
                # 计算覆盖率：该子串的所有出现共占据的字符比例
                coverage = (max_count * n) / len(recent)

                # 如果单个子串覆盖超过 25%，是幻觉
                if coverage > 0.25 and max_count >= 8:
                    display = re.sub(r'\s+', ' ', max_gram)[:30]
                    return True, f"重复子串(len={n}): \"{display}\" x{max_count}"

    # ========== 方法3: 连续重复块检测 ==========
    # 检测紧邻的相同块，如 [ABC][ABC][ABC]
    if len(text) > 1500:
        recent = text[-1500:]
        for block_len in range(15, 80, 5):
            if len(recent) < block_len * 6:
                continue

            i = 0
            while i + block_len * 2 <= len(recent):
                block1 = recent[i:i + block_len]
                block2 = recent[i + block_len:i + block_len * 2]
                if block1 == block2:
                    consecutive = 2
                    j = i + block_len * 2
                    while j + block_len <= len(recent):
                        if recent[j:j + block_len] == block1:
                            consecutive += 1
                            j += block_len
                        else:
                            break
                    if consecutive >= 5:
                        display = re.sub(r'\s+', ' ', block1)[:25]
                        return True, f"连续块(len={block_len}): \"{display}\" x{consecutive}"
                    i = j
                else:
                    i += 1

    # ========== 方法4: 长度限制（兜底）==========
    if len(text) > MAX_OUTPUT_LENGTH:
        return True, f"输出超过 {MAX_OUTPUT_LENGTH} 字符"

    return False, None


class VLLMOCR:
    """
    vLLM OCR 工具类。

    使用 vLLM 异步引擎，支持 Gundam 模式（动态分辨率，2-6 crops）。
    """

    def __init__(self, model_path='deepseek-ai/DeepSeek-OCR', cuda_device='0'):
        """
        初始化 vLLM OCR 引擎。

        Args:
            model_path: 模型路径（HuggingFace Hub 或本地路径）
            cuda_device: CUDA 设备编号
        """
        self.model_path = model_path
        self.engine = None
        self.sampling_params = None
        self.cuda_device = cuda_device
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    def initialize(self):
        """
        初始化 vLLM 异步引擎。

        引擎只需要初始化一次，之后可以连续处理多张图片。
        """
        print("\n" + "=" * 60)
        print(f"{timestamp()} 正在初始化 vLLM 引擎...")
        print("=" * 60)
        print(f"{timestamp()} 模型: {self.model_path}")
        print(
            f"{timestamp()} 模式: Gundam (base_size=1024, image_size=640, crop_mode=True)"
        )

        engine_args = AsyncEngineArgs(
            model=self.model_path,
            block_size=256,
            max_model_len=8192,
            enforce_eager=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            skip_special_tokens=False,
            extra_args={
                "ngram_size": NGRAM_SIZE,
                "window_size": WINDOW_SIZE,
                "whitelist_token_ids": WHITELIST_TOKEN_IDS,
            },
        )

        print(f"{timestamp()} ✓ vLLM 引擎初始化完成！")
        print(f"{timestamp()}   已启用流式幻觉检测（阈值: {STREAM_CHECK_START} 字符后开始）")
        print(f"{timestamp()}   推理超时限制: {INFERENCE_TIMEOUT_SECONDS} 秒")

    async def _process_image_async(self,
                                   image_path,
                                   output_path,
                                   prompt=None,
                                   show_progress=True,
                                   timeout=INFERENCE_TIMEOUT_SECONDS):
        """
        异步处理单张图片（内部方法）。

        Args:
            image_path: 输入图片路径
            output_path: 输出目录路径
            prompt: OCR prompt，默认为文档 OCR 模式
            show_progress: 是否显示流式输出进度
            timeout: 推理超时时间（秒）

        Returns:
            str: 处理后的 OCR 文本结果
        """
        if self.engine is None:
            raise RuntimeError("引擎未初始化，请先调用 initialize() 方法")

        if prompt is None:
            prompt = '<image>\n<|grounding|>Convert the document to markdown.'

        image = load_image(image_path)
        if image is None:
            raise ValueError(f"无法加载图片: {image_path}")
        image = image.convert('RGB')

        request = {"prompt": prompt, "multi_modal_data": {"image": image}}
        request_id = f"ocr-{uuid.uuid4().hex[:8]}-{Path(image_path).stem}"

        if show_progress:
            print(f"\n{timestamp()} 开始 OCR 识别（流式输出）...")
            print(f"{timestamp()} 图片: {Path(image_path).name}")
            print(f"{timestamp()} 超时限制: {timeout} 秒")

        printed_length = 0
        final_output = ""
        hallucination_detected = False
        hallucination_pattern = None
        last_check_length = 0
        start_time = asyncio.get_event_loop().time()

        try:
            async for request_output in self.engine.generate(
                    request, self.sampling_params, request_id):

                # 检查超时
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    print(f"\n{timestamp()} ⚠ 推理超时（{elapsed:.1f}秒），强制停止...")
                    break

                if request_output.outputs:
                    full_text = request_output.outputs[0].text
                    new_text = full_text[printed_length:]
                    if show_progress:
                        print(new_text, end='', flush=True)
                    printed_length = len(full_text)
                    final_output = full_text

                    # 流式幻觉检测（每隔一定字符数检测一次）
                    if len(final_output
                           ) - last_check_length >= STREAM_CHECK_INTERVAL:
                        last_check_length = len(final_output)
                        hallucination_detected, hallucination_pattern = \
                            detect_hallucination_in_stream(final_output)

                        if hallucination_detected:
                            print(f"\n{timestamp()} ⚠ 流式检测: 检测到幻觉，提前停止生成")
                            print(
                                f"{timestamp()}   幻觉模式: {hallucination_pattern}"
                            )
                            # 尝试取消请求
                            try:
                                await self.engine.abort(request_id)
                            except Exception:
                                pass
                            break

        except asyncio.CancelledError:
            print(f"\n{timestamp()} 推理被取消")
        except Exception as e:
            if show_progress:
                print('\n')
            raise RuntimeError(
                f"vLLM 引擎处理图片时发生错误 (request_id={request_id}): {e}") from e

        if show_progress:
            elapsed = asyncio.get_event_loop().time() - start_time
            print(
                f"\n{timestamp()} 生成完成，耗时: {elapsed:.1f} 秒，输出: {len(final_output)} 字符"
            )

        # 保存幻觉警告
        if hallucination_detected:
            image_name = Path(image_path).stem
            warning_file = Path(
                output_path) / f"{image_name}_hallucination_warning.txt"
            with open(warning_file, 'w', encoding='utf-8') as f:
                f.write(f"检测到幻觉输出\n")
                f.write(f"幻觉模式: {hallucination_pattern}\n")
                f.write(f"输出已在检测到幻觉时停止\n")

        image_name = Path(image_path).stem
        processed_output = save_ocr_results(final_output, image, image_name,
                                            output_path)

        return processed_output

    def process_image(self,
                      image_path,
                      output_path,
                      prompt=None,
                      show_progress=True):
        """
        处理单张图片（同步接口）。

        Args:
            image_path: 输入图片路径
            output_path: 输出目录路径
            prompt: OCR prompt，默认为文档 OCR 模式
            show_progress: 是否显示流式输出进度

        Returns:
            str: 处理后的 OCR 文本结果
        """
        return asyncio.run(
            self._process_image_async(image_path, output_path, prompt,
                                      show_progress))

    async def process_images_async(self,
                                   image_paths,
                                   output_paths,
                                   prompt=None,
                                   show_progress=True):
        """
        异步批量处理多张图片。

        Args:
            image_paths: 图片路径列表
            output_paths: 输出目录路径列表（与图片路径一一对应）
            prompt: OCR prompt，默认为文档 OCR 模式
            show_progress: 是否显示流式输出进度

        Returns:
            list: 处理后的 OCR 文本结果列表
        """
        if self.engine is None:
            raise RuntimeError("引擎未初始化，请先调用 initialize() 方法")

        results = []
        for i, (image_path,
                output_path) in enumerate(zip(image_paths, output_paths)):
            if show_progress:
                print(f"\n{timestamp()} 处理进度: {i+1}/{len(image_paths)}")
            result = await self._process_image_async(image_path, output_path,
                                                     prompt, show_progress)
            results.append(result)
        return results

    def process_images(self,
                       image_paths,
                       output_paths,
                       prompt=None,
                       show_progress=True):
        """
        批量处理多张图片（同步接口）。

        Args:
            image_paths: 图片路径列表
            output_paths: 输出目录路径列表（与图片路径一一对应）
            prompt: OCR prompt，默认为文档 OCR 模式
            show_progress: 是否显示流式输出进度

        Returns:
            list: 处理后的 OCR 文本结果列表
        """
        return asyncio.run(
            self.process_images_async(image_paths, output_paths, prompt,
                                      show_progress))
