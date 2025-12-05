#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vLLM OCR 工具类

使用 vLLM 推理引擎运行 DeepSeek-OCR 模型。
vLLM 默认使用 Gundam 模式（base_size=1024, image_size=640, crop_mode=True）。
"""

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path

# vLLM 环境配置（必须在 import vllm 之前设置）
os.environ['VLLM_USE_V1'] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

from ocr_utils import load_image, save_ocr_results

# NGram 重复抑制参数
NGRAM_SIZE = 30
WINDOW_SIZE = 90
WHITELIST_TOKEN_IDS = [128821, 128822]


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
        print("正在初始化 vLLM 引擎...")
        print("=" * 60)
        print(f"模型: {self.model_path}")
        print(f"模式: Gundam (base_size=1024, image_size=640, crop_mode=True)")

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

        print("✓ vLLM 引擎初始化完成！")

    async def _process_image_async(self,
                                   image_path,
                                   output_path,
                                   prompt=None,
                                   show_progress=True):
        """
        异步处理单张图片（内部方法）。

        Args:
            image_path: 输入图片路径
            output_path: 输出目录路径
            prompt: OCR prompt，默认为文档 OCR 模式
            show_progress: 是否显示流式输出进度

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
        # 使用 UUID 确保请求 ID 唯一性
        request_id = f"ocr-{uuid.uuid4().hex[:8]}-{Path(image_path).stem}"

        if show_progress:
            print(f"\n开始 OCR 识别（流式输出）...")
            print(f"图片: {Path(image_path).name}")

        printed_length = 0
        final_output = ""

        try:
            async for request_output in self.engine.generate(
                    request, self.sampling_params, request_id):
                if request_output.outputs:
                    full_text = request_output.outputs[0].text
                    new_text = full_text[printed_length:]
                    if show_progress:
                        print(new_text, end='', flush=True)
                    printed_length = len(full_text)
                    final_output = full_text
        except Exception as e:
            if show_progress:
                print('\n')
            raise RuntimeError(
                f"vLLM 引擎处理图片时发生错误 (request_id={request_id}): {e}") from e

        if show_progress:
            print('\n')

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
                print(f"\n处理进度: {i+1}/{len(image_paths)}")
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
