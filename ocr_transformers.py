#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformers OCR 工具类

使用 Transformers 框架运行 DeepSeek-OCR 模型。
"""

import os
import sys
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from ocr_utils import load_image, save_ocr_results


class TransformersOCR:
    """
    Transformers OCR 工具类。

    支持 Large 模型配置（1280×1280，400 vision tokens）。
    """

    def __init__(self, model_path='deepseek-ai/DeepSeek-OCR', cuda_device='0'):
        """
        初始化 Transformers OCR 引擎。

        Args:
            model_path: 模型路径（HuggingFace Hub 或本地路径）
            cuda_device: CUDA 设备编号
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.cuda_device = cuda_device
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    def _heartbeat_printer(self, stop_event):
        """定期输出心跳信息，让用户知道脚本还在运行"""
        while not stop_event.is_set():
            time.sleep(5)
            if not stop_event.is_set():
                print(".", end="", flush=True)

    def initialize(self):
        """
        加载模型和 tokenizer。

        如果模型未下载，将自动下载并显示进度。
        """
        print("\n" + "=" * 60)
        print("开始加载 Transformers 模型...")
        print("=" * 60)
        print(f"\n模型名称: {self.model_path}")
        print("模型配置: Large (1280×1280, 400 vision tokens)")

        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(target=self._heartbeat_printer,
                                            args=(stop_heartbeat, ),
                                            daemon=True)
        heartbeat_thread.start()

        try:
            print("\n步骤 1/3: 正在加载 tokenizer...")
            print("（如果 tokenizer 未下载，将自动下载并显示进度）")
            sys.stdout.flush()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True)
            stop_heartbeat.set()
            print("\n✓ Tokenizer 加载完成")

            print("\n步骤 2/3: 正在加载模型权重...")
            print("（如果模型未下载，将自动下载并显示进度，这可能需要一些时间）")
            print("（模型大小约 6.67GB，请确保网络连接稳定）")
            print("（✓ 支持断点续传：如果下载中断，重新运行脚本会从中断处继续）")

            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            incomplete_files = list(cache_dir.glob("**/*.incomplete"))
            if incomplete_files:
                print(f"\n检测到 {len(incomplete_files)} 个未完成的下载文件...")
                for inc_file in incomplete_files[:3]:
                    size_gb = inc_file.stat().st_size / (1024**3)
                    print(f"  - {inc_file.name[:50]}... ({size_gb:.2f} GB)")

            sys.stdout.flush()

            stop_heartbeat.clear()
            heartbeat_thread = threading.Thread(target=self._heartbeat_printer,
                                                args=(stop_heartbeat, ),
                                                daemon=True)
            heartbeat_thread.start()

            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

            try:
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True,
                )
            except (ImportError, OSError, ValueError, AttributeError,
                    Exception) as e:
                error_str = str(e).lower()
                if ('flash_attn' in error_str or 'flash_attention' in error_str
                        or 'sdpa' in error_str
                        or 'llamaflashattention' in error_str
                        or 'cannot import' in error_str):
                    print("⚠ FlashAttention2 不可用，使用 Eager Attention 实现...")
                    print("（注意：Eager 实现速度较慢，但功能完整）")
                    try:
                        self.model = AutoModel.from_pretrained(
                            self.model_path,
                            _attn_implementation='eager',
                            trust_remote_code=True,
                            use_safetensors=True,
                        )
                    except Exception as e2:
                        print(f"⚠ Eager 模式也失败，尝试不指定 attention 实现...")
                        print(f"错误: {e2}")
                        self.model = AutoModel.from_pretrained(
                            self.model_path,
                            trust_remote_code=True,
                            use_safetensors=True,
                        )
                else:
                    raise
            stop_heartbeat.set()
            print("\n✓ 模型权重加载完成")

            print("\n步骤 3/3: 正在将模型移动到 GPU 并设置精度...")
            sys.stdout.flush()
            self.model = self.model.eval().cuda().to(torch.bfloat16)
            print("✓ 模型初始化完成！")
        except KeyboardInterrupt:
            stop_heartbeat.set()
            print("\n\n用户中断，正在退出...")
            print("（提示：模型下载支持断点续传，下次运行会从中断处继续）")
            sys.exit(1)
        except Exception as e:
            stop_heartbeat.set()
            print(f"\n\n发生错误: {e}")
            raise

    def process_image(self, image_path, output_path, prompt=None):
        """
        处理单张图片。

        Args:
            image_path: 输入图片路径
            output_path: 输出目录路径
            prompt: OCR prompt，默认为文档 OCR 模式

        Returns:
            str: 处理后的 OCR 文本结果
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未初始化，请先调用 initialize() 方法")

        if prompt is None:
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "

        image_name = Path(image_path).stem

        print(f"\n开始 OCR 识别（Large 模型，1280×1280）...")
        print(f"图片: {Path(image_path).name}")

        res = self.model.infer(self.tokenizer,
                               prompt=prompt,
                               image_file=str(image_path),
                               output_path=str(output_path),
                               base_size=1280,
                               image_size=1280,
                               crop_mode=False,
                               save_results=True,
                               test_compress=False)

        result_mmd_file = Path(output_path) / "result.mmd"

        if res is not None:
            if isinstance(res, str):
                ocr_text = res
            elif hasattr(res, 'text'):
                ocr_text = res.text
            elif isinstance(res, dict):
                ocr_text = res.get('text', res.get('output', str(res)))
            else:
                ocr_text = str(res)
        else:
            if result_mmd_file.exists():
                with open(result_mmd_file, 'r', encoding='utf-8') as f:
                    ocr_text = f.read()
            else:
                ocr_text = "OCR 结果未找到"
                print("⚠ 警告: 未找到 OCR 结果文件")

        image = load_image(image_path)
        if image is None:
            raise ValueError(f"无法加载图片: {image_path}")
        image = image.convert('RGB')

        processed_output = save_ocr_results(ocr_text, image, image_name,
                                            output_path)

        return processed_output
