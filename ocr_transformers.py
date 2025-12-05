#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformers OCR 工具类

使用 Transformers 框架运行 DeepSeek-OCR 模型。
"""

import os
import re
import signal
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path

import torch
from transformers import (AutoModel, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList)

from ocr_utils import load_image, save_ocr_results


def timestamp():
    """返回当前时间戳字符串"""
    return datetime.now().strftime("[%H:%M:%S]")


# 幻觉检测配置
HALLUCINATION_PATTERN_THRESHOLD = 5  # 重复模式出现次数阈值
HALLUCINATION_MIN_PATTERN_LEN = 8  # 最小重复模式长度（字符）
MAX_OUTPUT_LENGTH = 50000  # 最大输出字符数
MAX_NUMBERED_LIST_ITEMS = 50  # 编号列表最大项数（超过视为幻觉）
INFERENCE_TIMEOUT_SECONDS = 300  # 单张图片推理超时时间（秒）= 5分钟
STREAM_CHECK_INTERVAL = 500  # 每生成多少个 token 检查一次幻觉
STREAM_CHECK_START = 2000  # 至少生成多少 token 后才开始检查（避免误判正常文档）
MAX_TOKENS_LIMIT = 16000  # 流式检测时的最大 token 数


class HallucinationStoppingCriteria(StoppingCriteria):
    """
    自定义停止条件：实时检测生成过程中的幻觉并提前停止。
    """

    def __init__(self,
                 tokenizer,
                 check_interval=STREAM_CHECK_INTERVAL,
                 check_start=STREAM_CHECK_START,
                 max_tokens=MAX_TOKENS_LIMIT):
        self.tokenizer = tokenizer
        self.check_interval = check_interval
        self.check_start = check_start
        self.max_tokens = max_tokens
        self.generated_text = ""
        self.token_count = 0
        self.stopped_reason = None

    def __call__(self, input_ids, scores, **kwargs):
        self.token_count += 1

        # 至少生成一定数量的 token 后才开始检查（避免误判）
        if self.token_count < self.check_start:
            return False

        # 每隔一定数量的 token 进行检查
        if self.token_count % self.check_interval == 0:
            # 解码当前生成的文本
            try:
                current_text = self.tokenizer.decode(input_ids[0],
                                                     skip_special_tokens=False)
                self.generated_text = current_text
            except:
                return False

            # 检测幻觉模式
            if self._detect_hallucination(current_text):
                self.stopped_reason = "检测到幻觉模式"
                print(
                    f"\n⚠ 流式检测: 检测到幻觉，提前停止生成 (已生成 {self.token_count} tokens)")
                return True

        # 超过最大 token 数限制
        if self.token_count >= self.max_tokens:
            self.stopped_reason = f"超过最大 token 数限制 ({self.max_tokens})"
            print(f"\n⚠ 达到最大 token 数限制 ({self.max_tokens})，停止生成")
            return True

        return False

    def _detect_hallucination(self, text):
        """
        通用幻觉检测算法（基于压缩比和 N-gram 频率）。
        
        核心原理：幻觉的本质是"异常重复"。
        """
        # 如果文本太短，不检测
        if len(text) < 2000:
            return False

        # ========== 方法1: 压缩比检测（最通用）==========
        text_bytes = text.encode('utf-8')
        if len(text_bytes) > 2000:
            compressed = zlib.compress(text_bytes, level=1)
            compression_ratio = len(compressed) / len(text_bytes)

            # 压缩比异常低说明有大量重复
            if len(text) > 10000 and compression_ratio < 0.12:
                return True
            if len(text) > 5000 and compression_ratio < 0.08:
                return True

        # ========== 方法2: 滑动窗口 N-gram 高频检测 ==========
        if len(text) > 3000:
            recent = text[-3000:]
            for n in [12, 20, 30, 50]:
                if len(recent) < n * 8:
                    continue
                ngrams = {}
                for i in range(len(recent) - n):
                    gram = recent[i:i + n]
                    ngrams[gram] = ngrams.get(gram, 0) + 1
                if ngrams:
                    max_gram, max_count = max(ngrams.items(),
                                              key=lambda x: x[1])
                    coverage = (max_count * n) / len(recent)
                    if coverage > 0.25 and max_count >= 8:
                        return True

        # ========== 方法3: 连续重复块检测 ==========
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
                            return True
                        i = j
                    else:
                        i += 1

        return False


def detect_and_truncate_hallucination(
        text,
        pattern_threshold=HALLUCINATION_PATTERN_THRESHOLD,
        min_pattern_len=HALLUCINATION_MIN_PATTERN_LEN):
    """
    检测并截断幻觉输出。

    支持检测：
    1. 完全相同的连续重复模式
    2. 带编号的重复模式（如 "1. xxx", "2. xxx", ...）
    3. 高频短语重复

    Args:
        text: OCR 输出文本
        pattern_threshold: 重复模式出现次数阈值
        min_pattern_len: 最小重复模式长度

    Returns:
        tuple: (处理后的文本, 是否检测到幻觉, 幻觉模式)
    """
    if len(text) < min_pattern_len * pattern_threshold:
        return text, False, None

    # 方法1: 检测带编号的重复模式 (如 "1. xxx\n2. xxx\n3. xxx...")
    # 将编号替换为占位符后检测重复
    normalized_text = re.sub(r'(\d+)\.\s*', '#. ', text)
    lines = normalized_text.split('\n')

    # 检测连续相同的行（忽略编号）
    if len(lines) > pattern_threshold:
        consecutive_count = 1
        prev_line = lines[0].strip()
        first_repeat_idx = 0

        for i, line in enumerate(lines[1:], 1):
            current_line = line.strip()
            if current_line == prev_line and len(
                    current_line) >= min_pattern_len:
                consecutive_count += 1
                if consecutive_count >= pattern_threshold:
                    # 找到重复模式，计算原始文本中的截断位置
                    # 找到第一次重复开始的位置
                    original_lines = text.split('\n')
                    truncate_line_idx = first_repeat_idx + 1
                    truncated = '\n'.join(original_lines[:truncate_line_idx])
                    pattern_display = prev_line.replace('#. ', 'N. ')[:50]
                    return truncated, True, f"编号列表重复: {pattern_display}..."
            else:
                consecutive_count = 1
                first_repeat_idx = i
            prev_line = current_line

    # 方法2: 检测高频短语（去除数字后）
    # 提取所有非空行，去除编号
    text_no_numbers = re.sub(r'\d+', '', text)
    # 检测长度>=min_pattern_len的重复短语
    for phrase_len in range(min_pattern_len,
                            min(50,
                                len(text_no_numbers) // pattern_threshold)):
        phrase_counts = {}
        for i in range(len(text_no_numbers) - phrase_len):
            phrase = text_no_numbers[i:i + phrase_len]
            if len(phrase.strip()) >= min_pattern_len:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        # 找出高频短语
        for phrase, count in phrase_counts.items():
            if count >= pattern_threshold * 3:  # 高频阈值更严格
                # 找到第一次出现的位置，在附近截断
                first_pos = text_no_numbers.find(phrase)
                # 映射回原始文本位置（近似）
                truncate_pos = min(first_pos + 500, len(text))
                # 找到最近的换行符
                newline_pos = text.rfind('\n', 0, truncate_pos)
                if newline_pos > truncate_pos // 2:
                    truncated = text[:newline_pos]
                else:
                    truncated = text[:truncate_pos]
                return truncated, True, f"高频重复短语: {phrase.strip()[:40]}... (出现{count}次)"

    # 方法3: 检测完全相同的连续重复模式
    for pattern_len in range(min_pattern_len,
                             min(100,
                                 len(text) // pattern_threshold)):
        for start in range(len(text) - pattern_len * pattern_threshold):
            pattern = text[start:start + pattern_len]
            if len(pattern.strip()) < min_pattern_len // 2:
                continue

            count = 1
            pos = start + pattern_len
            while pos + pattern_len <= len(
                    text) and text[pos:pos + pattern_len] == pattern:
                count += 1
                pos += pattern_len
                if count >= pattern_threshold:
                    truncated = text[:start + pattern_len]
                    last_newline = truncated.rfind('\n')
                    if last_newline > start - 100:
                        truncated = truncated[:last_newline + 1]
                    return truncated, True, f"连续重复: {pattern[:50]}..."

    # 方法4: 检查输出长度是否异常
    if len(text) > MAX_OUTPUT_LENGTH:
        truncated = text[:MAX_OUTPUT_LENGTH]
        last_para = truncated.rfind('\n\n')
        if last_para > MAX_OUTPUT_LENGTH // 2:
            truncated = truncated[:last_para]
        return truncated, True, f"输出超过 {MAX_OUTPUT_LENGTH} 字符"

    return text, False, None


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

    def _inject_stopping_criteria(self):
        """注入幻觉检测 StoppingCriteria 到模型的 generate 方法"""
        original_generate = self.model.generate

        def patched_generate(*args, **kwargs):
            # 创建幻觉检测停止条件
            hallucination_criteria = HallucinationStoppingCriteria(
                self.tokenizer,
                check_interval=STREAM_CHECK_INTERVAL,
                check_start=STREAM_CHECK_START,
                max_tokens=MAX_TOKENS_LIMIT)
            # 保存引用以便后续检查
            self._current_stopping_criteria = hallucination_criteria

            # 合并现有的 stopping_criteria
            if 'stopping_criteria' in kwargs and kwargs['stopping_criteria']:
                kwargs['stopping_criteria'].append(hallucination_criteria)
            else:
                kwargs['stopping_criteria'] = StoppingCriteriaList(
                    [hallucination_criteria])

            return original_generate(*args, **kwargs)

        self.model.generate = patched_generate
        self._current_stopping_criteria = None

    def initialize(self):
        """
        加载模型和 tokenizer。

        如果模型未下载，将自动下载并显示进度。
        """
        print("\n" + "=" * 60)
        print(f"{timestamp()} 开始加载 Transformers 模型...")
        print("=" * 60)
        print(f"{timestamp()} 模型名称: {self.model_path}")
        print(f"{timestamp()} 模型配置: Large (1280×1280, 400 vision tokens)")

        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(target=self._heartbeat_printer,
                                            args=(stop_heartbeat, ),
                                            daemon=True)
        heartbeat_thread.start()

        try:
            print(f"\n{timestamp()} 步骤 1/3: 正在加载 tokenizer...")
            print(f"{timestamp()} （如果 tokenizer 未下载，将自动下载并显示进度）")
            sys.stdout.flush()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True)
            stop_heartbeat.set()
            print(f"\n{timestamp()} ✓ Tokenizer 加载完成")

            print(f"\n{timestamp()} 步骤 2/3: 正在加载模型权重...")
            print(f"{timestamp()} （如果模型未下载，将自动下载并显示进度，这可能需要一些时间）")
            print(f"{timestamp()} （模型大小约 6.67GB，请确保网络连接稳定）")
            print(f"{timestamp()} （✓ 支持断点续传：如果下载中断，重新运行脚本会从中断处继续）")

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
                    print(
                        f"{timestamp()} ⚠ FlashAttention2 不可用，使用 Eager Attention 实现..."
                    )
                    print(f"{timestamp()} （注意：Eager 实现速度较慢，但功能完整）")
                    try:
                        self.model = AutoModel.from_pretrained(
                            self.model_path,
                            _attn_implementation='eager',
                            trust_remote_code=True,
                            use_safetensors=True,
                        )
                    except Exception as e2:
                        print(
                            f"{timestamp()} ⚠ Eager 模式也失败，尝试不指定 attention 实现..."
                        )
                        print(f"{timestamp()} 错误: {e2}")
                        self.model = AutoModel.from_pretrained(
                            self.model_path,
                            trust_remote_code=True,
                            use_safetensors=True,
                        )
                else:
                    raise
            stop_heartbeat.set()
            print(f"\n{timestamp()} ✓ 模型权重加载完成")

            print(f"\n{timestamp()} 步骤 3/3: 正在将模型移动到 GPU 并设置精度...")
            sys.stdout.flush()
            self.model = self.model.eval().cuda().to(torch.bfloat16)

            # 注入幻觉检测 StoppingCriteria
            self._inject_stopping_criteria()
            print(f"{timestamp()} ✓ 模型初始化完成！（已启用流式幻觉检测）")
        except KeyboardInterrupt:
            stop_heartbeat.set()
            print("\n\n用户中断，正在退出...")
            print("（提示：模型下载支持断点续传，下次运行会从中断处继续）")
            sys.exit(1)
        except Exception as e:
            stop_heartbeat.set()
            print(f"\n\n发生错误: {e}")
            raise

    def _run_inference(self, prompt, image_path, output_path):
        """在单独线程中运行推理（内部方法）"""
        return self.model.infer(self.tokenizer,
                                prompt=prompt,
                                image_file=str(image_path),
                                output_path=str(output_path),
                                base_size=1280,
                                image_size=1280,
                                crop_mode=False,
                                save_results=True,
                                test_compress=False)

    def process_image(self,
                      image_path,
                      output_path,
                      prompt=None,
                      timeout=INFERENCE_TIMEOUT_SECONDS):
        """
        处理单张图片（带超时保护）。

        Args:
            image_path: 输入图片路径
            output_path: 输出目录路径
            prompt: OCR prompt，默认为文档 OCR 模式
            timeout: 推理超时时间（秒），默认300秒

        Returns:
            str: 处理后的 OCR 文本结果
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未初始化，请先调用 initialize() 方法")

        if prompt is None:
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "

        image_name = Path(image_path).stem
        result_mmd_file = Path(output_path) / "result.mmd"

        print(f"\n{timestamp()} 开始 OCR 识别（Large 模型，1280×1280）...")
        print(f"{timestamp()} 图片: {Path(image_path).name}")
        print(f"{timestamp()} 超时限制: {timeout} 秒")

        # 记录开始时间
        start_time = time.time()

        # 使用线程池执行推理，支持超时
        res = None
        timed_out = False
        inference_error = None

        print(f"{timestamp()} 开始推理...")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_inference, prompt, image_path,
                                     output_path)
            try:
                res = future.result(timeout=timeout)
                elapsed = time.time() - start_time
                print(f"{timestamp()} 推理完成，耗时: {elapsed:.1f} 秒")
            except FuturesTimeoutError:
                timed_out = True
                elapsed = time.time() - start_time
                print(
                    f"\n{timestamp()} ⚠ 推理超时（实际耗时 {elapsed:.1f} 秒，限制 {timeout} 秒）"
                )
                print(f"{timestamp()} 可能发生幻觉，尝试读取已保存的部分结果...")
            except Exception as e:
                inference_error = e
                elapsed = time.time() - start_time
                print(f"\n{timestamp()} ⚠ 推理发生错误（耗时 {elapsed:.1f} 秒）: {e}")

        # 尝试读取结果
        print(f"{timestamp()} 处理推理结果...")
        ocr_text = ""

        # 优先从返回值获取结果
        if res is not None:
            print(f"{timestamp()} 从返回值读取结果...")
            if isinstance(res, str):
                ocr_text = res
            elif hasattr(res, 'text'):
                ocr_text = res.text
            elif isinstance(res, dict):
                ocr_text = res.get('text', res.get('output', str(res)))
            else:
                ocr_text = str(res)
            print(f"{timestamp()} 结果长度: {len(ocr_text)} 字符")

        # 如果返回值为空，尝试从文件读取
        if not ocr_text and result_mmd_file.exists():
            print(f"{timestamp()} 从文件读取结果: {result_mmd_file}")
            with open(result_mmd_file, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
            print(f"{timestamp()} 文件结果长度: {len(ocr_text)} 字符")

        # 如果仍然没有结果
        if not ocr_text:
            if timed_out:
                ocr_text = f"[推理超时] 图片: {Path(image_path).name}"
                print(f"{timestamp()} ⚠ 超时且未找到结果文件")
            elif inference_error:
                ocr_text = f"[推理错误] 图片: {Path(image_path).name} - {inference_error}"
                print(f"{timestamp()} ⚠ 推理错误且未找到结果文件")
            else:
                ocr_text = "OCR 结果未找到"
                print(f"{timestamp()} ⚠ 未找到 OCR 结果")

        # 幻觉检测和处理
        print(f"{timestamp()} 执行幻觉检测...")
        ocr_text, has_hallucination, pattern = detect_and_truncate_hallucination(
            ocr_text)
        if has_hallucination or timed_out:
            if has_hallucination:
                print(f"{timestamp()} ⚠ 检测到幻觉输出，已自动截断")
                print(f"{timestamp()}   幻觉模式: {pattern}")
            # 保存警告到单独文件
            warning_file = Path(
                output_path) / f"{image_name}_hallucination_warning.txt"
            with open(warning_file, 'w', encoding='utf-8') as f:
                if timed_out:
                    f.write(f"推理超时（{timeout}秒）\n")
                if has_hallucination:
                    f.write(f"检测到幻觉输出\n")
                    f.write(f"幻觉模式: {pattern}\n")
                f.write(f"输出已被自动截断\n")
        else:
            print(f"{timestamp()} 幻觉检测通过")

        print(f"{timestamp()} 加载原始图片...")
        image = load_image(image_path)
        if image is None:
            raise ValueError(f"无法加载图片: {image_path}")
        image = image.convert('RGB')

        print(f"{timestamp()} 保存处理结果...")
        processed_output = save_ocr_results(ocr_text, image, image_name,
                                            output_path)

        total_time = time.time() - start_time
        print(f"{timestamp()} 处理完成，总耗时: {total_time:.1f} 秒")
        return processed_output
