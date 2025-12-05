#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR 识别统一命令行入口

支持 Transformers 和 vLLM 两种框架，支持随机选择或批量处理图片。

使用说明:
  - Transformers 框架: 在 deepseek-ocr 环境中运行
    conda activate deepseek-ocr
    python run_ocr_cli.py --framework transformers --mode random

  - vLLM 框架: 在 deepseek-ocr-vllm 环境中运行
    conda activate deepseek-ocr-vllm
    python run_ocr_cli.py --framework vllm --mode all
"""

import argparse
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 环境变量设置（必须在导入框架模块之前）
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 支持的图片格式
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg'}


def get_image_files(input_dir):
    """
    获取输入目录中所有支持的图片文件。

    Args:
        input_dir: 输入目录路径

    Returns:
        list: 图片文件路径列表（按文件名排序）
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    if not image_files:
        raise ValueError(
            f"在 {input_dir} 中没有找到支持的图片文件（{', '.join(SUPPORTED_FORMATS)}）")

    image_files.sort(key=lambda x: x.name)
    return image_files


def process_with_transformers(image_files,
                              result_base_dir,
                              framework_dir,
                              mode='all'):
    """
    使用 Transformers 框架处理图片。

    Args:
        image_files: 图片文件列表
        result_base_dir: 结果基础目录
        framework_dir: 框架结果目录（Transformers）
        mode: 处理模式（'random' 或 'all'）

    Returns:
        list: 处理结果列表
    """
    from ocr_transformers import TransformersOCR

    print("\n" + "=" * 60)
    print("使用 Transformers 框架处理")
    print("=" * 60)

    ocr = TransformersOCR()
    ocr.initialize()

    if mode == 'random':
        image_files = [random.choice(image_files)]

    results = []
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理图片: {image_file.name}")
        image_name = image_file.stem

        output_dir = framework_dir / image_name
        output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_file, output_dir / image_file.name)

        try:
            result = ocr.process_image(str(image_file), str(output_dir))
            results.append({
                'image': image_file.name,
                'result': result,
                'output_dir': output_dir
            })
            print(f"✓ 完成: {image_file.name}")
        except Exception as e:
            print(f"✗ 错误: {image_file.name} - {e}")
            results.append({
                'image': image_file.name,
                'result': None,
                'error': str(e),
                'output_dir': output_dir
            })

    return results


def process_with_vllm(image_files, result_base_dir, framework_dir, mode='all'):
    """
    使用 vLLM 框架处理图片。

    Args:
        image_files: 图片文件列表
        result_base_dir: 结果基础目录
        framework_dir: 框架结果目录（vLLM）
        mode: 处理模式（'random' 或 'all'）

    Returns:
        list: 处理结果列表
    """
    import asyncio

    from ocr_vllm import VLLMOCR

    print("\n" + "=" * 60)
    print("使用 vLLM 框架处理")
    print("=" * 60)

    ocr = VLLMOCR()
    ocr.initialize()

    if mode == 'random':
        image_files = [random.choice(image_files)]

    # 准备输出目录和复制图片
    image_paths = []
    output_paths = []
    for image_file in image_files:
        image_name = image_file.stem
        output_dir = framework_dir / image_name
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_file, output_dir / image_file.name)
        image_paths.append(str(image_file))
        output_paths.append(str(output_dir))

    # 使用异步批量处理，确保使用同一个事件循环
    async def process_all():
        results = []
        for i, (image_path,
                output_path) in enumerate(zip(image_paths, output_paths), 1):
            image_file = Path(image_path)
            print(f"\n[{i}/{len(image_paths)}] 处理图片: {image_file.name}")
            try:
                result = await ocr._process_image_async(image_path,
                                                        output_path,
                                                        show_progress=True)
                results.append({
                    'image': image_file.name,
                    'result': result,
                    'output_dir': Path(output_path)
                })
                print(f"✓ 完成: {image_file.name}")
            except Exception as e:
                print(f"✗ 错误: {image_file.name} - {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'image': image_file.name,
                    'result': None,
                    'error': str(e),
                    'output_dir': Path(output_path)
                })
        return results

    # 使用同一个事件循环处理所有图片
    try:
        results = asyncio.run(process_all())
    except Exception as e:
        print(f"\n\nvLLM 引擎发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        # 返回部分结果
        results = [{
            'image': Path(p).name,
            'result': None,
            'error': f'引擎崩溃: {str(e)}',
            'output_dir': Path(op)
        } for p, op in zip(image_paths, output_paths)]

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='OCR 识别统一命令行入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 Transformers 随机处理一张图片（在 deepseek-ocr 环境中）
  conda activate deepseek-ocr
  python run_ocr_cli.py --framework transformers --mode random --input test_resouce/sample1

  # 使用 vLLM 处理所有图片（在 deepseek-ocr-vllm 环境中）
  conda activate deepseek-ocr-vllm
  python run_ocr_cli.py --framework vllm --mode all --input test_resouce/sample1

注意:
  - Transformers 框架需要在 deepseek-ocr 环境中运行
  - vLLM 框架需要在 deepseek-ocr-vllm 环境中运行
  - 两个框架的依赖版本不兼容，无法在同一环境中同时使用
        """)

    parser.add_argument('--framework',
                        choices=['transformers', 'vllm'],
                        required=True,
                        help='选择使用的框架: transformers 或 vllm（必选）')

    parser.add_argument('--mode',
                        choices=['random', 'all'],
                        default='random',
                        help='工作模式: random（随机选择1张）或 all（处理所有图片）（默认: random）')

    parser.add_argument('--input',
                        type=str,
                        default='test_resouce/sample1',
                        help='输入目录路径（默认: test_resouce/sample1）')

    parser.add_argument('--output',
                        type=str,
                        default='results',
                        help='输出基础目录（默认: results）')

    args = parser.parse_args()

    print("=" * 60)
    print("OCR 识别工具")
    print("=" * 60)
    print(f"框架: {args.framework}")
    print(f"模式: {args.mode}")
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print("=" * 60)

    # 获取图片文件列表
    try:
        image_files = get_image_files(args.input)
        print(f"\n找到 {len(image_files)} 张图片")
        if args.mode == 'random':
            print("（将随机选择1张处理）")
        else:
            print("（将按文件名顺序处理所有图片）")
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_base_dir = Path(args.output) / timestamp
    result_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n结果保存目录: {result_base_dir}")

    # 根据框架选择处理函数
    try:
        if args.framework == 'transformers':
            results = process_with_transformers(image_files, result_base_dir,
                                                result_base_dir, args.mode)
        else:  # vllm
            results = process_with_vllm(image_files, result_base_dir,
                                        result_base_dir, args.mode)
    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n处理发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"框架: {args.framework}")
    print(f"结果目录: {result_base_dir}")

    if results:
        success_count = sum(1 for r in results if r.get('result') is not None)
        print(f"成功: {success_count}/{len(results)}")

        if success_count < len(results):
            print(f"\n失败的图片:")
            for r in results:
                if r.get('result') is None:
                    error_msg = r.get('error', '未知错误')
                    print(f"  - {r['image']}: {error_msg}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
