#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR后处理命令行工具

使用 Llama-3.3-Swallow-70B-Instruct 模型对OCR结果进行后处理。
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from postprocess_pipeline import process_ocr_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='OCR后处理工具 - 支持8B和70B模型进行后处理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

8B模型示例:
  python run_postprocess_cli.py \\
    --input results/20241207_120000/vLLM \\
    --model models/Llama-3.3-Swallow-8B-Instruct \\
    --model-preset 8b

70B模型示例:
  python run_postprocess_cli.py \\
    --input results/20241207_120000/vLLM \\
    --model llama3.3-swallow-70b \\
    --model-preset 70b \\
    --context-before 2 --context-after 2

量化70B模型示例:
  python run_postprocess_cli.py \\
    --input results/20241207_120000/vLLM \\
    --model llama3.3-swallow-70b/models/Llama-3.3-Swallow-70B-Instruct-v0.4 \\
    --model-preset 70b \\
    --quant-method bitsandbytes
        """)

    parser.add_argument('--input',
                        type=str,
                        required=True,
                        help='OCR结果目录路径（包含各页markdown文件的目录）')

    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='输出目录路径（默认: 在输入目录下创建postprocessed子目录）')

    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='模型路径（必需：本地模型目录路径）')

    parser.add_argument('--cuda-device',
                        type=str,
                        default='0',
                        help='CUDA设备编号（默认: 0）')

    parser.add_argument('--context-before',
                        type=int,
                        default=1,
                        help='上下文窗口前页数（默认: 1）')

    parser.add_argument('--context-after',
                        type=int,
                        default=1,
                        help='上下文窗口后页数（默认: 1）')

    parser.add_argument('--model-preset',
                        type=str,
                        required=True,
                        choices=['8b', '70b'],
                        help='模型预设（必需：8b/70b - 选择对应的配置）')

    parser.add_argument('--quant-method',
                        type=str,
                        default=None,
                        choices=['bitsandbytes', 'gptq', 'awq'],
                        help='量化方法（可选：bitsandbytes/gptq/awq - 仅用于70B模型）')

    parser.add_argument('--load-format',
                        type=str,
                        default=None,
                        choices=['auto', 'safetensors', 'pt', 'bitsandbytes', 'gptq', 'awq'],
                        help='模型加载格式（可选：自动检测或指定格式）')

    parser.add_argument('--tensor-parallel-size',
                        type=int,
                        default=1,
                        help='张量并行大小（默认按预设选择，单GX10 128G推荐 1）')

    parser.add_argument('--no-progress',
                        action='store_true',
                        help='不显示生成进度')

    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='强制使用eager模式（适用于编译错误时，会降低性能）')

    args = parser.parse_args()

    # 验证输入目录
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"\n错误: 输入目录不存在: {args.input}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"\n错误: 输入路径不是目录: {args.input}")
        sys.exit(1)

    # 验证模型路径
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n错误: 模型路径不存在: {args.model}")
        print("提示: 请确保模型路径正确，例如:")
        if args.model_preset == "8b":
            print("  - models/Llama-3.3-Swallow-8B-Instruct")
        else:
            print("  - llama3.3-swallow-70b")
            print("  - /path/to/your/70b/model")
        sys.exit(1)

    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir / "postprocessed"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OCR后处理工具")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {args.model}")
    print(f"模型预设: {args.model_preset}")
    print(f"CUDA设备: {args.cuda_device}")
    if args.tensor_parallel_size:
        print(f"张量并行: {args.tensor_parallel_size}")
    print(f"上下文窗口: 前{args.context_before}页, 后{args.context_after}页")
    if args.quant_method:
        print(f"量化方法: {args.quant_method}")
        print(f"加载格式: {args.load_format or 'auto'}")

    # 显示模型配置信息
    print("\n模型配置:")
    if args.model_preset == "8b":
        print("  - 上下文长度: 16384 tokens")
        print("  - GPU内存使用: 90%")
        print("  - 批处理tokens: 2048")
    elif args.model_preset == "70b":
        print("  - 上下文长度: 8192 tokens")
        print("  - GPU内存使用: 85%")
        print("  - 批处理tokens: 1024")
        if args.quant_method == "bitsandbytes":
            print("  - 量化优化: int8 (内存减少约43%)")
        elif args.quant_method in ["gptq", "awq"]:
            print("  - 量化优化: 4bit (内存减少约68%)")
    print("=" * 60)

    # 运行异步处理
    try:
        asyncio.run(
            process_ocr_results(
                ocr_result_dir=input_dir,
                output_dir=output_dir,
                model_path=args.model,
                cuda_device=args.cuda_device,
                context_before=args.context_before,
                context_after=args.context_after,
                show_progress=not args.no_progress,
                model_preset=args.model_preset,
                tensor_parallel_size=args.tensor_parallel_size,
                load_format=args.load_format,
                quant_method=args.quant_method,
                enforce_eager=args.enforce_eager,
            ))
    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n处理发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
