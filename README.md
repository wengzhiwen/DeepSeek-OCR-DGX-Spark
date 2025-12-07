# DeepSeek-OCR on DGX Spark (ASUS GX10)

本仓库是 [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 的 Fork，专注于在 **NVIDIA DGX Spark (ASUS GX10)** 上配置可运行的原生环境。

> 📖 **原项目**: [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) | [论文](https://arxiv.org/abs/2510.18234) | [HuggingFace 模型](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

## 目录

- [硬件环境](#硬件环境)
- [软件版本](#软件版本)
- [快速验证](#快速验证)
- [一、Transformers 推理环境 (deepseek-ocr)](#一transformers-推理环境-deepseek-ocr)
- [二、Transformers 推理使用](#二transformers-推理使用)
- [三、vLLM 推理环境 (deepseek-ocr-vllm)](#三vllm-推理环境-deepseek-ocr-vllm)
- [四、量化 vLLM 环境 (deepseek-ocr-70b-quant)](#四量化-vllm-环境-deepseek-ocr-70b-quant)
- [五、环境使用指南](#五环境使用指南)
- [六、参考资源](#六参考资源)

## 硬件环境

| 项目 | 配置 |
|------|------|
| 机器 | ASUS GX10 (NVIDIA DGX Spark) |
| GPU | NVIDIA GB10 (Blackwell, CUDA Capability 12.1) |
| 架构 | ARM64 (aarch64) |
| 驱动 | 580.95.05 |
| CUDA | 13.0 |

## 软件版本

由于 Transformers 和 vLLM 版本依赖不兼容，本项目提供两个独立的 conda 环境：

### 环境 1: deepseek-ocr (Transformers 推理)

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12.9 | conda 环境 |
| PyTorch | 2.9.0+cu130 | ARM64 + CUDA 13.0 |
| Transformers | 4.45.2 | 包含 LlamaFlashAttention2 |
| Tokenizers | 0.20.3 | 兼容 Transformers 4.45.2 |
| Attention | Eager | 标准实现（慢但稳定） |
| 运行脚本 | `run_ocr_cli.py --framework transformers` | |

**配置状态**: ✅ 已配置完成

### 环境 2: deepseek-ocr-vllm (vLLM 推理)

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12.9 | conda 环境 |
| PyTorch | 2.9.0+cu130 | ARM64 + CUDA 13.0 |
| Transformers | 4.57.3 | 包含 DeepseekV3Config |
| Tokenizers | 0.22.1 | 兼容新版 Transformers |
| vLLM | 0.11.3.dev0 | 从源码编译 (基于 v0.11.2) |
| Triton | 3.5.0 | vLLM 依赖 |
| Attention | Flash Attention | 高性能实现 |
| 运行脚本 | `run_ocr_cli.py --framework vllm` | |

**配置状态**: ✅ 已配置完成

### 环境 3: deepseek-ocr-70b-quant (量化 vLLM 推理)

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12.9 | conda 环境 |
| PyTorch | 2.9.0+cu130 | ARM64 + CUDA 13.0 |
| Transformers | 4.57.3 | 与 vLLM 环境一致 |
| Tokenizers | 0.22.1 | 与 vLLM 环境一致 |
| vLLM | 0.11.3.dev0+g275de3417.d20251204 | 源码编译，CUDA 13.0 |
| 量化/加速组件 | bitsandbytes 0.48.2；compressed-tensors 0.12.2；flashinfer 0.5.2；gguf 0.17.1；cupy-cuda12x 13.6.0 | 面向 70B 量化推理的核心依赖 |
| CUDA 库 | nvidia-cublas/cudnn/cusparselt/cu13 系列 | 与 GPU 驱动/Blackwell 兼容 |
| 运行脚本 | `run_ocr_cli.py --framework vllm` | 用于量化模型推理 |

**配置状态**: ✅ 已配置完成（专用于 70B 量化 vLLM）

### 环境对比

| 特性 | deepseek-ocr | deepseek-ocr-vllm |
|------|--------------|-------------------|
| **推理引擎** | Transformers | vLLM |
| **性能** | 较慢 | 更快 |
| **Attention 实现** | Eager | Flash Attention |
| **适用场景** | 开发测试 | 生产部署 |
| **并发能力** | 低 | 高 (173.81x @ 8K tokens) |
| **内存效率** | 一般 | 优秀 (KV cache 优化) |

---

## 快速验证

### 验证环境配置

验证两个环境是否配置正确：

**验证 deepseek-ocr 环境**：
```bash
conda activate deepseek-ocr
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import tokenizers; print(f'Tokenizers: {tokenizers.__version__}')"
```

预期输出：
```
PyTorch: 2.9.0+cu130, CUDA: 13.0
Transformers: 4.45.2
Tokenizers: 0.20.3
```

**验证 deepseek-ocr-vllm 环境**：
```bash
conda activate deepseek-ocr-vllm
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import tokenizers; print(f'Tokenizers: {tokenizers.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
```

预期输出：
```
PyTorch: 2.9.0+cu130, CUDA: 13.0
Transformers: 4.57.3
Tokenizers: 0.22.1
vLLM: 0.11.3.dev0+g275de3417.d20251204
```

### 验证 OCR 功能

**使用 Transformers 框架测试**：
```bash
conda activate deepseek-ocr
python run_ocr_cli.py --framework transformers --mode random --input test_resouce/sample1
```

**使用 vLLM 框架测试**：
```bash
conda activate deepseek-ocr-vllm
python run_ocr_cli.py --framework vllm --mode random --input test_resouce/sample1
```

如果运行成功，会在 `results/` 目录下生成带时间戳的结果文件夹，包含 OCR 识别的文本、Markdown 和带边界框的图片。

---

## 一、Transformers 推理环境 (deepseek-ocr)

### 1.1 创建 Conda 环境

```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

### 1.2 安装 PyTorch (CUDA 13.0 + ARM64)

**重要**: DGX Spark 使用 CUDA 13.0 + ARM64 架构，需要安装对应版本的 PyTorch。

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
```

验证安装：
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
```

预期输出：
```
PyTorch: 2.9.0+cu130, CUDA: 13.0, GPU: NVIDIA GB10
```

> ⚠️ **注意**: 会出现 CUDA capability 警告（12.1 vs 12.0），这是正常的，不影响使用。

### 1.3 安装 Transformers 和 Tokenizers

```bash
# 必须使用 4.45.2 版本，该版本包含 DeepSeek-OCR 模型代码所需的 LlamaFlashAttention2
pip install transformers==4.45.2 tokenizers==0.20.3
```

验证安装：
```bash
python -c "import transformers, tokenizers; print(f'Transformers: {transformers.__version__}, Tokenizers: {tokenizers.__version__}')"
```

预期输出：
```
Transformers: 4.45.2, Tokenizers: 0.20.3
```

### 1.4 安装基础依赖

```bash
pip install -r requirements.txt
```

### 1.5 配置环境变量（可选）

环境变量会在激活 conda 环境时自动设置。如果未自动设置，请手动创建激活脚本：

```bash
mkdir -p ~/miniconda3/envs/deepseek-ocr/etc/conda/activate.d
cat > ~/miniconda3/envs/deepseek-ocr/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
echo "✓ DeepSeek-OCR 环境变量已设置"
EOF
```

---

## 二、Transformers 推理使用

Transformers 推理相对简单，不需要编译 vLLM。依赖已包含在 `requirements.txt` 中。

### 2.1 关于 FlashAttention

在 DGX Spark (GB10) 上，FlashAttention 2.7.3 **无法正常编译**（Blackwell 架构支持问题）。

**解决方案**: 使用 `eager` attention 实现：

```python
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='eager',  # 不使用 flash_attention_2
    trust_remote_code=True,
    use_safetensors=True,
)
```

> ⚠️ `eager` 实现速度较慢，但功能完整。

### 2.2 运行 OCR 识别

使用统一的命令行工具 `run_ocr_cli.py`：

```bash
conda activate deepseek-ocr

# 随机处理一张图片（推荐用于测试）
python run_ocr_cli.py --framework transformers --mode random --input test_resouce/sample1

# 处理所有图片
python run_ocr_cli.py --framework transformers --mode all --input test_resouce/sample1
```

**test_resouce/sample1**是一个装着数张图片的文件夹

### 2.3 运行模式

| 模式 | base_size | image_size | crop_mode | vision tokens |
|------|-----------|------------|-----------|---------------|
| Tiny | 512 | 512 | False | 64 |
| Small | 640 | 640 | False | 100 |
| Base | 1024 | 1024 | False | 256 |
| **Large** | 1280 | 1280 | False | 400 |
| Gundam | 1024 | 640 | True | 动态 |

---

## 三、vLLM 推理环境 (deepseek-ocr-vllm)

### 3.1 为什么需要独立环境？

Transformers 和 vLLM 对依赖版本要求不兼容：

| 依赖 | Transformers 环境 | vLLM 环境 |
|------|------------------|----------|
| transformers | 4.45.2（含 LlamaFlashAttention2） | 4.56.0+（含 DeepseekV3Config） |
| tokenizers | 0.20.3 | 0.21.1+ |

因此需要创建独立的 `deepseek-ocr-vllm` 环境。

### 3.2 创建 Conda 环境

```bash
conda create -n deepseek-ocr-vllm python=3.12.9 -y
conda activate deepseek-ocr-vllm
```

### 3.3 安装 PyTorch (CUDA 13.0 + ARM64)

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
```

### 3.4 安装 Transformers 和 Tokenizers

```bash
# vLLM 需要较新版本的 transformers（会自动安装最新版本）
pip install 'transformers>=4.56.0' 'tokenizers>=0.21.1'
```

验证安装：
```bash
python -c "import transformers, tokenizers; print(f'Transformers: {transformers.__version__}, Tokenizers: {tokenizers.__version__}')"
```

预期输出：
```
Transformers: 4.57.3, Tokenizers: 0.22.1
```

### 3.5 安装基础依赖

```bash
pip install -r requirements.txt
```

### 3.6 配置环境变量

创建环境激活脚本：

```bash
mkdir -p ~/miniconda3/envs/deepseek-ocr-vllm/etc/conda/activate.d
cat > ~/miniconda3/envs/deepseek-ocr-vllm/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
echo "✓ DeepSeek-OCR-vLLM 环境变量已设置"
EOF
```

### 3.7 为什么需要源码编译 vLLM？

vLLM 官方预编译的 wheel 包是基于 **CUDA 12.x + x86_64** 的，在 DGX Spark 上会遇到：

1. **架构不匹配**: 预编译包是 x86_64，DGX Spark 是 ARM64 (aarch64)
2. **CUDA 版本不匹配**: 预编译包基于 CUDA 12.x，DGX Spark 是 CUDA 13.0
3. **符号版本问题**: 即使使用兼容层也无法解决 `libcudart.so.12` 符号版本问题

**解决方案**: 从源码编译 vLLM。

### 3.8 准备编译工具

```bash
# 确保已激活 vLLM 环境
conda activate deepseek-ocr-vllm

# 安装编译依赖
pip install cmake ninja pybind11 setuptools wheel setuptools_scm
```

### 3.9 获取 vLLM 源码

```bash
mkdir -p ~/vllm-install
cd ~/vllm-install
git clone --recursive https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.11.2
git submodule update --init --recursive
```

### 3.10 修复 pyproject.toml

vLLM v0.11.2 的 `pyproject.toml` 需要修复 license 字段格式：

```bash
cd ~/vllm-install/vllm
sed -i 's/^license = "Apache-2.0"$/license = {text = "Apache-2.0"}/' pyproject.toml
sed -i '/^license-files = /d' pyproject.toml
```

### 3.11 编译安装

```bash
cd ~/vllm-install/vllm

# 编译安装（约 15-20 分钟）
# 注意：环境变量已在 conda 环境激活时自动设置，无需手动 export
pip install --no-build-isolation -e .
```

### 3.12 验证安装

```bash
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor; print('DeepSeek-OCR support: OK')"
```

预期输出：
```
vLLM: 0.11.3.dev0+g275de3417.d20251204
DeepSeek-OCR support: OK
```

> **关于版本号**: checkout `v0.11.2` 但显示 `0.11.3.dev0` 是正常的。vLLM 使用 `setuptools_scm` 从 git 自动生成版本号，格式为 `{next_version}.dev{distance}+g{commit}`。v0.11.2 tag 之后的下一个版本是 0.11.3，所以显示为 dev 版本。

### 3.13 运行 OCR 识别

使用统一的命令行工具 `run_ocr_cli.py`：

```bash
conda activate deepseek-ocr-vllm

# 随机处理一张图片（推荐用于测试）
python run_ocr_cli.py --framework vllm --mode random --input test_resouce/sample1

# 处理所有图片（推荐用于批量处理）
python run_ocr_cli.py --framework vllm --mode all --input test_resouce/sample1
```

**vLLM 默认使用 Gundam 模式**（硬编码在源码中），适合处理大尺寸文档图片。

---

## 四、量化 vLLM 环境 (deepseek-ocr-70b-quant)

专门用于 70B 量化模型的 vLLM 推理环境，基于源码编译的 CUDA 13.0 版本，并预置常用量化/加速组件。

- 核心版本：Python 3.12.9；PyTorch 2.9.0+cu130；vLLM 0.11.3.dev0+g275de3417.d20251204；Transformers 4.57.3；Tokenizers 0.22.1。
- 量化/加速组件：bitsandbytes 0.48.2、compressed-tensors 0.12.2、flashinfer 0.5.2、gguf 0.17.1、cupy-cuda12x 13.6.0；CUDA 13.0 的 nvidia-cu*、cudnn、cusparselt 库已就位。
- 辅助工具：accelerate 1.12.0、optimum 2.0.0、tiktoken 0.12.0 等，便于量化权重加载与高吞吐推理。
- 使用方式：`conda activate deepseek-ocr-70b-quant` 后与 `deepseek-ocr-vllm` 相同，直接运行 `python run_ocr_cli.py --framework vllm ...`。
- 快速校验（可选）：
```bash
conda activate deepseek-ocr-70b-quant
python - <<'PY'
import torch, vllm, bitsandbytes, flashinfer
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("flashinfer:", flashinfer.__version__)
PY
```

---

## 五、环境使用指南

### 5.1 如何选择环境

| 场景 | 推荐环境 | 原因 |
|------|---------|------|
| 开发调试 | deepseek-ocr | 简单直接，便于调试 |
| 生产部署 | deepseek-ocr-vllm | 性能更好，支持高并发 |
| 单张图片处理 | deepseek-ocr | 启动快，无需预热 |
| 批量处理 | deepseek-ocr-vllm | 吞吐量高，内存效率好 |
| 首次使用 | deepseek-ocr | 配置简单，依赖少 |

### 5.2 环境切换和使用

```bash
# 切换到 Transformers 环境
conda activate deepseek-ocr
python run_ocr_cli.py --framework transformers --mode random --input test_resouce/sample1

# 切换到 vLLM 环境
conda activate deepseek-ocr-vllm
python run_ocr_cli.py --framework vllm --mode all --input test_resouce/sample1

# 切换到量化 vLLM 环境（同样使用 vllm 框架）
conda activate deepseek-ocr-70b-quant
python run_ocr_cli.py --framework vllm --mode all --input test_resouce/sample1
```

**命令行参数说明**：
- `--framework`: 必选，指定使用的框架（`transformers` 或 `vllm`）
- `--mode`: 可选，工作模式（`random` 随机选择1张，`all` 处理所有图片，默认 `random`）
- `--input`: 可选，输入目录路径（默认 `test_resouce/sample1`）
- `--output`: 可选，输出基础目录（默认 `results`）

查看完整帮助：
```bash
python run_ocr_cli.py --help
```

### 5.3 环境维护

**查看已安装的环境**：
```bash
conda env list
```

**更新依赖**：
```bash
# Transformers 环境
conda activate deepseek-ocr
pip install -r requirements.txt --upgrade

# vLLM 环境
conda activate deepseek-ocr-vllm
pip install -r requirements.txt --upgrade
```

**删除环境**（如需重新配置）：
```bash
conda remove -n deepseek-ocr --all
conda remove -n deepseek-ocr-vllm --all
conda remove -n deepseek-ocr-70b-quant --all
```

### 5.4 环境变量说明

三套环境共享同一组 CUDA/Triton 变量（在各自的 conda activate 脚本中设置）：

| 变量名 | 值 | 作用 |
|--------|-----|------|
| `TORCH_CUDA_ARCH_LIST` | `12.1a` | 指定 CUDA 架构（GB10） |
| `TRITON_PTXAS_PATH` | `/usr/local/cuda/bin/ptxas` | Triton 编译器路径 |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `1` | 允许 vLLM 运行时更新 |

这些变量用于解决 Blackwell 架构（GB10）的 Triton 编译问题。

---

## 六、参考资源

- [DeepSeek-OCR 官方仓库](https://github.com/deepseek-ai/DeepSeek-OCR)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM DeepSeek-OCR 支持](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- [HuggingFace 模型](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [论文 (arXiv)](https://arxiv.org/abs/2510.18234)

---

## i18n

### English Summary

**DeepSeek-OCR on NVIDIA DGX Spark (ASUS GX10)**

This repository is a fork of [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), optimized for running on **NVIDIA DGX Spark (ASUS GX10)** with native ARM64 + CUDA 13.0 support.

**Key Features:**
- ✅ Three conda environments: Transformers, vLLM, and quantized vLLM (70B)
- ✅ Full support for NVIDIA GB10 (Blackwell architecture, CUDA Capability 12.1)
- ✅ Pre-configured environment variables for Triton compilation
- ✅ Unified CLI tool (`run_ocr_cli.py`) supporting both frameworks
- ✅ Batch processing and random sampling modes

**Hardware Requirements:**
- Machine: ASUS GX10 (NVIDIA DGX Spark)
- GPU: NVIDIA GB10 (Blackwell, CUDA Capability 12.1)
- Architecture: ARM64 (aarch64)
- CUDA: 13.0

**Software Stack:**
- **deepseek-ocr**: Transformers 4.45.2, Python 3.12.9, PyTorch 2.9.0+cu130
- **deepseek-ocr-vllm**: Transformers 4.57.3, vLLM 0.11.3.dev0 (compiled from source), Python 3.12.9
- **deepseek-ocr-70b-quant**: Same vLLM stack + quantization helpers (bitsandbytes, compressed-tensors, flashinfer, gguf)

**Quick Start:**
```bash
# Transformers framework
conda activate deepseek-ocr
python run_ocr_cli.py --framework transformers --mode random

# vLLM framework
conda activate deepseek-ocr-vllm
python run_ocr_cli.py --framework vllm --mode all
```

---

### 日本語概要

**NVIDIA DGX Spark (ASUS GX10) での DeepSeek-OCR**

このリポジトリは [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) のフォークで、**NVIDIA DGX Spark (ASUS GX10)** 上で ARM64 + CUDA 13.0 のネイティブサポートで動作するように最適化されています。

**主な特徴:**
- ✅ Transformers と vLLM 推論用の2つの独立した conda 環境
- ✅ NVIDIA GB10 (Blackwell アーキテクチャ、CUDA Capability 12.1) の完全サポート
- ✅ Triton コンパイル用の事前設定済み環境変数
- ✅ 両フレームワークをサポートする統一 CLI ツール (`run_ocr_cli.py`)
- ✅ バッチ処理とランダムサンプリングモード

**ハードウェア要件:**
- マシン: ASUS GX10 (NVIDIA DGX Spark)
- GPU: NVIDIA GB10 (Blackwell、CUDA Capability 12.1)
- アーキテクチャ: ARM64 (aarch64)
- CUDA: 13.0

**ソフトウェアスタック:**
- **deepseek-ocr**: Transformers 4.45.2、Python 3.12.9、PyTorch 2.9.0+cu130
- **deepseek-ocr-vllm**: Transformers 4.57.3、vLLM 0.11.3.dev0 (ソースからコンパイル)、Python 3.12.9

**クイックスタート:**
```bash
# Transformers フレームワーク
conda activate deepseek-ocr
python run_ocr_cli.py --framework transformers --mode random

# vLLM フレームワーク
conda activate deepseek-ocr-vllm
python run_ocr_cli.py --framework vllm --mode all
```
