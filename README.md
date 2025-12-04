# DeepSeek-OCR on DGX Spark (ASUS GX10)

本仓库是 [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 的 Fork，专注于在 **NVIDIA DGX Spark (ASUS GX10)** 上配置可运行的原生环境。

> 📖 **原项目**: [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) | [论文](https://arxiv.org/abs/2510.18234) | [HuggingFace 模型](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

## 硬件环境

| 项目 | 配置 |
|------|------|
| 机器 | ASUS GX10 (NVIDIA DGX Spark) |
| GPU | NVIDIA GB10 (Blackwell, CUDA Capability 12.1) |
| 架构 | ARM64 (aarch64) |
| 驱动 | 580.95.05 |
| CUDA | 13.0 |

## 软件版本

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12.9 | conda 环境 |
| PyTorch | 2.9.0+cu130 | ARM64 + CUDA 13.0 |
| Transformers | 4.57.3 | |
| vLLM | 0.11.2 | 从源码编译 |
| Triton | 3.5.0 | |

---

## 一、通用环境配置

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

### 1.3 安装基础依赖

```bash
pip install -r requirements.txt
```

---

## 二、Transformers 推理

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

### 2.2 运行模式

| 模式 | base_size | image_size | crop_mode | vision tokens |
|------|-----------|------------|-----------|---------------|
| Tiny | 512 | 512 | False | 64 |
| Small | 640 | 640 | False | 100 |
| Base | 1024 | 1024 | False | 256 |
| **Large** | 1280 | 1280 | False | 400 |
| Gundam | 1024 | 640 | True | 动态 |

---

## 三、vLLM 推理（从源码编译）

### 3.1 为什么需要源码编译？

vLLM 官方预编译的 wheel 包是基于 **CUDA 12.x + x86_64** 的，在 DGX Spark 上会遇到：

1. **架构不匹配**: 预编译包是 x86_64，DGX Spark 是 ARM64 (aarch64)
2. **CUDA 版本不匹配**: 预编译包基于 CUDA 12.x，DGX Spark 是 CUDA 13.0
3. **符号版本问题**: 即使使用兼容层也无法解决 `libcudart.so.12` 符号版本问题

**解决方案**: 从源码编译 vLLM。

### 3.2 准备工作

```bash
# 确保已激活 conda 环境
conda activate deepseek-ocr

# 安装编译依赖
pip install cmake ninja pybind11 setuptools wheel setuptools_scm
```

### 3.3 获取 vLLM 源码

```bash
mkdir -p ~/vllm-install
cd ~/vllm-install
git clone --recursive https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.11.2
git submodule update --init --recursive
```

### 3.4 修复 pyproject.toml

vLLM v0.11.2 的 `pyproject.toml` 需要修复 license 字段格式：

```bash
cd ~/vllm-install/vllm
sed -i 's/^license = "Apache-2.0"$/license = {text = "Apache-2.0"}/' pyproject.toml
sed -i '/^license-files = /d' pyproject.toml
```

### 3.5 编译安装

```bash
cd ~/vllm-install/vllm

# 设置编译环境变量
export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# 编译安装（约 15-20 分钟）
pip install --no-build-isolation -e .
```

### 3.6 验证安装

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

**vLLM 默认使用 Gundam 模式**（硬编码在源码中）：

---

## 四、参考资源

- [DeepSeek-OCR 官方仓库](https://github.com/deepseek-ai/DeepSeek-OCR)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM DeepSeek-OCR 支持](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- [HuggingFace 模型](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [论文 (arXiv)](https://arxiv.org/abs/2510.18234)
