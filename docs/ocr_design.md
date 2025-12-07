# OCR 功能设计

> 环境构建与依赖请直接参考仓库根目录的 README；本文只描述功能和流程。

## 目标
- 通过统一的 CLI 入口支持两套推理栈（Transformers 与 vLLM），处理单张或批量图片。
- 输出结构化的识别结果，便于后续后处理和审阅。

## 输入 / 输出
- 输入：`--input` 指向的图片目录（支持 `.png/.jpg/.jpeg`），可随机选取一张或处理全部。
- 输出：在 `results/{timestamp}/` 下按框架分目录：
  - `Transformers/图片名/`：原图拷贝、识别文本等。
  - `vLLM/图片名/`：原图拷贝、识别文本等。
- 结果路径与元信息由 `run_ocr_cli.py` 统一管理。

## 核心流程
1) CLI 解析：框架选择（`--framework transformers|vllm`）、模式（`--mode random|all`）、提示词模式与语言。
2) 文件收集：`get_image_files` 按文件名排序收集支持的图片，必要时随机抽取一张。
3) Prompt 构建：`build_prompt(prompt_mode, language)` 生成指令（基础/增强），传入模型。
4) 推理：
   - Transformers 路径：`TransformersOCR.initialize()` -> `process_image()` 逐图调用。
   - vLLM 路径：`VLLMOCR.initialize()` -> `_process_image_async()`，同一事件循环内串行 await，避免多 loop 冲突。
5) 结果落盘：为每张图创建子目录，拷贝原图并写出识别文本（Markdown/纯文本，取决于实现）。
6) 日志与容错：每张图独立 try/except，错误会记录到结果列表并打印，确保批处理不中断。

## 重要模块
- `run_ocr_cli.py`：统一入口，处理参数、路径、模式选择，调度两套框架。
- `ocr_transformers.py`：Transformers 实现（模型加载、推理、图像预处理、提示词支持）。
- `ocr_vllm.py`：vLLM 实现（引擎初始化、异步推理、生成流式打印）。
- `ocr_utils.py` / `ngram_norepeat.py`：辅助工具，如重复惩罚、预处理工具等。

## 配置与扩展点
- Prompt：`prompt_mode`/`language` 影响 `build_prompt`，可扩展更多模板。
- 模式：`random` 适合快速验证；`all` 适合批量跑全量数据。
- 设备选择：`CUDA_VISIBLE_DEVICES` 通过环境变量控制（脚本内默认 `0`，可按需覆盖）。
- 结果组织：如需添加 JSON/结构化输出，可在对应框架的处理函数内增加落盘步骤。

## 运行示例（功能验证）
```bash
# Transformers
python run_ocr_cli.py --framework transformers --mode random --input test_resouce/sample1

# vLLM
python run_ocr_cli.py --framework vllm --mode all --input test_resouce/sample1
```
