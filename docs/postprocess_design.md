# 后处理 功能设计

> 环境构建与依赖请直接参考仓库根目录的 README；本文只描述后处理功能和流程。

## 目标
- 对 OCR Markdown 结果逐页进行语法修正，并通过“修正提案”驱动的行级应用最小化对原文的破坏。
- 记录 token 使用（CSV `token.log`）与性能摘要，产出处理报告与修正提案日志。

## 输入 / 输出
- 输入：`--input` 指向的 OCR 结果目录（包含分页面的 `.md` 文件），通常为识别阶段的 `results/{timestamp}/{framework}/`。
- 输出：默认写入 `--input/postprocessed/`（可用 `--output` 指定）：
  - `postprocessed_result.md`：合并后的正文（按页顺序，分节 `## page_name`）。
  - `postprocess_report.md`：每页语法修正/合并状态表。
  - `token.log`：CSV 格式的 token/耗时流水。
  - `token_summary.json`：汇总统计。

## 核心流程（`process_ocr_results`）
1) 初始化：打印输入输出信息；`init_token_logger(output_dir)` 落 token.csv；列出所有 Markdown 页。
2) 引擎：构造 `PostProcessVLLM`（模型路径、并行、量化、加载格式、是否强制 eager 等），调用 `initialize()`。
3) 语法修正（行级提案）：
   - 对每页生成“修正提案”列表（JSON 数组），包含 `line`（1-based）、`find`、`replace`、`reason`。
   - 对包含 `<table` 等富文本段落的页跳过 LLM，直接清理特殊标记后使用原文。
   - 由 Python 按行+子串应用提案：行号越界或未匹配则跳过并记录错误，保持原行不变；不允许改变行数。
   - 语法修正结果立即写入输出文件（不添加页眉/分隔符）。
   - 修正提案与应用结果落盘到 `postprocess_fixes.jsonl`，便于审计。
4) 收尾：
   - 生成 `postprocess_report.md`（表格化状态）。
   - `token_logger.save_summary()` 输出 `token_summary.json` 与控制台摘要。

## token 记录
- 文件：`token.log`（CSV，字段含 timestamp/request_id/step/page/prompt_tokens/completion_tokens/total_tokens/end_time/duration_sec/tokens_per_sec）。
- 兼容旧 JSON 行格式：初始化时会自动检测并迁移为 CSV。
- 汇总：`get_summary()` 解析 CSV 汇总后由 `save_summary()` 写入 JSON。

## CLI 关键参数（`run_postprocess_cli.py`）
- `--model` (必需)：本地模型路径或别名；`--model-preset` 必选（`8b` / `32b` / `70b`）。
- `--tensor-parallel-size`：张量并行；`--load-format`：模型加载格式；`--quant-method`：量化方式（bitsandbytes/gptq/awq，主要用于 70B）。
- `--enforce-eager`：编译失败时的兜底模式（降速换稳定）。
- `--no-progress`：关闭生成时的流式打印。

### 32B 预设说明
- 适配 128G 统一内存，默认 fp16 / 无量化；`gpu_memory_utilization=0.90`，`max_model_len=12288`，`max_num_batched_tokens=1536`。
- 采样参数强调幻觉抑制：`temperature=0.08, top_p=0.88, repetition_penalty=1.25, frequency_penalty=0.8`，输出上限 `max_tokens=1100`。
- 使用示例：
```bash
python run_postprocess_cli.py \
  --input results/20251205_143115/vLLM \
  --output results/20251205_143115/postprocessed \
  --model models/Flux-Japanese-Qwen2.5-32B-Instruct-V1.0 \
  --model-preset 32b \
  --tensor-parallel-size 1 \
  --load-format auto
```

## 扩展/修改建议
- 调整步骤：可在 `PostProcessStep` 基类基础上添加新步骤，并在流程中插入（例如术语替换、格式修正）。
- 日志：如需对接外部监控，可在 `log_request` 处追加自定义落盘/上报逻辑。
