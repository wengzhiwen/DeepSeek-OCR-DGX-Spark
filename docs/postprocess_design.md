# 后处理 功能设计

> 环境构建与依赖请直接参考仓库根目录的 README；本文只描述后处理功能和流程。

## 目标
- 对 OCR Markdown 结果进行语法修正与上下文合并，生成可阅读的单一 Markdown 文档。
- 记录 token 使用（CSV `token.log`）与性能摘要，产出处理报告。

## 输入 / 输出
- 输入：`--input` 指向的 OCR 结果目录（包含分页面的 `.md` 文件），通常为识别阶段的 `results/{timestamp}/{framework}/`。
- 输出：默认写入 `--input/postprocessed/`（可用 `--output` 指定）：
  - `postprocessed_result.md`：合并后的正文（按页顺序，分节 `## page_name`）。
  - `postprocess_report.md`：每页语法修正/合并状态表。
  - `token.log`：CSV 格式的 token/耗时流水。
  - `token_summary.json`：汇总统计。

## 核心流程（`process_ocr_results`）
1) 初始化：打印上下文信息；`init_token_logger(output_dir)` 落 token.csv；列出所有 Markdown 页。
2) 引擎：构造 `PostProcessVLLM`（模型路径、并行、量化、加载格式、是否强制 eager 等），调用 `initialize()`。
3) 语法修正（前瞻队列）：
   - 读取页面文本，`correct_grammar_async` 最多重试 2 次；失败时回退原文并标记原因。
   - 预填充 `context_after+1` 的修正版，形成 lookahead 队列。
4) 上下文合并：
   - 基于窗口 `context_before/context_after` 取历史与前瞻页面，估算 token 并按模型窗口截断上下文。
   - 调用 `merge_with_context_async` 最多重试 2 次，失败则保留当前内容。
5) 增量写出：
   - 处理顺序推进；每页写入 `postprocessed_result.md`（页间用 `---` 分隔）。
   - 记录 `report_entries`（语法/合并状态与原因）。
6) 收尾：
   - 生成 `postprocess_report.md`（表格化状态）。
   - `token_logger.save_summary()` 输出 `token_summary.json` 与控制台摘要。

## token 记录
- 文件：`token.log`（CSV，字段含 timestamp/request_id/step/page/prompt_tokens/completion_tokens/total_tokens/end_time/duration_sec/tokens_per_sec）。
- 兼容旧 JSON 行格式：初始化时会自动检测并迁移为 CSV。
- 汇总：`get_summary()` 解析 CSV 汇总后由 `save_summary()` 写入 JSON。

## CLI 关键参数（`run_postprocess_cli.py`）
- `--model` (必需)：本地模型路径或别名；`--model-preset` 必选（`8b` / `70b`）。
- `--context-before/--context-after`：上下文窗口大小。
- `--tensor-parallel-size`：张量并行；`--load-format`：模型加载格式；`--quant-method`：量化方式（bitsandbytes/gptq/awq，主要用于 70B）。
- `--enforce-eager`：编译失败时的兜底模式（降速换稳定）。
- `--no-progress`：关闭生成时的流式打印。

## 扩展/修改建议
- 调整步骤：可在 `PostProcessStep` 基类基础上添加新步骤，并在流程中插入（例如术语替换、格式修正）。
- 上下文策略：当前按估算 token 截断并分配前/后上下文，可替换为更精细的分片或召回策略。
- 日志：如需对接外部监控，可在 `log_request` 处追加自定义落盘/上报逻辑。
