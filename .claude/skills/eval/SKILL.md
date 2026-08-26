---
name: eval
description: 运行智能客服 Agent 的离线评测，输出意图准确率/检索命中/答案覆盖/转人工 4 项指标报告。
---

# 评测智能客服 Agent

在项目根目录执行离线评测（默认 Mock，零成本、确定性）：

```bash
.venv/Scripts/python eval/run_eval.py
```

输出 4 项指标并写入 `reports/评测报告.md`：

- 意图识别准确率
- 检索命中率（Hit@TopK）
- 答案关键信息覆盖
- 转人工准确率

## 用真实模型评测

会调用 DeepSeek（每轮 2~15 秒，较慢），指定输出路径：

```bash
.venv/Scripts/python eval/run_eval.py --real --out reports/评测报告.md
```

## 评测集

`eval/eval_set.jsonl`（26 条，覆盖 4 类意图 + 多跳 + 转人工）。新增用例按 JSONL 一行一条追加，字段：

```json
{"id": "qa-012", "input": "问题", "intent": "qa", "expected_doc": "04-退款说明", "expected_contains": ["退款"], "should_ticket": false}
```

- `expected_doc`：期望命中的文档 source（文件名，无扩展名）；无期望填 `null`。
- `expected_contains`：答案必须包含的关键词列表。
- `should_ticket`：是否应生成工单。

## 注意

- 评测默认走离线 Mock + `hash_embed`（确定性）；加用例后若指标下降，先检查用例意图/期望文档是否与知识库一致。
