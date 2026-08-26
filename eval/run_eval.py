"""评测脚本：加载评测集，运行 agent，计算意图准确率 / 检索命中率 / 答案覆盖 / 转人工准确率。

用法（在项目根目录执行）：
    python eval/run_eval.py                # 使用当前配置（无 Key 时自动 Mock）
    python eval/run_eval.py --out reports/评测报告.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# 保证可从任意 cwd 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs_agent.app import build_agent  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "eval_set.jsonl"


def load_cases(path: Path):
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run(agent, cases):
    results = []
    for i, c in enumerate(cases):
        r = agent.run(c["input"], session_id=f"eval-{c['id']}")
        results.append(
            {
                "id": c["id"],
                "input": c["input"],
                "intent_expected": c["intent"],
                "intent_predicted": r.get("intent"),
                "expected_doc": c.get("expected_doc"),
                "sources": list(r.get("sources", [])),
                "expected_contains": c.get("expected_contains", []),
                "answer": r.get("answer", ""),
                "should_ticket": c["should_ticket"],
                "has_ticket": r.get("ticket") is not None,
            }
        )
    return results


def evaluate(results):
    n = len(results)

    intent_ok = sum(1 for r in results if r["intent_predicted"] == r["intent_expected"])
    intent_acc = intent_ok / n if n else 0.0

    retrieval_cases = [r for r in results if r["expected_doc"]]
    retrieval_ok = sum(1 for r in retrieval_cases if r["expected_doc"] in r["sources"])
    retrieval_hit = retrieval_ok / len(retrieval_cases) if retrieval_cases else 0.0

    contain_cases = [r for r in results if r["expected_contains"]]
    contain_ok = sum(
        1 for r in contain_cases
        if all(k in r["answer"] for k in r["expected_contains"])
    )
    contain_acc = contain_ok / len(contain_cases) if contain_cases else 0.0

    ticket_ok = sum(1 for r in results if r["has_ticket"] == r["should_ticket"])
    ticket_acc = ticket_ok / n if n else 0.0

    return {
        "total": n,
        "intent_accuracy": intent_acc,
        "intent_ok": intent_ok,
        "retrieval_hit": retrieval_hit,
        "retrieval_ok": retrieval_ok,
        "retrieval_n": len(retrieval_cases),
        "answer_containment": contain_acc,
        "contain_ok": contain_ok,
        "contain_n": len(contain_cases),
        "ticket_accuracy": ticket_acc,
        "ticket_ok": ticket_ok,
    }


def render_report(metrics, results, provider_name, out_path: Path):
    lines = []
    lines.append("# 智能客服 Agent 评测报告\n")
    lines.append(f"- 生成时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    lines.append(f"- LLM 后端：`{provider_name}`")
    lines.append(f"- 评测样本数：{metrics['total']}\n")

    lines.append("## 总览\n")
    lines.append("| 指标 | 得分 | 说明 |")
    lines.append("| ---- | ---- | ---- |")
    lines.append(f"| 意图识别准确率 | {metrics['intent_accuracy']:.2%} | {metrics['intent_ok']}/{metrics['total']} |")
    lines.append(f"| 检索命中率（Hit@TopK） | {metrics['retrieval_hit']:.2%} | {metrics['retrieval_ok']}/{metrics['retrieval_n']} |")
    lines.append(f"| 答案关键信息覆盖 | {metrics['answer_containment']:.2%} | {metrics['contain_ok']}/{metrics['contain_n']} |")
    lines.append(f"| 转人工准确率 | {metrics['ticket_accuracy']:.2%} | {metrics['ticket_ok']}/{metrics['total']} |")
    lines.append("")

    lines.append("## 逐样本结果\n")
    lines.append("| ID | 输入 | 期望意图 | 预测意图 | 期望文档 | 检索命中 | 关键信息覆盖 | 期望转人工 | 实际转人工 |")
    lines.append("| -- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |")
    for r in results:
        doc = r["expected_doc"] or "-"
        if r["expected_doc"]:
            hit = "✅" if r["expected_doc"] in r["sources"] else "❌"
        else:
            hit = "-"
        if r["expected_contains"]:
            covered = all(k in r["answer"] for k in r["expected_contains"])
            contain = "✅" if covered else "❌"
        else:
            contain = "-"
        ticket_mark = "✅" if r["has_ticket"] == r["should_ticket"] else "❌"
        lines.append(
            f"| {r['id']} | {r['input']} | {r['intent_expected']} | {r['intent_predicted']} "
            f"| {doc} | {hit} | {contain} | {'是' if r['should_ticket'] else '否'} | "
            f"{'是' if r['has_ticket'] else '否'}{ticket_mark} |"
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/评测报告.md")
    parser.add_argument("--real", action="store_true", help="使用配置的真实 LLM（默认离线 Mock 基准）")
    args = parser.parse_args()

    cases = load_cases(EVAL_SET)
    agent, _, _ = build_agent(force_mock=not args.real)
    results = run(agent, cases)
    metrics = evaluate(results)

    summary = (
        f"意图准确率 {metrics['intent_accuracy']:.2%} | "
        f"检索命中 {metrics['retrieval_hit']:.2%} | "
        f"答案覆盖 {metrics['answer_containment']:.2%} | "
        f"转人工 {metrics['ticket_accuracy']:.2%}"
    )
    print(f"[评测] LLM={agent.provider.name} 样本={metrics['total']}")
    print(summary)

    out_path = Path(args.out)
    render_report(metrics, results, agent.provider.name, out_path)
    print(f"[评测] 报告已写入 {out_path}")


if __name__ == "__main__":
    main()
