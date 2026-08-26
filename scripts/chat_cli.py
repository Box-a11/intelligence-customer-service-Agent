"""终端多轮对话测试脚本：连接本地 FastAPI 服务，观察意图识别、澄清与转人工。

用法（服务启动后）：
    .venv/Scripts/python scripts/chat_cli.py
    输入消息回车发送；输入 exit / quit / q 退出；输入 reset 开启新会话。

依赖 httpx（已随 requirements.txt 安装）。
"""
from __future__ import annotations

import httpx

BASE = "http://127.0.0.1:8000"

INTENT_LABELS = {
    "qa": "问答",
    "consultation": "咨询",
    "complaint": "投诉",
    "unclear": "模糊",
}


def main():
    print("=" * 60)
    print("智能客服多轮对话测试  (输入 exit 退出，reset 开启新会话)")
    print("=" * 60)
    session_id = "cli-" + str(id(object()))
    print(f"[会话] session_id={session_id}\n")

    with httpx.Client(timeout=30) as client:
        while True:
            try:
                text = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not text:
                continue
            if text.lower() in {"exit", "quit", "q"}:
                print("再见！")
                break
            if text.lower() == "reset":
                session_id = "cli-" + str(id(object()))
                print(f"[新会话] session_id={session_id}\n")
                continue

            resp = client.post(f"{BASE}/chat", json={"message": text, "session_id": session_id})
            data = resp.json()

            intent = data.get("intent") or "-"
            print(f"客服: {data['reply']}")
            print(f"      [意图={INTENT_LABELS.get(intent, intent)} "
                  f"澄清={'是' if data.get('needs_clarification') else '否'} "
                  f"工单={'是' if data.get('ticket') else '否'}]")
            if data.get("ticket"):
                print(f"      [工单号={data['ticket']['id']} 状态={data['ticket']['status']}]")
            if data.get("sources"):
                print(f"      [来源={data['sources']}]")
            print()


if __name__ == "__main__":
    main()
