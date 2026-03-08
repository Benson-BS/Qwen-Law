"""
Compare base Qwen2.5-3B vs LoRA fine-tuned (Qwen-Law) model on legal questions.

Outputs:
  - comparison_results.json   : structured data for programmatic use
  - comparison_results.html   : visual side-by-side report (open in browser)
  - comparison_results.md     : markdown report

Usage: python compare.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "./qwen2.5-3b"
LORA_ADAPTER_PATH = "./saves/qwen3-4b/lora/sft"
OUTPUT_DIR = Path("./results")

LEGAL_QUESTIONS = [
    "什么是正当防卫？正当防卫的构成要件有哪些？",
    "劳动者在试用期内被公司无故辞退，应该如何维权？",
    "借钱给朋友没有写借条，现在对方不还钱，我该怎么办？",
    "交通事故中，双方对责任认定有异议，应该怎么处理？",
    "租房合同未到期，房东要求提前收回房屋，租客有什么权利？",
    "网购商品出现质量问题，商家拒绝退货，消费者如何维权？",
    "遗产继承中，遗嘱和法定继承发生冲突时以哪个为准？",
    "公司拖欠员工工资超过三个月，员工可以采取哪些法律手段？",
]

GENERATION_CONFIG = dict(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

def load_base_model(path: str):
    print(f"[INFO] Loading base model from {path} ...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_lora_model(base_model, adapter_path: str):
    print(f"[INFO] Loading LoRA adapter from {adapter_path} ...")
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    lora_model.eval()
    return lora_model


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": "你是一个专业的中国法律助手，请根据中国现行法律法规回答用户的问题。"},
        {"role": "user", "content": question},
    ]


@torch.no_grad()
def generate(model, tokenizer, question: str) -> tuple[str, float]:
    """Returns (answer_text, elapsed_seconds)."""
    messages = build_prompt(question)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    t0 = time.perf_counter()
    output_ids = model.generate(**inputs, **GENERATION_CONFIG, pad_token_id=pad_token_id)
    elapsed = time.perf_counter() - t0

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, elapsed


# ---------------------------------------------------------------------------
#  Save: JSON
# ---------------------------------------------------------------------------

def save_json(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON saved to {path}")


# ---------------------------------------------------------------------------
#  Save: Markdown
# ---------------------------------------------------------------------------

def save_markdown(records: list[dict], path: Path):
    lines: list[str] = []
    lines.append("# Qwen-Law 微调前后对比报告\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    total_base_time = sum(r["base_time"] for r in records)
    total_lora_time = sum(r["lora_time"] for r in records)
    avg_base_len = sum(len(r["base_answer"]) for r in records) / len(records)
    avg_lora_len = sum(len(r["lora_answer"]) for r in records) / len(records)

    lines.append("## 总体统计\n")
    lines.append(f"| 指标 | Base Qwen2.5-3B | Qwen-Law (LoRA) |")
    lines.append(f"|------|-----------------|-----------------|")
    lines.append(f"| 总生成时间 | {total_base_time:.1f}s | {total_lora_time:.1f}s |")
    lines.append(f"| 平均回答长度 | {avg_base_len:.0f} 字 | {avg_lora_len:.0f} 字 |")
    lines.append("")

    for i, r in enumerate(records, 1):
        lines.append(f"---\n")
        lines.append(f"## Q{i}: {r['question']}\n")
        lines.append(f"### Base Model（耗时 {r['base_time']:.1f}s，{len(r['base_answer'])} 字）\n")
        lines.append(f"{r['base_answer']}\n")
        lines.append(f"### Qwen-Law LoRA（耗时 {r['lora_time']:.1f}s，{len(r['lora_answer'])} 字）\n")
        lines.append(f"{r['lora_answer']}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Markdown saved to {path}")


# ---------------------------------------------------------------------------
#  Save: HTML (visual side-by-side comparison)
# ---------------------------------------------------------------------------

def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def save_html(records: list[dict], path: Path):
    total_base_time = sum(r["base_time"] for r in records)
    total_lora_time = sum(r["lora_time"] for r in records)
    avg_base_len = sum(len(r["base_answer"]) for r in records) / len(records)
    avg_lora_len = sum(len(r["lora_answer"]) for r in records) / len(records)

    cards_html = ""
    for i, r in enumerate(records, 1):
        base_esc = _html_escape(r["base_answer"])
        lora_esc = _html_escape(r["lora_answer"])
        cards_html += f"""
        <div class="card">
            <div class="question">Q{i}: {_html_escape(r['question'])}</div>
            <div class="answers">
                <div class="answer base">
                    <div class="label base-label">Base Qwen2.5-3B</div>
                    <div class="meta">耗时 {r['base_time']:.1f}s · {len(r['base_answer'])} 字</div>
                    <div class="text">{base_esc}</div>
                </div>
                <div class="answer lora">
                    <div class="label lora-label">Qwen-Law (LoRA)</div>
                    <div class="meta">耗时 {r['lora_time']:.1f}s · {len(r['lora_answer'])} 字</div>
                    <div class="text">{lora_esc}</div>
                </div>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen-Law 微调前后对比报告</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
           background: #f0f2f5; color: #333; padding: 24px; }}
    h1 {{ text-align: center; margin-bottom: 8px; font-size: 28px; }}
    .subtitle {{ text-align: center; color: #888; margin-bottom: 24px; font-size: 14px; }}

    .stats {{ display: flex; justify-content: center; gap: 32px; margin-bottom: 32px; flex-wrap: wrap; }}
    .stat-box {{ background: #fff; border-radius: 12px; padding: 18px 28px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; min-width: 200px; }}
    .stat-box .stat-title {{ font-size: 13px; color: #999; margin-bottom: 6px; }}
    .stat-box .stat-row {{ display: flex; justify-content: center; gap: 24px; }}
    .stat-box .stat-val {{ font-size: 22px; font-weight: 700; }}
    .stat-box .stat-label {{ font-size: 11px; color: #aaa; }}
    .base-color {{ color: #e67e22; }}
    .lora-color {{ color: #27ae60; }}

    .card {{ background: #fff; border-radius: 12px; margin-bottom: 24px;
             box-shadow: 0 2px 8px rgba(0,0,0,0.06); overflow: hidden; }}
    .question {{ background: #2c3e50; color: #fff; padding: 16px 24px;
                 font-size: 16px; font-weight: 600; }}
    .answers {{ display: grid; grid-template-columns: 1fr 1fr; }}
    @media (max-width: 840px) {{ .answers {{ grid-template-columns: 1fr; }} }}
    .answer {{ padding: 20px 24px; }}
    .answer.base {{ border-right: 1px solid #eee; }}
    .label {{ font-weight: 700; font-size: 14px; margin-bottom: 4px; }}
    .base-label {{ color: #e67e22; }}
    .lora-label {{ color: #27ae60; }}
    .meta {{ font-size: 12px; color: #aaa; margin-bottom: 12px; }}
    .text {{ font-size: 14px; line-height: 1.8; white-space: pre-wrap; word-break: break-word; }}
</style>
</head>
<body>
    <h1>Qwen-Law 微调前后对比报告</h1>
    <div class="subtitle">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-title">总生成时间</div>
            <div class="stat-row">
                <div><div class="stat-val base-color">{total_base_time:.1f}s</div><div class="stat-label">Base</div></div>
                <div><div class="stat-val lora-color">{total_lora_time:.1f}s</div><div class="stat-label">LoRA</div></div>
            </div>
        </div>
        <div class="stat-box">
            <div class="stat-title">平均回答长度</div>
            <div class="stat-row">
                <div><div class="stat-val base-color">{avg_base_len:.0f}</div><div class="stat-label">Base (字)</div></div>
                <div><div class="stat-val lora-color">{avg_lora_len:.0f}</div><div class="stat-label">LoRA (字)</div></div>
            </div>
        </div>
        <div class="stat-box">
            <div class="stat-title">测试题数</div>
            <div class="stat-row">
                <div><div class="stat-val" style="color:#2c3e50;">{len(records)}</div><div class="stat-label">道法律问题</div></div>
            </div>
        </div>
    </div>

    {cards_html}
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] HTML report saved to {path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def run_comparison():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_model, tokenizer = load_base_model(BASE_MODEL_PATH)

    print("\n[INFO] Generating base model answers ...")
    base_results = []
    for i, q in enumerate(LEGAL_QUESTIONS, 1):
        print(f"  Base model - question {i}/{len(LEGAL_QUESTIONS)} ...")
        answer, elapsed = generate(base_model, tokenizer, q)
        base_results.append((answer, elapsed))
        print(f"    done in {elapsed:.1f}s ({len(answer)} chars)")

    lora_model = load_lora_model(base_model, LORA_ADAPTER_PATH)

    print("\n[INFO] Generating LoRA fine-tuned model answers ...")
    lora_results = []
    for i, q in enumerate(LEGAL_QUESTIONS, 1):
        print(f"  LoRA model - question {i}/{len(LEGAL_QUESTIONS)} ...")
        answer, elapsed = generate(lora_model, tokenizer, q)
        lora_results.append((answer, elapsed))
        print(f"    done in {elapsed:.1f}s ({len(answer)} chars)")

    records = []
    for i, q in enumerate(LEGAL_QUESTIONS):
        records.append({
            "id": i + 1,
            "question": q,
            "base_answer": base_results[i][0],
            "base_time": round(base_results[i][1], 2),
            "lora_answer": lora_results[i][0],
            "lora_time": round(lora_results[i][1], 2),
        })

    # ---- Print to console ----
    print("\n" + "=" * 90)
    print("  COMPARISON: Base Qwen2.5-3B  vs  Qwen-Law (LoRA Fine-tuned)")
    print("=" * 90)
    for r in records:
        print(f"\n{'─' * 90}")
        print(f"  Q{r['id']}: {r['question']}")
        print(f"{'─' * 90}")
        print(f"\n  [Base Model] (耗时 {r['base_time']}s, {len(r['base_answer'])} 字)\n")
        for line in r["base_answer"].splitlines():
            print(f"    {line}")
        print(f"\n  [Qwen-Law LoRA] (耗时 {r['lora_time']}s, {len(r['lora_answer'])} 字)\n")
        for line in r["lora_answer"].splitlines():
            print(f"    {line}")
        print()

    # ---- Save to files ----
    save_json(records, OUTPUT_DIR / "comparison_results.json")
    save_markdown(records, OUTPUT_DIR / "comparison_results.md")
    save_html(records, OUTPUT_DIR / "comparison_results.html")

    print(f"\n{'=' * 90}")
    print(f"  All results saved to {OUTPUT_DIR.resolve()}/")
    print(f"    - comparison_results.json  (structured data)")
    print(f"    - comparison_results.md    (markdown report)")
    print(f"    - comparison_results.html  (visual report, open in browser)")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    run_comparison()
