"""Gradio app: route a support message to a team and compare all four models.

python app/app.py  →  http://127.0.0.1:7860
"""
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router.pipeline import route  # noqa: E402
from router.registry import DEFAULT_MODEL, load_available  # noqa: E402

MODELS, MISSING = load_available()
if not MODELS:
    raise SystemExit("No trained models found. Run: python scripts/download_models.py")
for _m in MODELS.values():  # warm-up so the first live message is not slow
    route("merhaba bulaşık makinem çalışmıyor", _m)

EXAMPLES = [
    "bulaşık makinem su boşaltmıyor, model DW-6534",
    "my fridge stopped cooling, model RF 4410",
    "garanti belgemi kaybettim, cihaz yine de garantide mi?",
    "can I get a new rack for my dishwasher?",
    "what is the balance on my checking account",
    "Merhaba!",
    "I feel tired of my washer",
    "who won the football match last night",
]

TEAM_COLORS = {"Human agent": "#B54708", "Auto reply": "#475467"}


def decision_html(d, model_name: str) -> str:
    color = TEAM_COLORS.get(d.team, "#1D4ED8")
    conf = "—" if d.confidence is None else f"{d.confidence:.1%}"
    codes = ", ".join(d.model_codes) if d.model_codes else "none"
    reply = f"<p style='margin:8px 0 0'><b>Reply:</b> {d.reply}</p>" if d.reply else ""
    return f"""
<div style="border:1px solid #D0D5DD;border-radius:10px;padding:14px 16px">
  <div style="font-size:13px;color:#667085">Routed to</div>
  <div style="font-size:26px;font-weight:700;color:{color}">{d.team}</div>
  <div style="margin-top:6px">Intent <b>{d.intent}</b> · confidence <b>{conf}</b> · decided by {d.source}</div>
  <div style="margin-top:4px;color:#475467">Model codes: <b>{codes}</b> · normalized: <code>{d.normalized}</code></div>
  {reply}
  <div style="margin-top:8px;font-size:12px;color:#667085">{model_name} · {d.latency_ms:.1f} ms</div>
</div>"""


def on_route(message: str, model_name: str):
    if not message.strip():
        return "", gr.update()
    decision = route(message, MODELS[model_name])
    return decision_html(decision, model_name), compare(message)


def compare(message: str) -> list[list]:
    rows = []
    for name, model in MODELS.items():
        d = route(message, model)
        conf = "" if d.confidence is None else round(d.confidence, 3)
        rows.append([name, d.intent, d.team, conf, round(d.latency_ms, 2)])
    return rows


with gr.Blocks(title="Support Ticket Router") as demo:
    gr.Markdown("# Customer Support Ticket Router\nTurkish and English support messages → the right team. "
                "Low-confidence and out-of-scope messages go to a human agent; greetings get a rule-based reply.")
    with gr.Row():
        with gr.Column(scale=3):
            msg = gr.Textbox(label="Customer message", placeholder="e.g. bulaşık makinem su boşaltmıyor, model DW-6534",
                             lines=2)
            with gr.Row():
                model_dd = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL if DEFAULT_MODEL in MODELS else list(MODELS)[0],
                                       label="Model")
                btn = gr.Button("Route", variant="primary")
            gr.Examples(EXAMPLES, inputs=msg)
        with gr.Column(scale=3):
            out = gr.HTML()
    gr.Markdown("### All models side by side")
    table = gr.Dataframe(headers=["Model", "Intent", "Team", "Confidence", "Latency (ms)"],
                         interactive=False, wrap=True)
    if MISSING:
        gr.Markdown("Not loaded: " + "; ".join(f"{k} ({v})" for k, v in MISSING.items()))
    btn.click(on_route, [msg, model_dd], [out, table])
    msg.submit(on_route, [msg, model_dd], [out, table])


if __name__ == "__main__":
    demo.launch()
