"""
ModernFinBERT Gradio Demo — Financial Sentiment Analysis.
Loads neoyipeng/ModernFinBERT-base and classifies input text as
NEGATIVE, NEUTRAL, or POSITIVE with calibrated confidence scores.
"""

import json
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "neoyipeng/ModernFinBERT-base"
LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Load calibration config (temperature scaling)
TEMPERATURE = 1.0
cal_path = Path(__file__).resolve().parent.parent / "calibration_config.json"
if cal_path.exists():
    with open(cal_path) as f:
        cal_config = json.load(f)
    TEMPERATURE = cal_config["temperature"]
    print(f"Calibration loaded: T={TEMPERATURE:.4f} "
          f"(ECE: {cal_config['ece_before']:.4f} -> {cal_config['ece_after']:.4f})")
else:
    print("No calibration_config.json found, using raw softmax (T=1.0)")

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
print(f"Model loaded on {device}.")


def predict(text: str) -> dict[str, float]:
    """Return calibrated sentiment confidences for the input text."""
    if not text or not text.strip():
        return {label: 0.0 for label in LABEL_NAMES}

    inputs = tokenizer(
        text, return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        calibrated_logits = logits / TEMPERATURE
        probs = torch.softmax(calibrated_logits, dim=-1).squeeze().cpu().numpy()

    return {label: float(round(prob, 4)) for label, prob in zip(LABEL_NAMES, probs)}


examples = [
    ["Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007."],
    ["Revenue for the quarter declined 12% year-over-year due to weakening demand."],
    ["The company maintained its market position with stable quarterly results."],
    ["Net sales increased by 61.4% to EUR 19.4 mn, and profit before taxes was EUR 4.0 mn."],
    ["The Board of Directors proposes that no dividend be paid for the financial year 2009."],
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Financial Text",
        placeholder="Enter a financial sentence or paragraph...",
        lines=4,
    ),
    outputs=gr.Label(
        label="Sentiment Prediction",
        num_top_classes=3,
    ),
    title="ModernFinBERT: Financial Sentiment Analysis",
    description=(
        "**ModernFinBERT** is a financial sentiment analysis model built on the "
        "[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) architecture, "
        "fine-tuned on aggregated financial sentiment data. It classifies text as "
        "**Negative**, **Neutral**, or **Positive**.\n\n"
        "Confidence scores are **calibrated** via temperature scaling "
        f"(T={TEMPERATURE:.2f}) for reliable uncertainty estimates.\n\n"
        "Model: [`neoyipeng/ModernFinBERT-base`](https://huggingface.co/neoyipeng/ModernFinBERT-base) "
        "| Paper: *ModernFinBERT: Modernizing Financial Sentiment Analysis with ModernBERT*"
    ),
    examples=examples,
    cache_examples=False,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
