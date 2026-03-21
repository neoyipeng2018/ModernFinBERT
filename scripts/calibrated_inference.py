"""
Calibrated inference for ModernFinBERT.

Applies post-hoc temperature scaling to produce calibrated confidence scores.
The temperature parameter is loaded from calibration_config.json.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION_PATH = ROOT / "calibration_config.json"


class CalibratedModernFinBERT:
    LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    def __init__(self, model_id="neoyipeng/ModernFinBERT-base", calibration_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = self.model.to(self.device).eval()

        cal_path = Path(calibration_path) if calibration_path else DEFAULT_CALIBRATION_PATH
        with open(cal_path) as f:
            config = json.load(f)
        self.temperature = config["temperature"]

    def predict(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            calibrated_logits = logits / self.temperature
            probs = torch.softmax(calibrated_logits, dim=-1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        return {
            "label": self.LABEL_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                name: float(round(p, 4))
                for name, p in zip(self.LABEL_NAMES, probs)
            },
            "calibrated": True,
        }

    def predict_batch(self, texts, batch_size=32):
        all_results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                calibrated_logits = logits / self.temperature
                probs = torch.softmax(calibrated_logits, dim=-1).cpu().numpy()

            for p in probs:
                pred_idx = int(np.argmax(p))
                all_results.append({
                    "label": self.LABEL_NAMES[pred_idx],
                    "confidence": float(p[pred_idx]),
                    "probabilities": {
                        name: float(round(v, 4))
                        for name, v in zip(self.LABEL_NAMES, p)
                    },
                    "calibrated": True,
                })
        return all_results


if __name__ == "__main__":
    print("Loading CalibratedModernFinBERT...")
    model = CalibratedModernFinBERT()
    print(f"Temperature: {model.temperature:.4f}")

    test_texts = [
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007.",
        "Revenue declined 5% year-over-year.",
        "The company maintained its market position with stable quarterly results.",
        "Shares dropped 8% in after-hours trading following the guidance cut.",
        "Net sales increased by 4% to EUR 1.2 billion.",
    ]

    print("\nCalibrated Predictions:")
    print("-" * 80)
    results = model.predict_batch(test_texts)
    for text, result in zip(test_texts, results):
        print(f"  {result['label']:<10} (conf={result['confidence']:.3f})  {text[:65]}...")
