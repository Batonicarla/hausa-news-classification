import json, os, re
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TAIL_FRAC = 0.25


def clean_text(t):
    """Identical to the cleaning used when the dataset was prepared."""
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class HausaNewsClassifier:
    """Loads the fine-tuned model (Hub id or local folder) and classifies Hausa news text."""

    def __init__(self, model_path):
        if os.path.isdir(model_path):
            meta_file = os.path.join(model_path, "meta.json")
        else:
            meta_file = hf_hub_download(model_path, "meta.json")
        self.meta = json.load(open(meta_file))
        self.max_len = int(self.meta["max_len"])
        self.strategy = self.meta["strategy"]          # "head" or "head+tail", must match training
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, low_cpu_mem_usage=True).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        t = self.tok
        self.start = t.cls_token_id if t.cls_token_id is not None else t.bos_token_id
        self.end = t.sep_token_id if t.sep_token_id is not None else t.eos_token_id

    def encode(self, text):
        ids = self.tok(clean_text(text), add_special_tokens=False, truncation=False)["input_ids"]
        n_total, budget = len(ids), self.max_len - 2
        if n_total > budget:
            if self.strategy == "head":
                ids = ids[:budget]
            else:
                tail = int(budget * TAIL_FRAC)
                ids = ids[:budget - tail] + ids[-tail:]
        return [self.start] + ids + [self.end], n_total, budget

    @torch.inference_mode()
    def predict_probs(self, text):
        ids, n_total, budget = self.encode(text)
        out = self.model(input_ids=torch.tensor([ids]),
                         attention_mask=torch.ones(1, len(ids), dtype=torch.long))
        return torch.softmax(out.logits[0].float(), -1).numpy(), n_total, budget
