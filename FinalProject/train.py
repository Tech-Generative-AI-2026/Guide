"""baseline 학습. 공개된 모든 단계의 data/level*/train.txt를 next-token 예측으로 학습하고 submission/checkpoint.pt를 저장한다.

실행: python train.py   (CPU 몇 분 이내)
"""
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
sys.path.insert(0, str(SUB))

from model import build_model  # noqa: E402
from tokenizer import encode  # noqa: E402

STEPS = 3000
BATCH = 32
LR = 3e-3
SEED = 0


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    config = json.loads((SUB / "config.json").read_text(encoding="utf-8"))
    model = build_model(config)
    # 채점과 같은 방식: 학습 여부와 무관하게 모든 parameter와 buffer를 센다
    n_params = sum(t.numel() for t in {id(t): t for t in [*model.parameters(), *model.buffers()]}.values())
    print(f"parameters: {n_params}")

    lines = []
    for path in sorted((ROOT / "data").glob("level*/train.txt")):
        lines += path.read_text(encoding="utf-8").splitlines()
    seqs = [encode(line) for line in lines]
    max_len = max(len(s) for s in seqs)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    model.train()
    for step in range(STEPS):
        batch = [random.choice(seqs) for _ in range(BATCH)]
        x = torch.zeros(BATCH, max_len - 1, dtype=torch.long)
        y = torch.full((BATCH, max_len - 1), -100, dtype=torch.long)
        for i, s in enumerate(batch):
            x[i, : len(s) - 1] = torch.tensor(s[:-1])
            y[i, : len(s) - 1] = torch.tensor(s[1:])
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 500 == 0 or step == STEPS - 1:
            print(f"step {step} loss {loss.item():.4f}")

    torch.save(model.state_dict(), SUB / "checkpoint.pt")
    config["n_params"] = n_params
    (SUB / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved submission/checkpoint.pt")


if __name__ == "__main__":
    main()
