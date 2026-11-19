"""baseline 학습 (재학습 평가의 학습 진입점).

--data 아래의 모든 levelN/train.txt를 next-token 예측으로 학습하고 --out에 checkpoint.pt와 config.json을 저장한다.

로컬 실행 (FinalProject 폴더에서):
  python submission/train.py              # data/ 를 읽고 submission/checkpoint.pt를 저장 (CPU 몇 분 이내)
  python train.py                         # 위와 같다
교수자 재학습 평가가 실행하는 방식:
  python /sub/train.py --data /data --out /out --seed 0
  /sub에는 submission/의 .py·.json·.md만 있고(checkpoint.pt 없음, 읽기 전용), /data에는 공개된 단계의
  levelN/train.txt·val.txt만 있으며, 네트워크는 없고 GPU 1개를 정해진 시간 안에서 쓴다.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

SUB = Path(__file__).resolve().parent
sys.path.insert(0, str(SUB))

from model import build_model  # noqa: E402
from tokenizer import encode  # noqa: E402

STEPS = 3000
BATCH = 32
LR = 3e-3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(SUB.parent / "data"), help="levelN/train.txt·val.txt가 있는 폴더")
    p.add_argument("--out", default=str(SUB), help="checkpoint.pt와 config.json을 쓸 폴더")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = json.loads((SUB / "config.json").read_text(encoding="utf-8"))
    model = build_model(config).to(device)
    # 채점과 같은 방식: 학습 여부와 무관하게 모든 parameter와 buffer를 센다
    n_params = sum(t.numel() for t in {id(t): t for t in [*model.parameters(), *model.buffers()]}.values())
    print(f"parameters: {n_params}, device: {device}")

    lines = []
    for path in sorted(Path(args.data).glob("level*/train.txt")):
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
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 500 == 0 or step == STEPS - 1:
            print(f"step {step} loss {loss.item():.4f}")

    out.mkdir(parents=True, exist_ok=True)
    # 모델째 CPU로 옮긴 뒤 저장한다. tensor마다 .cpu()로 옮기면 weight tying으로 공유한 tensor가 따로 저장돼 두 번 세어진다
    torch.save(model.cpu().state_dict(), out / "checkpoint.pt")
    config["n_params"] = n_params
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
