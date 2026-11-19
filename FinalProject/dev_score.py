"""공개된 단계별 val로 개발 점수를 확인한다. 암호문 답의 완전 일치만 본다.

이 점수는 최종 채점과 다르다. 최종 채점은 해독한 평문의 의미 필드를 비교한다.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
sys.path.insert(0, str(SUB))

from model import build_model  # noqa: E402
from tokenizer import EOS_ID, decode, encode  # noqa: E402


def greedy(model, prompt_ids, block_size, max_new=64):
    idx = torch.tensor([prompt_ids])
    out = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(idx[:, -block_size:])[:, -1]
            nxt = int(logits.argmax(-1))
            if nxt == EOS_ID:
                break
            out.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
    return out


def main():
    config = json.loads((SUB / "config.json").read_text(encoding="utf-8"))
    model = build_model(config)
    model.load_state_dict(torch.load(SUB / "checkpoint.pt", weights_only=True, map_location="cpu"))
    model.eval()

    for path in sorted((ROOT / "data").glob("level*/val.txt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        correct = 0
        for line in lines:
            prompt, gold = line.split("<SEP>")
            gold = gold[: -len("<EOS>")]
            pred = decode(greedy(model, encode(prompt + "<SEP>"), config["block_size"]))
            correct += pred == gold
        print(f"{path.parent.name} dev exact match: {correct}/{len(lines)} = {100 * correct / len(lines):.1f}%")


if __name__ == "__main__":
    main()
