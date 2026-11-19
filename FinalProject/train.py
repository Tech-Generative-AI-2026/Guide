"""baseline 학습 실행 바로가기. 실제 학습 코드는 submission/train.py에 있다.

실행: python train.py   (= python submission/train.py, CPU 몇 분 이내)
재학습 평가는 submission/train.py를 실행하므로, 학습 방법을 바꿀 때는 submission/train.py를 고친다.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "submission" / "train.py"
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
