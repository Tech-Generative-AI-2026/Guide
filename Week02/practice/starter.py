"""Week02 starter: NumPy만 사용해 선형회귀 경사하강법을 구현한다."""

from pathlib import Path

import numpy as np


DATA_PATH = Path(__file__).parent / "data" / "linear_train.csv"


def predict(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """TODO: 예측식 y_hat = w * x + b를 반환한다."""
    raise NotImplementedError


def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    """TODO: 평균제곱오차 mean((y_hat - y) ** 2)를 반환한다."""
    raise NotImplementedError


def gradients(x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple[float, float]:
    """TODO: MSE의 dL/dw, dL/db를 반환한다."""
    raise NotImplementedError


def train(x: np.ndarray, y: np.ndarray, learning_rate: float, steps: int) -> list[dict]:
    """TODO: w=b=0에서 시작해 매 반복의 step, w, b, loss를 기록한다."""
    raise NotImplementedError


if __name__ == "__main__":
    # data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)
    # x, y = data[:, 0], data[:, 1]
    # 아래 README의 고정 검증값을 assert로 확인한 뒤 결과를 출력한다.
    pass
