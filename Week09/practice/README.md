# 다음 토큰 점수표 모델의 학습·검증 손실 기록

## 제출 마감 및 push 권한 안내

- **마감: 2026년 11월 5일(목) 오전 11:00 (KST, 한국시간)**까지 개인 저장소에 push를 완료한다. 로컬 저장이나 commit만으로는 제출되지 않는다.
- **push 차단: 11:01(KST)부터.** 11:01(KST)에 모든 학생 저장소의 push 권한을 일괄 회수하는 작업이 실행된다.
- **서술형 응시:** 서술형 문제는 **다음 실습 수업 시작 30분 후 전원이 함께** 응시한다. 마감 시점에 push된 `Week09/` 코드가 문제와 채점의 기준이다.
- **권한이 닫힌 기간:** push 권한은 마감부터 다음 서술형 응시와 평가가 끝날 때까지 닫혀 있다. 이 기간에 다음 주차 실습 코드는 로컬에 commit만 해 두고, 권한이 열리면 push한다.
- **출석 조건:** 서술형 시작 시각 전에 UCheck 출석 또는 지각 기록이 없으면 응시하더라도 0점이다.
- 문제를 처음 연 순간부터 5분 이내에 답안 제출을 완료해야 합니다. 제한시간이 지나면 문제 화면이 자동으로 사라지고 제출할 수 없습니다. 시간 내 제출하지 않으면 해당 서술형 문항은 0점 처리됩니다.

## 목표와 전달물

단편소설 “The Verdict”의 토큰열로 다음 토큰 예측 모델을 학습하고, epoch마다 학습 손실과 검증 손실을 기록한다. 이 기록으로 검증 손실이 가장 낮은 epoch(학습을 멈출 시점)를 고르고, 두 손실의 변화를 그래프로 저장한다.

전달물은 다음 두 가지다.

- `loss_tracking.py`: 데이터 준비 → 손실 계산 → 학습 루프 → 기록·판정 → 그래프 저장을 한 번에 실행하는 Python 소스 한 개
- `loss_curve.png`: epoch 0–12의 학습·검증 손실 곡선

강의에서 다룬 교차엔트로피 손실, `flatten` 후 `cross_entropy`, 학습·검증 분할, batch 평균 손실, `model.eval()`·`torch.no_grad()` 평가, `zero_grad → backward → step` 학습 루프, 과적합 판정을 하나의 실행 흐름에 넣는다.

## 제공 입력과 환경

Python, PyTorch, matplotlib를 사용한다. GPU는 필요하지 않다. 예시 구현은 Python 3.11, PyTorch 2.7.0 CPU, matplotlib 3.11에서 실행했으며 수 초 안에 끝난다.

제공 파일은 `README.md`와 `verdict_ids.json`이다. 실행할 Python 파일은 직접 작성한다. AI로 코드 초안을 작성해도 되며, 실제 출력이 아래 기대값과 맞는지는 직접 확인한다.

`verdict_ids.json`은 강의와 같은 방식으로 만든 토큰 ID 데이터다. 원문을 문자 기준 90% : 10%로 나눈 뒤 GPT-2 tokenizer로 각각 토큰화했다. 이 소설에 실제로 나오는 GPT-2 토큰 1,417종에만 0–1416의 새 번호를 붙였으므로, 어휘 수가 50,257이 아니라 1,417이다.

| 키 | 값 | 역할 |
|---|---|---|
| `vocab_size` | `1417` | 어휘 수 V. 모델 크기와 균등 추측 기준값 log V(자연로그)를 정한다 |
| `train_ids` | 길이 4,612의 정수 목록 | 학습 데이터(weight 갱신에 사용) |
| `val_ids` | 길이 534의 정수 목록 | 검증 데이터(손실만 계산, 갱신에 사용하지 않음) |
| `gpt2_ids` | 길이 1,417의 정수 목록 | 새 번호 i의 원래 GPT-2 ID. 참고용이며 사용하지 않아도 된다 |

모델은 **다음 토큰 점수표** `torch.nn.Embedding(V, V)`를 사용한다. 입력 토큰 ID의 행(길이 V)을 그 위치의 다음 토큰 logits로 그대로 쓰는 모델로, 앞 문맥은 쓰지 않고 바로 앞 토큰 하나만 보고 다음 토큰 점수를 낸다. 강의의 GPT와 입력 [B, T] → 출력 [B, T, V] 형태가 같아 강의의 손실 계산·학습 루프를 그대로 적용할 수 있다. 모든 weight를 `torch.nn.init.zeros_`로 0으로 만든 뒤 학습을 시작하므로, 학습 전 모든 후보의 확률은 1/V이고 학습 전 손실은 log V여야 한다.

| 설정 | 값 | 이유 |
|---|---|---|
| 창 길이·stride | 32, 32 | 겹치지 않는 input–target 창. target은 input을 한 칸 민 토큰열 |
| batch 크기 | 8 | 학습 창 144개 → 18 batch, 검증 창 16개 → 2 batch |
| batch 순서 | 섞지 않음(앞에서부터) | 누구나 같은 결과를 재현하기 위해 고정 |
| 학습 데이터 마지막 불완전 batch | 버림 | 144 = 8 × 18이므로 실제로 버려지는 창은 없다 |
| optimizer | `torch.optim.AdamW(model.parameters(), lr=0.2, weight_decay=0.1)` | 강의의 GPT(lr=0.0004)와 달리 점수표는 각 값이 logits에 직접 쓰여 큰 학습률로 12 epoch 안에 변화가 보인다 |
| epoch 수 | 12 | epoch는 학습 batch 18개를 한 번씩 모두 쓰는 단위(1 epoch = 18 step) |

## 처리 순서

1. `verdict_ids.json`을 읽는다.
2. `train_ids`와 `val_ids` 각각에서 시작점 `range(0, len(ids) - 32, 32)`마다 input `ids[i:i+32]`, target `ids[i+1:i+33]`을 만든다. 학습 창은 144개, 검증 창은 16개다.
3. 창을 앞에서부터 8개씩 묶어 학습 batch 18개, 검증 batch 2개를 만든다. 각 batch의 input·target shape은 [8, 32]다. `DataLoader`를 써도 되고(`shuffle=False`) 직접 잘라도 된다.
4. `torch.nn.Embedding(1417, 1417)`을 만들고 weight를 모두 0으로 바꾼다.
5. batch 손실: logits [8, 32, 1417]을 `flatten(0, 1)`로 [256, 1417], target을 [256]으로 펴서 `torch.nn.functional.cross_entropy`에 넣는다.
6. 데이터 전체 손실: 모든 batch 손실의 평균. 평가는 `model.eval()`과 `with torch.no_grad():` 안에서 하고, 끝나면 `model.train()`으로 되돌린다.
7. 학습 전(epoch 0)에 두 손실을 기록한다.
8. epoch 1–12마다 학습 batch 18개에 대해 `optimizer.zero_grad()` → 손실 → `loss.backward()` → `optimizer.step()`을 실행하고, epoch이 끝날 때 6번 방식으로 두 손실을 기록한다.
9. 검증 손실이 가장 낮은 epoch를 기록에서 찾아 출력하고, 마지막 epoch의 검증 손실 − 학습 손실을 출력한다.
10. 가로축 epoch 0–12, 세로축 손실로 두 곡선(학습·검증)을 그려 `loss_curve.png`로 저장한다. 강의는 손실 곡선을 읽는 방법을 다뤘고 그래프를 그리는 코드는 다루지 않았으므로 판정 기준만 제시한다: 가로축에 epoch 0–12의 13개 점, 세로축에 손실, 학습·검증 두 곡선을 구분하는 범례, 실행할 때마다 같은 이름으로 새로 저장. 그래프 코드는 AI와 함께 작성해도 된다.

## 실행과 확인

`Week09/` 폴더에서 다음 명령을 실행한다.

```bash
python loss_tracking.py
```

명령은 같은 폴더의 `verdict_ids.json`을 읽고, 터미널에 기록을 출력하며, `loss_curve.png`를 새로 만들거나 덮어쓴다. 제공 파일 `verdict_ids.json`은 수정하지 않는다.

다음 값이 출력에 포함되어야 한다. 출력 문구와 배치는 자유이며, 손실은 소수 넷째 자리까지 반올림해 표시한다(계산 중에는 반올림하지 않는다).

```text
train_batches: 18, val_batches: 2
log(vocab_size): 7.2563
epoch 0: train 7.2563, val 7.2563
epoch 1: train 5.3918, val 6.4323
epoch 6: train 3.0650, val 5.8678
epoch 12: train 2.8844, val 5.8860
best_epoch: 6, best_val_loss: 5.8678
final_gap: 3.0016
```

- epoch 0의 두 손실이 log 1417 = 7.2563과 같으면 점수표가 모든 후보에 1/1417을 주는 균등 상태에서 출발한 것이다.
- 학습 손실은 12 epoch 동안 계속 줄지만, 검증 손실은 epoch 6의 5.8678 이후 조금씩 커진다. 이 기록으로 epoch 6을 학습을 멈출 시점으로 고른다.
- 위에 적지 않은 epoch의 값도 함께 출력한다. 값이 다르면 창·batch 순서, weight 0 초기화, 평가 상태, 학습률 설정을 먼저 확인한다.

## 실습결과 제출 양식

학생 저장소의 `Week09/`에 다음 파일을 둔다.

| 경로 | 내용 |
|---|---|
| `Week09/loss_tracking.py` | 작성한 Python 소스. 빈 줄과 주석을 포함해 100줄 이내 |
| `Week09/verdict_ids.json` | 제공 파일을 그대로 포함 |
| `Week09/loss_curve.png` | `loss_tracking.py` 실행으로 저장한 손실 곡선 |

손실 값은 `loss_tracking.py`를 실행했을 때 터미널에 출력되도록 작성한다. 결과 숫자를 직접 적어 출력하지 말고 데이터에서 실제로 계산한다. 별도 결과 보고서·스크린샷·고찰문은 제출하지 않는다.

1. `Week09/` 폴더에서 위 명령을 실행하고 출력이 기대값과 같은지, `loss_curve.png`가 새로 저장되었는지 확인한다.
2. 세 파일을 같은 commit으로 개인 저장소에 push한다.
3. GitHub 웹에서 `Week09/loss_tracking.py`와 `Week09/loss_curve.png`를 열어 최신 내용이 올라갔는지 확인한다.

개인 이해도는 다음 수업 시작 때 교수자 감독 아래 Discord 비공개 서술형 문항 한 개로 별도 평가한다. 문항은 마감 시점에 push된 코드를 기준으로 답한다.

## 평가기준

1. 제출한 Python 소스가 `.py` 파일 한 개이고 100줄 이내인가?
2. 0으로 초기화한 `Embedding(1417, 1417)`에 대해 `flatten` 후 `cross_entropy`로 batch 손실을, batch 평균으로 데이터 손실을 계산하고, `model.eval()`·`torch.no_grad()` 평가와 `zero_grad → backward → step` 학습 루프를 실제로 실행해 epoch 0의 `7.2563 / 7.2563`과 epoch 12의 `2.8844 / 5.8860`을 출력하는가?
3. 기록한 검증 손실에서 최저 epoch `6`(5.8678)을 계산해 출력하고, epoch 0–12의 학습·검증 손실 두 곡선을 담은 `loss_curve.png`를 실행으로 생성해 제출했는가?
