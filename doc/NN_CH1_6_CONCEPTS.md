# 신경망 1~6장 개념 흐름 정리

이 문서는 `밑바닥부터 시작하는 딥러닝` 1~6장과 현재 MNIST 과제 구현 흐름을 한 번에 연결해서 보기 위한 요약본입니다.

참고 코드:

- 교재 코드: `C:\jungle5\mnist`
- 과제 코드: `C:\jungle5\week13\src`

## 0. 전체 그림

신경망 학습은 한 문장으로 말하면 다음과 같습니다.

```text
입력 데이터를 넣고
-> 예측값을 만들고
-> 정답과 비교해서 loss를 구하고
-> loss를 줄이는 방향으로 W, b를 조금씩 고치는 과정
```

과제 코드에서는 이 흐름이 아래처럼 나뉩니다.

```text
data.py
-> activations.py
-> layers.py
-> losses.py
-> optimizers.py
-> network.py
-> training.py
```

가장 중요한 흐름은 이것입니다.

```text
Forward
-> Loss
-> Backward
-> Update
```

**중요:** 신경망은 "정답을 외우는 코드"가 아니라, `W`와 `b`를 조정해서 loss가 작아지도록 만드는 함수입니다.

## 1. Python과 NumPy

교재 1장은 신경망 구현을 위한 Python과 NumPy 준비 단계입니다.

NumPy 배열은 신경망에서 데이터를 담는 기본 도구입니다.

```python
x = np.array([1.0, 2.0, 3.0])
```

신경망에서 중요한 NumPy 개념은 다음과 같습니다.

| 개념 | 뜻 | 과제에서 쓰이는 곳 |
| --- | --- | --- |
| `ndim` | 배열이 몇 차원인지 | 1개 샘플인지 batch인지 구분 |
| `shape` | 각 차원의 크기 | 행렬곱 shape 확인 |
| `reshape` | 배열 모양 변경 | `(784,)`를 `(1, 784)`로 변경 |
| `np.dot`, `@` | 행렬곱 | Affine 계층 |
| `axis=0` | 같은 열끼리 계산 | `db`, BatchNorm 평균 |
| `axis=1` | 같은 행끼리 계산 | Softmax에서 샘플별 확률합 |
| `keepdims=True` | 차원 유지 | broadcasting 오류 방지 |
| `np.clip` | 값 범위 제한 | `log(0)` 방지 |

**중요:** 신경망 구현에서 에러의 대부분은 수학보다 `shape` 문제에서 납니다.

예를 들어 Affine 계층은 다음 shape가 맞아야 합니다.

```text
x: (batch_size, input_dim)
W: (input_dim, output_dim)
b: (output_dim,)

x @ W + b
= (batch_size, output_dim)
```

## 2. 퍼셉트론

퍼셉트론은 신경망의 가장 단순한 형태입니다.

```text
입력 x
-> 가중치 W를 곱함
-> 편향 b를 더함
-> 기준을 넘으면 1, 아니면 0
```

수식으로는 다음과 같습니다.

```text
y = 1 if x1*w1 + x2*w2 + b > 0
y = 0 otherwise
```

여기서 각각의 의미는 다음과 같습니다.

| 이름 | 의미 |
| --- | --- |
| `x` | 입력 신호 |
| `W` | 입력의 중요도 |
| `b` | 얼마나 쉽게 켜질지 정하는 기준 |
| `y` | 출력 |

퍼셉트론은 AND, OR 같은 단순한 논리 게이트를 만들 수 있습니다.

하지만 XOR처럼 직선 하나로 나눌 수 없는 문제는 단층 퍼셉트론으로 풀기 어렵습니다.

그래서 여러 층을 쌓은 다층 퍼셉트론, 즉 신경망이 필요합니다.

**중요:** 신경망은 퍼셉트론을 여러 층 쌓고, 중간에 활성화 함수를 넣은 구조입니다.

## 3. 신경망의 기본 구조

신경망은 보통 다음 구조를 가집니다.

```text
입력층
-> 은닉층
-> 출력층
```

MNIST 과제에서는 입력 이미지가 `28 x 28` 픽셀입니다.

```text
28 x 28 = 784
```

그래서 입력 하나는 `(784,)` 벡터로 들어갑니다.

예시 구조는 다음과 같습니다.

```text
784
-> Affine(512)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(256)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(10)
-> Softmax
```

출력이 10인 이유는 숫자 클래스가 `0~9`까지 10개이기 때문입니다.

## 4. Affine 계층

Affine 계층은 실제로 `W`를 곱하고 `b`를 더하는 계층입니다.

```text
a = xW + b
```

과제 코드에서는 `layers.py`의 `Affine`이 이 역할을 합니다.

Forward:

```python
out = x @ W + b
```

Backward:

```python
dx = dout @ W.T
dW = x.T @ dout
db = np.sum(dout, axis=0)
```

각 gradient의 뜻은 다음과 같습니다.

| 값 | 뜻 | 어디로 가나 |
| --- | --- | --- |
| `dx` | 입력 `x`에 대한 gradient | 앞 레이어로 return |
| `dW` | 가중치 `W`에 대한 gradient | optimizer가 사용 |
| `db` | 편향 `b`에 대한 gradient | optimizer가 사용 |

**중요:** `return`은 `dx`만 합니다. `dW`, `db`는 앞 레이어로 넘기는 값이 아니라 해당 Affine 계층의 파라미터를 업데이트하기 위한 값입니다.

## 5. 활성화 함수

Affine만 여러 번 쌓으면 결국 하나의 큰 선형 변환과 비슷해집니다.

그래서 중간에 비선형 함수인 활성화 함수가 필요합니다.

```text
Affine
-> Activation
-> Affine
-> Activation
```

활성화 함수는 신경망이 복잡한 패턴을 배울 수 있게 해줍니다.

### 5.1 Sigmoid

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

특징:

- 출력 범위가 `0~1`
- 예전 신경망에서 많이 사용
- 값이 너무 크거나 작으면 gradient가 거의 0이 됨

단점:

- vanishing gradient 문제가 생기기 쉽습니다.
- 출력 중심이 0이 아니라 학습이 느려질 수 있습니다.

### 5.2 ReLU

```text
ReLU(x) = max(0, x)
```

특징:

- 양수는 그대로 통과
- 0 이하 값은 0
- 계산이 빠름
- gradient가 비교적 잘 흐름

과제의 `ReLU.forward`:

```python
self.mask = (x <= 0)
out = x.copy()
out[self.mask] = 0
return out
```

과제의 `ReLU.backward`:

```python
dout[self.mask] = 0
return dout
```

**중요:** ReLU는 forward 때 막힌 위치를 backward 때도 막습니다.

### 5.3 Softmax

Softmax는 출력층에서 점수를 확률처럼 바꿉니다.

```text
점수 [2.0, 1.0, 0.1]
-> 확률 [0.66, 0.24, 0.10]
```

수치 안정성을 위해 최댓값을 뺍니다.

```python
x = x - np.max(x, axis=1, keepdims=True)
exp_x = np.exp(x)
out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

**중요:** Softmax 출력은 각 샘플마다 합이 1입니다.

### 5.4 활성화 함수 비교

| 함수 | 주 사용 위치 | 장점 | 단점 |
| --- | --- | --- | --- |
| Step | 퍼셉트론 | 단순함 | 미분이 어려움 |
| Sigmoid | 예전 은닉층, 이진 확률 | 0~1 출력 | gradient가 사라지기 쉬움 |
| ReLU | 은닉층 | 빠르고 학습 잘 됨 | 음수 영역 gradient가 0 |
| Softmax | 다중분류 출력층 | 클래스 확률로 해석 가능 | 단독보다 Cross Entropy와 함께 사용 |

MNIST 과제에서는 은닉층에 ReLU, 출력층에 Softmax를 씁니다.

## 6. 손실 함수

손실 함수는 모델이 얼마나 틀렸는지를 숫자로 나타냅니다.

```text
예측이 정답과 가까움 -> loss 작음
예측이 정답과 멂 -> loss 큼
```

MNIST 같은 분류 문제에서는 Cross Entropy Loss를 씁니다.

```python
loss = -np.sum(np.log(correct_probs)) / batch_size
```

과제 구현 흐름:

```python
y_pred = np.clip(y_pred, 1e-7, 1.0)
batch_size = y_pred.shape[0]
correct_probs = y_pred[np.arange(batch_size), y_true]
loss = -np.sum(np.log(correct_probs)) / batch_size
```

왜 `np.clip`이 필요한가?

```text
log(0)은 계산할 수 없기 때문
```

그래서 확률의 최솟값을 `1e-7` 같은 작은 양수로 제한합니다.

**중요:** Cross Entropy는 정답 클래스의 확률만 봅니다. 정답 확률이 높으면 loss가 작고, 낮으면 loss가 큽니다.

## 7. 미분과 gradient

학습하려면 `W`, `b`를 어떻게 고쳐야 하는지 알아야 합니다.

이때 필요한 것이 미분입니다.

```text
dL/dW: W를 바꾸면 loss가 어떻게 변하는가
dL/db: b를 바꾸면 loss가 어떻게 변하는가
```

gradient의 의미:

```text
양수 -> 이 값을 키우면 loss가 커지는 방향
음수 -> 이 값을 키우면 loss가 작아지는 방향
크기 큼 -> loss에 영향이 큼
크기 작음 -> loss에 영향이 작음
```

그래서 업데이트는 보통 gradient 반대 방향으로 합니다.

```python
W = W - learning_rate * dW
b = b - learning_rate * db
```

**중요:** 학습은 loss를 줄이는 방향으로 파라미터를 조금씩 움직이는 과정입니다.

## 8. 계산그래프와 역전파

계산그래프는 복잡한 계산을 작은 노드들로 나눠 표현한 것입니다.

예:

```text
x, W, b
-> Affine
-> ReLU
-> Affine
-> Softmax
-> Loss
```

Forward는 왼쪽에서 오른쪽으로 계산합니다.

```text
입력 -> 예측 -> loss
```

Backward는 오른쪽에서 왼쪽으로 gradient를 전달합니다.

```text
loss -> 출력층 -> 은닉층 -> 입력 방향
```

역전파의 핵심은 연쇄법칙입니다.

```text
합성함수의 미분을 뒤에서 앞으로 곱해가며 계산
```

과제에서 backward가 필요한 이유:

- `ReLU.backward`: 막힌 곳 gradient 제거
- `Affine.backward`: `dx`, `dW`, `db` 계산
- `BatchNorm.backward`: `dx`, `dgamma`, `dbeta` 계산
- `Dropout.backward`: 꺼진 뉴런 gradient 제거
- `NeuralNetwork.backward`: 모든 layer를 역순으로 연결

**중요:** 역전파는 `W`, `b`가 loss에 얼마나 영향을 줬는지 빠르게 구하기 위한 방법입니다.

## 9. Optimizer

Optimizer는 gradient를 이용해 파라미터를 실제로 업데이트합니다.

### 9.1 SGD

```python
params[key] -= lr * grads[key]
```

장점:

- 단순하고 이해하기 쉬움

단점:

- 지그재그로 움직일 수 있음
- 학습이 느릴 수 있음

### 9.2 Momentum

이전 이동 방향을 기억해서 관성처럼 움직입니다.

장점:

- SGD보다 덜 흔들림
- 골짜기 방향으로 더 빨리 움직일 수 있음

단점:

- momentum 값을 추가로 정해야 함

### 9.3 AdaGrad

자주 많이 움직인 파라미터는 학습률을 줄입니다.

장점:

- 파라미터별로 학습률 조정

단점:

- 학습률이 계속 작아져서 나중에 거의 멈출 수 있음

### 9.4 Adam

Momentum과 AdaGrad/RMSProp 계열 아이디어를 섞은 optimizer입니다.

```text
m: gradient의 이동평균
v: gradient 제곱의 이동평균
```

과제에서는 Adam이 가장 추천됩니다.

교재 참고 코드:

```text
C:\jungle5\mnist\common\optimizer.py
```

### 9.5 Optimizer 비교

| Optimizer | 핵심 아이디어 | 장점 | 단점 |
| --- | --- | --- | --- |
| SGD | gradient 반대 방향으로 이동 | 단순함 | 느리고 지그재그 가능 |
| Momentum | 이전 속도 반영 | 더 부드럽고 빠름 | 추가 하이퍼파라미터 필요 |
| AdaGrad | 자주 움직인 방향 학습률 감소 | 파라미터별 조정 | 학습률이 너무 작아질 수 있음 |
| Adam | m, v 이동평균 사용 | 빠르고 안정적 | 내부 상태가 조금 복잡 |

**중요:** 처음 과제 정확도를 올릴 때는 Adam부터 쓰는 것이 실용적입니다.

## 10. 가중치 초기화

가중치 초기화는 학습 시작 전 `W`를 어떤 값으로 시작할지 정하는 것입니다.

나쁜 초기화:

```text
W를 전부 0으로 초기화
```

문제:

- 모든 뉴런이 같은 계산을 하게 됨
- 서로 다른 특징을 배우지 못함

정규분포 초기화:

```python
W = np.random.randn(input_dim, output_dim) * scale
```

scale을 어떻게 잡느냐가 중요합니다.

### 10.1 Xavier 초기화

Sigmoid나 Tanh에 어울립니다.

```text
scale = sqrt(1 / input_dim)
```

### 10.2 He 초기화

ReLU에 어울립니다.

```text
scale = sqrt(2 / input_dim)
```

### 10.3 초기화 비교

| 초기화 | 어울리는 활성화 함수 | 특징 |
| --- | --- | --- |
| std=0.01 | 실험용 | 값이 너무 작아 학습이 느릴 수 있음 |
| Xavier | Sigmoid, Tanh | 분산을 적당히 유지 |
| He | ReLU | ReLU에서 죽는 절반을 고려 |

**중요:** ReLU를 쓰면 He 초기화를 우선 고려합니다.

## 11. Batch Normalization

BatchNorm은 학습 중 각 배치의 중간값을 평균 0, 분산 1에 가깝게 표준화합니다.

보통 흐름:

```text
Affine
-> BatchNorm
-> ReLU
```

Affine 출력:

```text
a = xW + b
```

BatchNorm:

```text
x_hat = (a - mean) / sqrt(var + eps)
out = gamma * x_hat + beta
```

여기서 `x_hat`은 원본 이미지 입력 `x`가 아니라, BatchNorm에 들어온 값을 표준화한 중간 결과입니다.

Batch가 100개이고 Affine 출력이 `(100, 512)`라면:

```text
1번 뉴런 출력값 100개를 모아 평균/분산 계산
2번 뉴런 출력값 100개를 모아 평균/분산 계산
...
512번 뉴런 출력값 100개를 모아 평균/분산 계산
```

BatchNorm의 파라미터:

| 값 | 의미 |
| --- | --- |
| `gamma` | 표준화된 값을 다시 scale |
| `beta` | 표준화된 값에 shift 추가 |
| `running_mean` | 추론 때 쓸 평균 |
| `running_var` | 추론 때 쓸 분산 |

학습 모드:

```text
현재 batch의 mean, var 사용
running_mean, running_var 갱신
```

추론 모드:

```text
학습 중 저장한 running_mean, running_var 사용
```

**중요:** BatchNorm은 초깃값을 정하려고 미리 학습하는 것이 아니라, 학습 중 forward 과정에서 매 배치마다 중간값을 표준화하는 계층입니다.

## 12. 오버피팅과 정규화

오버피팅은 훈련 데이터에는 잘 맞지만 새 데이터에는 약한 상태입니다.

```text
train accuracy 높음
test accuracy 낮음
```

원인:

- 모델이 너무 큼
- 데이터가 적음
- 너무 오래 학습

대응 방법:

- Dropout
- Weight decay
- 데이터 증가
- 적절한 epoch
- 검증 데이터로 하이퍼파라미터 선택

## 13. Dropout

Dropout은 학습 중 일부 뉴런을 무작위로 꺼서 과적합을 줄입니다.

학습 모드:

```python
self.mask = np.random.rand(*x.shape) > drop_ratio
out = x * self.mask
```

추론 모드:

```python
out = x * (1 - drop_ratio)
```

Backward:

```python
dx = dout * self.mask
```

꺼진 뉴런은 backward 때도 gradient가 흐르지 않습니다.

Dropout 비율:

| 값 | 의미 |
| --- | --- |
| `0.2` | 20% 끔, 비교적 약한 규제 |
| `0.3` | 30% 끔, 과제 권장 |
| `0.5` | 50% 끔, 강한 규제 |

**중요:** Dropout은 학습 때만 랜덤으로 끄고, 추론 때는 랜덤성을 사용하지 않습니다.

## 14. 학습과 추론

학습:

```text
정답을 보고
loss를 계산하고
backward로 gradient를 구하고
W, b를 업데이트
```

추론:

```text
이미 학습된 W, b로
forward만 해서 예측
```

차이:

| 구분 | 학습 | 추론 |
| --- | --- | --- |
| 정답 사용 | 사용 | 보통 사용하지 않음 |
| backward | 함 | 안 함 |
| update | 함 | 안 함 |
| Dropout | 랜덤 mask 사용 | scale만 적용 |
| BatchNorm | batch 통계 사용 | running 통계 사용 |

**중요:** `train=True`와 `train=False`는 BatchNorm과 Dropout 때문에 반드시 구분해야 합니다.

## 15. MNIST 과제 구현 흐름

현재 과제의 핵심 함수 흐름:

```text
load_mnist
-> model.forward
-> cross_entropy_loss
-> dout 계산
-> model.backward
-> optimizer.update
-> evaluate
```

학습 루프의 한 배치:

```python
y_pred = model.forward(x_batch, train=True)
loss = cross_entropy_loss(y_pred, y_batch)

dout = y_pred.copy()
dout[np.arange(batch_size), y_batch] -= 1
dout /= batch_size

model.backward(dout)
optimizer.update(model.params, model.grads)
```

**중요:** Softmax와 Cross Entropy를 합친 gradient는 `y_pred`에서 정답 위치에 1을 빼고 batch size로 나눈 형태입니다.

## 16. 과제 파일별 역할

| 파일 | 역할 |
| --- | --- |
| `data.py` | MNIST 다운로드/로드 |
| `activations.py` | ReLU, Softmax |
| `layers.py` | Affine, BatchNorm, Dropout |
| `losses.py` | Cross Entropy Loss |
| `optimizers.py` | SGD, Adam |
| `network.py` | 여러 layer를 묶은 NeuralNetwork |
| `training.py` | train, evaluate, plot |

현재 구현 순서 추천:

```text
1. ReLU
2. Softmax
3. Affine
4. Cross Entropy
5. SGD
6. Adam
7. BatchNorm
8. Dropout
9. NeuralNetwork
10. train
11. 전체 테스트
12. 정확도 튜닝
```

## 17. 중요한 개념 연결 지도

`shape`를 알아야 Affine을 이해할 수 있습니다.

Affine을 알아야 `W`, `b`가 어디서 쓰이는지 알 수 있습니다.

`W`, `b`를 알아야 loss를 줄이기 위해 무엇을 바꿔야 하는지 알 수 있습니다.

loss를 알아야 미분의 목적을 알 수 있습니다.

미분을 알아야 backward를 이해할 수 있습니다.

backward를 알아야 optimizer가 무슨 값을 쓰는지 알 수 있습니다.

optimizer를 알아야 학습이 실제로 어떻게 진행되는지 알 수 있습니다.

초기화를 알아야 학습이 처음부터 망가지지 않게 할 수 있습니다.

BatchNorm을 알아야 중간값의 분포를 안정화하는 이유를 알 수 있습니다.

Dropout을 알아야 과적합을 줄이는 이유를 알 수 있습니다.

최종적으로 이 모든 것이 합쳐져 MNIST 정확도가 올라갑니다.

## 18. 방법별 비교 정리

### 활성화 함수

| 방법 | 쓰는 곳 | 추천 상황 |
| --- | --- | --- |
| Sigmoid | 예전 은닉층, 이진 확률 | 개념 학습용 |
| ReLU | 은닉층 | MNIST 과제 기본 추천 |
| Softmax | 출력층 | 다중분류 확률 |

### 손실 함수

| 방법 | 쓰는 곳 | 특징 |
| --- | --- | --- |
| Mean Squared Error | 회귀, 예전 예제 | 분류에서는 덜 적합 |
| Cross Entropy | 분류 | 정답 확률을 직접 벌점화 |

### Optimizer

| 방법 | 추천도 | 설명 |
| --- | --- | --- |
| SGD | 기본 이해용 | 가장 단순 |
| Momentum | 중간 | 관성 추가 |
| AdaGrad | 실험용 | 파라미터별 학습률 |
| Adam | 과제 추천 | 빠르고 안정적 |

### 초기화

| 방법 | 추천 상황 |
| --- | --- |
| 작은 정규분포 `0.01` | 실험 비교용 |
| Xavier | Sigmoid/Tanh |
| He | ReLU |

### 정규화/규제

| 방법 | 목적 |
| --- | --- |
| BatchNorm | 학습 안정화, 빠른 수렴 |
| Dropout | 과적합 감소 |
| Weight Decay | 가중치가 너무 커지는 것 방지 |

## 19. 실제 훈련 조합 정리

### 조합 1. 가장 단순한 기준 모델

```text
구조: Affine -> ReLU -> Affine -> Softmax
초기화: He
Optimizer: SGD
BatchNorm: 없음
Dropout: 없음
```

장점:

- 이해하기 쉬움
- 디버깅이 쉬움

단점:

- 정확도가 낮거나 학습이 느릴 수 있음

### 조합 2. 과제 시작 추천 모델

```text
구조: 784 -> Affine(512) -> ReLU -> Affine(10) -> Softmax
초기화: He
Optimizer: Adam
learning rate: 0.001
BatchNorm: 없음
Dropout: 없음
```

장점:

- 먼저 전체 학습 루프가 되는지 확인하기 좋음
- BatchNorm/Dropout 없이 버그를 줄일 수 있음

### 조합 3. 과제 권장 기본 모델

```text
구조:
784
-> Affine(512)
-> BatchNorm
-> ReLU
-> Dropout(0.2 또는 0.3)
-> Affine(256)
-> BatchNorm
-> ReLU
-> Dropout(0.2 또는 0.3)
-> Affine(10)
-> Softmax

초기화: He
Optimizer: Adam
learning rate: 0.001
batch size: 128
epochs: 15~20
```

장점:

- 정확도 95~97% 목표에 적합
- BatchNorm으로 학습 안정화
- Dropout으로 과적합 완화

주의:

- 구현할 부품이 많아 버그 추적이 어려울 수 있음

### 조합 4. 정확도 튜닝용

```text
구조: 784 -> 512 -> 256 -> 10
초기화: He
Optimizer: Adam
learning rate 후보: 0.001, 0.0005, 0.0001
Dropout 후보: 0.2, 0.3, 0.5
BatchNorm: 사용
epochs: 20 이상 실험
```

확인 순서:

```text
1. train loss가 줄어드는가
2. train accuracy가 올라가는가
3. test accuracy가 올라가는가
4. train만 높고 test가 낮으면 과적합인가
5. loss가 안 줄면 learning rate가 너무 큰가/작은가
```

### 조합 5. 과적합 확인용

```text
큰 모델
Dropout 없음
BatchNorm 선택
epoch 많이
```

목적:

- train accuracy와 test accuracy 차이를 확인
- Dropout/weight decay 필요성을 이해

### 조합 6. BatchNorm 효과 확인용

```text
같은 모델을 2개 학습
모델 A: BatchNorm 없음
모델 B: BatchNorm 있음
```

비교:

- loss가 더 빨리 줄어드는가
- 초기화에 덜 민감한가
- accuracy가 안정적인가

### 조합 7. Dropout 효과 확인용

```text
같은 모델을 2개 학습
모델 A: Dropout 없음
모델 B: Dropout 있음
```

비교:

- train accuracy는 조금 낮아질 수 있음
- test accuracy가 더 좋아질 수 있음
- 과적합이 줄어드는지 확인

## 20. 최종 추천 실험 순서

처음부터 큰 모델로 가지 말고 아래 순서로 확인합니다.

```text
1. 전체 테스트 통과
2. 작은 모델로 loss 감소 확인
3. Adam 적용
4. He 초기화 적용
5. hidden layer 1개로 95% 근처 확인
6. hidden layer 2개로 확장
7. BatchNorm 추가
8. Dropout 0.2 또는 0.3 추가
9. learning rate 조정
10. REPORT.md에 실험 기록
```

## 21. 한 문장 요약

신경망은 `Affine`으로 계산하고, `Activation`으로 비선형성을 만들고, `Loss`로 틀린 정도를 재고, `Backward`로 책임을 계산하고, `Optimizer`로 `W`, `b`를 고치며, `초기화`, `BatchNorm`, `Dropout`으로 학습을 더 안정적이고 잘 일반화되게 만드는 구조입니다.
