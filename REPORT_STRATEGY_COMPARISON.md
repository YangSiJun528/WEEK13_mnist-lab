# MNIST 전략 비교 실험 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| **반** | (기입) |
| **팀원** | (기입) |

---

## 1. 실험 목적

MNIST 10-class 분류를 **NumPy만으로 구현한 신경망**으로 수행하고, 다음 세 축이 학습 과정과 테스트 성능에 주는 영향을 비교한다.

1. SGD vs Adam(ADRM)
2. BatchNorm, Dropout 유무
3. learning rate 0.01, 0.001, decay loss 비교

모든 실험에서 **ReLU 활성화 함수와 He 초기화는 고정**한다. 기반 코드는 `67c2996e378527ce7dad540cf549b158ee616b3a` 시점의 `src/` 구현을 그대로 사용하고, 비교 실험 코드는 `mnist_lab_for_test.ipynb` 안에서만 실행한다.

---

## 2. 모델 구조

| 구분 | 내용 |
| --- | --- |
| **입력** | 784차원 벡터 (28x28 픽셀, 0~1 정규화) |
| **기본 은닉층** | Affine(512) -> BatchNorm -> ReLU -> Dropout -> Affine(256) -> BatchNorm -> ReLU -> Dropout |
| **출력층** | Affine(10) -> Softmax |
| **손실 함수** | Cross Entropy Loss |
| **고정 조건** | ReLU, He 초기화 |

기본 구조는 `784 -> 512 -> 256 -> 10`이다. 비교 전략에 따라 optimizer, learning rate, BatchNorm 사용 여부, Dropout 사용 여부만 바꾼다.

---

## 3. 학습 설정 및 예상 테스트 횟수

공통 설정은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| epochs | 20 |
| batch_size | 128 |
| train metric 측정 | 고정 train subset 10,000개 |
| validation/test metric 측정 | 전체 `x_test`, `y_test` |
| seed | 42 |
| Dropout 비율 | 0.5 |
| BatchNorm momentum | 0.9 |
| 초기화 | He |

총 예상 테스트 횟수는 **6회**이다. `adam_baseline`을 세 비교 그룹에서 공통 기준으로 재사용하여 중복 실행을 줄인다.

| strategy | 비교 그룹 | optimizer | lr | lr schedule | BatchNorm | Dropout | 예상 params |
| --- | --- | --- | ---: | --- | --- | --- | ---: |
| `adam_baseline` | 공통 기준 | Adam | 0.001 | 없음 | on | on | 537,354 |
| `sgd_lr_0_01` | SGD vs Adam | SGD | 0.01 | 없음 | on | on | 537,354 |
| `no_batchnorm` | BatchNorm 유무 | Adam | 0.001 | 없음 | off | on | 535,818 |
| `no_dropout` | Dropout 유무 | Adam | 0.001 | 없음 | on | off | 537,354 |
| `adam_lr_0_01` | 학습률 비교 | Adam | 0.01 | 없음 | on | on | 537,354 |
| `adam_lr_decay` | 학습률 비교 | Adam | 0.01 시작 | `0.01 * 0.6^epoch` | on | on | 537,354 |

파라미터 수 계산:

- Affine 계층: `784*512+512 + 512*256+256 + 256*10+10 = 535,818`
- BatchNorm 학습 파라미터: `gamma1/beta1 512*2 + gamma2/beta2 256*2 = 1,536`
- BatchNorm on 모델: `535,818 + 1,536 = 537,354`
- Dropout, ReLU, Softmax는 학습 파라미터가 없다.
- Adam의 `m`, `v`는 optimizer state이므로 모델 파라미터 수에 포함하지 않는다.

---

## 4. 실행 및 기록 방식

실행은 Google Colab에서 `mnist_lab_for_test.ipynb`의 비교 실험 셀을 실행한다. 로컬에서는 학습을 실행하지 않는다.

노트북은 실행 결과를 다음 파일로 저장한다.

| 파일 | 내용 |
| --- | --- |
| `experiment_logs/grouped_strategy_run_<timestamp>.txt` | 사람이 읽을 수 있는 전체 epoch 로그 |
| `experiment_logs/grouped_strategy_run_<timestamp>.csv` | strategy, epoch, lr, loss, accuracy, params 등 구조화 로그 |
| `experiment_logs/grouped_strategy_summary_<timestamp>.md` | 전체 비교 수치표 |
| `report_assets/*.svg` | 리포트에 붙일 그룹별 비교 그래프 |

CSV에는 다음 항목이 기록된다.

```text
strategy,label,optimizer,epoch,lr,train_loss,train_acc,val_loss,val_acc,params,
use_batchnorm,use_dropout,dropout_ratio,init_method
```

---

## 5. 전체 비교

전체 비교는 그래프 없이 수치표로만 정리한다. Colab 실행 후 `grouped_strategy_summary_<timestamp>.md`의 표를 아래에 붙인다.

| strategy | optimizer | final val acc | best val acc | best epoch | train acc | val loss | final lr | params | BN | Dropout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `adam_baseline` | Adam | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | 0.001000 | 537,354 | True | True |
| `sgd_lr_0_01` | SGD | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | 0.010000 | 537,354 | True | True |
| `no_batchnorm` | Adam | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | 0.001000 | 535,818 | False | True |
| `no_dropout` | Adam | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | 0.001000 | 537,354 | True | False |
| `adam_lr_0_01` | Adam | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | 0.010000 | 537,354 | True | True |
| `adam_lr_decay` | Adam | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | (실행 후 입력) | 537,354 | True | True |

---

## 6. SGD vs Adam(ADRM) 비교

비교 대상:

| strategy | optimizer | lr | BatchNorm | Dropout |
| --- | --- | ---: | --- | --- |
| `sgd_lr_0_01` | SGD | 0.01 | on | on |
| `adam_baseline` | Adam | 0.001 | on | on |

확인할 점:

- Adam이 SGD보다 같은 epoch 안에서 더 빠르게 validation loss를 낮추는가
- SGD가 더 느리지만 안정적으로 수렴하는가
- 최종 정확도뿐 아니라 epoch별 loss 감소 속도를 함께 비교한다.

![SGD vs Adam validation accuracy](report_assets/optimizer_sgd_vs_adam_val_accuracy.svg)

![SGD vs Adam validation loss](report_assets/optimizer_sgd_vs_adam_val_loss.svg)

---

## 7. BatchNorm, Dropout 유무 비교

비교 대상:

| strategy | BatchNorm | Dropout | 목적 |
| --- | --- | --- | --- |
| `adam_baseline` | on | on | 기준 |
| `no_batchnorm` | off | on | BatchNorm 효과 확인 |
| `no_dropout` | on | off | Dropout 일반화 효과 확인 |

확인할 점:

- BatchNorm 제거 시 validation loss가 더 흔들리거나 수렴이 느려지는가
- Dropout 제거 시 train accuracy와 validation accuracy 차이가 커지는가
- `no_dropout`은 train loss가 빠르게 내려가도 validation loss가 높아지는 과적합 패턴이 있는지 확인한다.

![BatchNorm Dropout validation accuracy](report_assets/regularization_bn_dropout_val_accuracy.svg)

![BatchNorm Dropout validation loss](report_assets/regularization_bn_dropout_val_loss.svg)

![BatchNorm Dropout train validation gap](report_assets/regularization_bn_dropout_train_val_gap.svg)

---

## 8. 학습률 비교

비교 대상:

| strategy | optimizer | lr 설정 | 목적 |
| --- | --- | --- | --- |
| `adam_baseline` | Adam | 0.001 고정 | 기준 학습률 |
| `adam_lr_0_01` | Adam | 0.01 고정 | 큰 학습률 |
| `adam_lr_decay` | Adam | `0.01 * 0.6^epoch` | 큰 lr로 시작 후 감소 |

확인할 점:

- `0.01` 고정이 초반 수렴은 빠르지만 후반 loss가 흔들리는가
- `0.001` 고정이 더 안정적으로 수렴하는가
- decay loss 전략이 초반 속도와 후반 안정성을 동시에 얻는가

![Learning rate validation accuracy](report_assets/learning_rate_comparison_val_accuracy.svg)

![Learning rate validation loss](report_assets/learning_rate_comparison_val_loss.svg)

![Learning rate schedule](report_assets/learning_rate_comparison_lr_schedule.svg)

---

## 9. 결론 작성 기준

Colab 실행 후 아래 기준으로 결론을 작성한다.

- 최종 정확도 97% 이상이면 과제 목표는 달성한 것으로 본다.
- optimizer 비교는 정확도보다 loss 감소 속도와 안정성을 우선 해석한다.
- BatchNorm/Dropout 비교는 train-validation gap과 validation loss를 같이 본다.
- learning rate 비교는 최종 정확도, 최고 epoch, validation loss 흔들림을 같이 본다.
- 전체 6개 전략의 순위는 수치표로만 제시하고, 세부 해석은 세 비교 그룹 안에서 따로 작성한다.
