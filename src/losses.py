# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np

#교차 엔트로피 오차, y_pred : 모델이 예측한 확률, y_true : 정답
def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    # TODO: 정답 클래스 확률의 log 값을 이용해 batch 평균 cross entropy를 계산하세요.
    # 힌트: np.clip으로 log(0)을 피하고, np.arange(batch_size)로 정답 위치를 고릅니다.
    #raise NotImplementedError("cross_entropy_loss를 구현하세요.")

    #y_pred가 1차원인지 확인
    if y_pred.ndim == 1:
        #1차원 예측값을 2차원으로 바꿈
        y_pred = y_pred.reshape(1, y_pred.size)
        #정답도 배치 1개 모양으로 맞춤  ex : [[1.0, 2.0, 3.0]]
        y_true = y_true.reshape(1)

    #예측 확률이 너무 작으면 1e-7로 올리고, 1보다 크면 1로 제한
    y_pred = np.clip(y_pred, 1e-7, 1.0)
    #배치 사이즈 
    batch_size = y_pred.shape[0]
    #각 데이터에서 정답 클래스의 확률만 뽑은 뒤, log를 씌우고, 마이너스 붙이고/ 배치 크기로 나눔
    return -np.sum(np.log(y_pred[np.arange(batch_size), y_true])) / batch_size