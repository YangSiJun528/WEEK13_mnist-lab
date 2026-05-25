# -*- coding: utf-8 -*-
"""
신경망 layer 모음.

학생 구현 대상:
- Affine.forward, Affine.backward
- BatchNorm.forward, BatchNorm.backward
- Dropout.forward, Dropout.backward
"""

import numpy as np


class Affine:
    """
    완전연결층(Fully Connected Layer).

    수식은 y = xW + b 입니다.
    MNIST에서는 784개 픽셀 입력을 은닉층/출력층 차원으로 선형 변환하는 역할을 합니다.
    """

    def __init__(self, W, b):
        """가중치 W와 편향 b를 외부 params dict와 같은 배열 객체로 공유합니다."""
        self.W = W
        self.b = b

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim)
        """
        # TODO: backward에서 사용할 입력 x를 저장하고 x @ W + b를 반환하세요.
        #raise NotImplementedError("Affine.forward를 구현하세요.")

        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        """
        Args:
            dout: (batch_size, output_dim)

        Returns:
            dx: (batch_size, input_dim)

        Side effects:
            self.dW, self.db에 optimizer가 사용할 gradient를 저장합니다.
        """
        # TODO: self.dW, self.db, dx를 계산하세요.
        # 힌트: dW = x.T @ dout, db = batch 방향 합, dx = dout @ W.T
        #raise NotImplementedError("Affine.backward를 구현하세요.")

        #x,W,b 각각에 대한 gradient를 구함, 곱하기 역전파 상황이니 dx엔 W를, dW엔 x를 곱함
        dx = np.dot(dout, self.W.T)
        #가중치 W에 대한 gradient, W를 바꾸면 loss가 어떻게 변하는지 알려주는 값
        self.dW = np.dot(self.x.T, dout)
        #편향 b는 batch의 모든 데이터에 똑같이 더해졌기 때문에, batch방향으로 전부 더함
        self.db = np.sum(dout, axis=0)

        #W와 b는  현재 층이 가진 값들이므로 넘겨줄 필요가 없고, dx만 앞 레이어로 전달함
        return dx

class BatchNorm:
    """
    Batch Normalization.

    미니배치 단위로 각 feature의 평균과 분산을 맞춰 학습을 안정화합니다.
    train=True일 때는 현재 배치 통계를 쓰고, 추론 때는 누적 running_mean/running_var를 사용합니다.
    """

    def __init__(self, gamma, beta, momentum=0.9):
        """
        Args:
            gamma: 정규화된 값을 다시 scale하는 학습 파라미터
            beta: 정규화된 값에 더하는 shift 학습 파라미터
            momentum: running_mean/running_var 이동평균 비율
        """
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = np.zeros_like(beta)
        self.running_var = np.zeros_like(beta)
        self.eps = 1e-7

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, feature_dim)
            train: True면 배치 통계, False면 running 통계 사용

        Returns:
            정규화 후 gamma, beta가 적용된 배열
        """
        # TODO: train=True에서는 batch mean/var로 정규화하고 running 통계를 갱신하세요.
        # TODO: train=False에서는 running_mean/running_var를 사용하세요.
        #raise NotImplementedError("BatchNorm.forward를 구현하세요.")

        if train:
            #현재 batch의 평균
            mu = np.mean(x, axis=0)
            #x에서 평균을 뺀 값
            xc = x - mu
            #현재 batch의 분산
            var = np.mean(xc ** 2, axis=0)
            #현재 batch의 표준편차
            std = np.sqrt(var + self.eps)
            #a=xW+b를 표준화한 값
            x_hat = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.x_hat = x_hat
            self.std = std

            #running_mean : 학습 중 여러 batch의 평균을 조금씩 누적해둔 값, 조금씩 갱신
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            #학습 중 저장해둔 평균을 뺌
            xc = x - self.running_mean
            #학습 중 저장해둔 분산으로 표준화
            x_hat = xc / np.sqrt(self.running_var + self.eps)

        #표준화된 값을 다시 조정, gamma는 scale, beta는 shift역할
        out = self.gamma * x_hat + self.beta
        return out


    def backward(self, dout):
        """
        BatchNorm 입력 x, scale gamma, shift beta에 대한 gradient를 계산합니다.

        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            dx: BatchNorm 입력 x에 대한 gradient
        """
        # TODO: self.dbeta, self.dgamma, dx를 계산하세요.
        # 힌트: 먼저 dbeta와 dgamma shape가 beta/gamma와 같은지 확인합니다.
        #raise NotImplementedError("BatchNorm.backward를 구현하세요.")

        #batchNorm 결과를 다음 계층으로 넘김. beta에 대한 gradient, 더하기 역전파라 batch 방향으로 전부 더함
        self.dbeta = np.sum(dout, axis=0)
        #gamma에 대한 gradient, gamma는 x_hat에 곱해졌기 때문에  x_hat * dout을 더함
        self.dgamma = np.sum(self.x_hat * dout, axis=0)
        #forward에서 hat * gamma였으므로 역전파에선 gamma를 곱함
        dx_hat = dout * self.gamma
        
        dxc = dx_hat / self.std
        dstd = -np.sum(dx_hat * self.xc / (self.std ** 2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        return dx


class Dropout:
    """
    Dropout.

    학습 중 일부 뉴런 출력을 무작위로 0으로 만들어 과적합을 줄입니다.
    이 구현은 추론 시 출력에 (1 - drop_ratio)를 곱하는 기본 dropout 방식을 사용합니다.
    """

    def __init__(self, drop_ratio=0.5):
        """Args: drop_ratio: 학습 중 0으로 만들 뉴런 비율."""
        self.drop_ratio = drop_ratio

    def forward(self, x, train=True):
        """
        Args:
            x: 입력 배열
            train: True면 무작위 mask 적용, False면 평균적인 출력 크기로 scale
        """
        # TODO: train=True에서는 mask를 만들고 x에 곱하세요.
        # TODO: train=False에서는 x * (1 - drop_ratio)를 반환하세요.
        #raise NotImplementedError("Dropout.forward를 구현하세요.")
    
        #학습모드에선 x와 같은 shape의 랜덤 배열을 만들고, drop_ratio보다 큰 위치만 True
        if train:
            self.mask = np.random.rand(*x.shape) > self.drop_ratio
            return x * self.mask
        #추론모드에선 뉴런을 끄지 않고, 학습 때 살아남은 비율만큼 값을 줄여줌
        else : 
            return x * (1.0 - self.drop_ratio)

    def backward(self, dout):
        """forward에서 꺼졌던 뉴런 위치에는 gradient도 흘리지 않습니다."""
        # TODO: forward에서 만든 mask를 dout에 곱하세요.
        #raise NotImplementedError("Dropout.backward를 구현하세요.")
        #그대로 전파, 꺼졌던 뉴런의 gradient도 막기 위해 mask를 곱해줌
        return dout * self.mask
