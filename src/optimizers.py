# -*- coding: utf-8 -*-
"""파라미터 업데이트 규칙을 모아 둔 optimizer 모듈."""

import numpy as np


class SGD:
    """
    확률적 경사하강법(SGD).

    가장 단순한 optimizer로, 각 파라미터를 gradient 반대 방향으로 lr만큼 이동합니다.
    """

    def __init__(self, lr=0.01):
        """Args: lr: 한 번 업데이트할 때 gradient에 곱할 학습률."""
        self.lr = lr

    def update(self, params, grads):
        """params dict의 모든 파라미터를 제자리(in-place)에서 갱신합니다."""
        # TODO: params[key]를 gradient 반대 방향으로 업데이트하세요.
        #raise NotImplementedError("SGD.update를 구현하세요.")
        for key in params.keys():
            params[key] -= self.lr * grads[key]



class Adam:
    """
    Adam Optimizer.

    gradient의 이동평균(m)과 제곱 이동평균(v)을 함께 사용해 파라미터별 학습률을 조절합니다.
    MNIST 과제에서는 SGD보다 빠르게 손실이 내려가는지 비교해 볼 수 있습니다.
    """

    def __init__(self, lr=0.001):
        """Args: lr: Adam 업데이트의 기본 학습률."""
        self.lr = lr
        self.m, self.v = {}, {}
        self.t = 0

    def update(self, params, grads):
        """Adam 공식에 따라 params dict의 모든 파라미터를 갱신합니다."""
        # TODO: m, v 이동평균과 bias correction을 사용해 params를 업데이트하세요.
        #raise NotImplementedError("Adam.update를 구현하세요.")
        
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-7

        #최초 업데이트시
        if not self.m:
            for key, value in params.items():
                self.m[key] = np.zeros_like(value)
                self.v[key] = np.zeros_like(value)
        #업데이트 횟수 증가
        self.t += 1
        #파라미터 업데이트
        for key in params.keys():
            #이전 평균90%반영, 현재 gradient 10%반영
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads[key]
            #이전 평균 99.9% 반영, 현재 gradient 제곱을 0.1% 반영
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grads[key] ** 2)
            #초반에 m,v가 0에서 시작해서 너무 작게 잡히는 문제를 보정
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)
            #파라미터 업데이트, SGD처럼 그냥 gradient를 빼는 게 아니라, m_hat을 sqrt(v_hat)으로 나눠서 더 안정적
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)