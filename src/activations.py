# -*- coding: utf-8 -*-
"""
활성화 함수 모음.

학생 구현 대상:
- ReLU.forward, ReLU.backward
- Softmax.forward, Softmax.backward
"""

import numpy as np


class ReLU:
    """
    ReLU(Rectified Linear Unit) 활성화 함수.

    은닉층에서 음수 값은 0으로 막고, 양수 값은 그대로 통과시킵니다.
    forward에서 만든 mask는 backward 때 "어느 위치로 gradient를 흘릴지" 결정하는 데 사용됩니다.
    """

    def forward(self, x):
        """
        Args:
            x: 임의 shape의 입력 배열

        Returns:
            x와 같은 shape. x > 0인 위치만 원래 값을 유지합니다.
        """
        # TODO: x > 0 위치를 self.mask에 저장하고, 음수/0 위치는 0으로 바꾸세요.
        # raise NotImplementedError("ReLU.forward를 구현하세요.")

        #x = np.array([1,-2,3,0])이라면 self.mask는 [False, True, False, True] 로 저장
        self.mask = (x <= 0)
        #원본 x를 수정하면 안되니 복사본 생성
        out = x.copy()
        #self.mask에서 True였던 부분을 0으로 변경
        out[self.mask] = 0

        return out


    def backward(self, dout):
        """
        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            ReLU 입력 x에 대한 gradient. forward 때 x <= 0이었던 위치는 0입니다.
        """
        # TODO: forward에서 저장한 self.mask를 이용해 gradient가 흐를 위치만 남기세요.
        # raise NotImplementedError("ReLU.backward를 구현하세요.")

        #dout은 뒤에서 넘어온 gradient, self.mask 가 True였던 부분은 W를 바꿀 필요가 없으니 전달하지 않음
        dout[self.mask] = 0
        #0으로 바꾸고 return하기 위해 배열 하나 생성
        dx = dout

        return dx


class Softmax:
    """
    Softmax 출력층.

    각 샘플의 로짓(logit)을 클래스별 확률로 바꿉니다.
    exp 계산 전에 행별 최댓값을 빼면 큰 숫자에서 overflow가 나는 것을 줄일 수 있습니다.
    """

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_classes) 로짓

        Returns:
            (batch_size, num_classes) 확률. 각 행의 합은 1입니다.
        """
        # TODO: 수치 안정성을 위해 row별 max를 뺀 뒤 softmax 확률을 계산하세요.
        # 힌트: np.max(..., axis=1, keepdims=True), np.exp, np.sum을 사용합니다.
        #raise NotImplementedError("Softmax.forward를 구현하세요.")

        #overflow 방지하기 위해 각 행에서 가장 큰 값을 빼줌, 정규화 전처리
        x = x - np.max(x, axis=1, keepdims=True)
        #지수함수로 변경
        exp_x = np.exp(x)
        #출력값 계산
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        return out


    def backward(self, dout):
        """
        Softmax와 Cross Entropy를 함께 미분한 gradient를 train()에서 직접 만들기 때문에
        여기서는 받은 gradient를 그대로 통과시킵니다.
        """
        # TODO: train()에서 만든 gradient를 그대로 반환하세요.
        #raise NotImplementedError("Softmax.backward를 구현하세요.")
        return dout