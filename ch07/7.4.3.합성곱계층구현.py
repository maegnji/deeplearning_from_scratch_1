import numpy as np
import sys, os
sys.path.append('/Users/maengseongjin/Library/Mobile Documents/com~apple~CloudDocs/7. python/220201_deeplearning_from_scratch_1') # 부모 디렉토리 파일을 가져올 수 있도록 설정
from common.util import im2col

x1 = np.random.rand(1,3,7,7) # (데이터수, 채널, 높이, 너비)
# imput_data (데이터 수, 채널 수, 높이, 너비)의 4차원 배열로 이뤄진 입력 데이터
# filter_h 필터의 높이
# filter_w 필터의 너비
# stride 스트라이드
# pad 패딩
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape) # (9.75)

x2 = np.random.rand(10, 3, 7, 7) # 데이터 10개
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) # (90, 75)


class Convolution:
    # 필터(가중치), 편향, 스트라이드, 패딩 초기화
    # 필터는 (FN, C, FH, FW)의 4차원
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        # 연산 후 높이
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        # 연산 후 너비
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        # 입력데이터를 im2col을 이용해서 2차원배열로 전개
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 필터도 reshape을 통해 2차원 배열로 전개 (-1이 있으므로, FN이후 모두 1차원으로 변환)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        # 출력모양으로 재배열
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 전개(1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 최대값(2)
        # 1번째 차원의 축의 최대값 구함 (axis=0이면 열 / axis=1이면 행)
        out = np.max(col, axis=1)

        # 성형(3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

