# -*- coding: utf-8 -*-
 
import math
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
 
 
def readyuv420(filename, bitdepth, W, H, startframe, totalframe, show=False):
  # 从第startframe（含）开始读（0-based），共读totalframe帧
 
    uv_H = H // 2
    uv_W = W // 2
    
    if bitdepth == 8:
        Y = np.zeros((totalframe, H, W), np.uint8)
        U = np.zeros((totalframe, uv_H, uv_W), np.uint8)
        V = np.zeros((totalframe, uv_H, uv_W), np.uint8)
    elif bitdepth == 10:
        Y = np.zeros((totalframe, H, W), np.uint16)
        U = np.zeros((totalframe, uv_H, uv_W), np.uint16)
        V = np.zeros((totalframe, uv_H, uv_W), np.uint16)
    
    plt.ion()
    
    bytes2num = partial(int.from_bytes, byteorder='little', signed=False)
    
    bytesPerPixel = math.ceil(bitdepth / 8)
    seekPixels = startframe * H * W * 3 // 2
    fp = open(filename, 'rb')
    fp.seek(bytesPerPixel * seekPixels)
    
    for i in range(totalframe):
    
        for m in range(H):
            for n in range(W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    Y[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    Y[i, m, n] = np.uint16(pel)
        
        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    U[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    U[i, m, n] = np.uint16(pel)
        
        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    V[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    V[i, m, n] = np.uint16(pel)
 
        if show:
            print(i)
            plt.subplot(131)
            plt.imshow(Y[i, :, :], cmap='gray')
            plt.subplot(132)
            plt.imshow(U[i, :, :], cmap='gray')
            plt.subplot(133)
            plt.imshow(V[i, :, :], cmap='gray')
            plt.show()
            plt.pause(1)
            #plt.pause(0.001)
        
    if totalframe==1:
        return Y[0], U[0], V[0]
    else:
        return Y,U,V
 
 
if __name__ == '__main__':
    y, u, v = readyuv420(r'akiyo_qcif.yuv', 8, 176, 144, 0, 5, True)
    print(y.shape,u.shape,v.shape)
