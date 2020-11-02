import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import lpr_locate
from keras import models
from keras import optimizers, losses
import keras.backend as K
import copy
import heapq

size = 1280


PROVINCE_SAVER_DIR = "./saved_h5_model/"
DIGITS_SAVER_DIR = "./saved_h5_model/"

PROVINCES = ("京","闽","粤","苏","沪","浙")
LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z")


def trans_digits(data):
    # 由预测的标签编号，获得真实标签
    digits = ""
    for i in data:
        digits += LETTERS_DIGITS[i]
    return digits

def decode_predictions(preds:list,classes, top:int):
    # 解析预测结果
    max_num_index=map(preds.index, heapq.nlargest(top,preds))
    ret = list()
    if classes == 'province':
        arr = PROVINCES
    else:
        arr = LETTERS_DIGITS
    for i in max_num_index:
        temp = str('%0.2f%%'%(preds[i]*100))
        ret.append([i,arr[i], temp])
    return ret 

def predict(model_path, parts, ):
    # 清空session
    K.clear_session()

    # 加载模型
    model = models.load_model(model_path,compile=False)
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy, metrics=["accuracy"])

    predict = []
    length = len(parts)
    for img in parts:
        # 转换成二值化图像
        _, img_binary = cv.threshold(img, 190, 1, cv.THRESH_BINARY_INV)
        # Reshape节点的输入需要是(1, 32, 40, 1)形状的
        input_image = img_binary.reshape(1, 32, 40, 1)
        preds = model.predict(input_image).reshape(-1).tolist()
        # print(type(preds))
        if length == 1:
            ret = decode_predictions(preds, 'province', top=3)
            print('Predicted:', ret)
        else:
            ret = decode_predictions(preds, 'digits', top=3)
            print('Predicted:', ret)
        # 返回概率最大的下标
        max_index = preds.index(max(preds)) 
        predict.append(max_index)

    return predict

import time
def predict_car_license(parts, kind = "baseline"):
    '''
    kind = 'baseline' / 'pruned'
    '''
    print(kind)
    time_begin = time.time()
    # 输入为原始图片中分割出的小图
    lpr = ""
    # 第一个字符为省份
    province = predict(PROVINCE_SAVER_DIR+'province_%s.h5'%(kind),parts[:1])
    lpr += PROVINCES[province[0]]

    # 后续字符为字母和数字
    digits = predict(DIGITS_SAVER_DIR+'digits_%s.h5'%(kind),parts[1:])
    lpr += trans_digits(digits)

    print("花费时间：",time.time()-time_begin)
    return lpr

def show_card(img, card, parts):
    plt.figure(figsize=(10,5))
    # 绘制原图
    plt.subplot(1,2,1)
    origin = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    plt.imshow(origin), plt.axis('off')

    #绘制车牌
    plt.subplot(2, 2, 2)
    pil_card = cv.cvtColor(card, cv.COLOR_BGR2RGB)
    plt.imshow(pil_card), plt.axis('off')

    #绘制分割后的字符
    for i, part in enumerate(parts):
        plt.subplot(2, 2*len(parts), i + 1 + 3*len(parts))
        pil_part = Image.fromarray(np.uint8(part))
        plt.imshow(pil_part), plt.axis('off')

    plt.savefig('./static/images/target.jpg', format = 'jpg', bbox_inches = 'tight')
    # test_img = cv.imread('tmp.png')
    # cv.imshow("LPR predict",test_img)
    # cv.waitKey(0)


if __name__ =='__main__':
    image_name = sys.argv[1]
    print(image_name)
    kind = sys.argv[2]
    img = cv.imread(image_name)

    card, parts = lpr_locate.locate(img)

    show_card(img, card, parts)
    lpr = predict_car_license(parts,kind)
    print(lpr)
 
    cv.destroyAllWindows()   