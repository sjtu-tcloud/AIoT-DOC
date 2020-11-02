# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
 
def yuv2rgb(yuvfilename, W, H, startframe, totalframe, fps=12, show=False, out=False):
    start = time.time()
  # 从第startframe（含）开始读（0-based），共读totalframe帧
    arr = np.zeros((totalframe,H,W,3), np.uint8)
    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 12
    # size = (591,705) #图片的分辨率片
    file_path = r"akiyo_qcif_" + str(int(time.time())) + ".mp4"#导出路径
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    print((W,H))
    video = cv2.VideoWriter( file_path, fourcc, fps, (176,144))

    plt.ion()
    with open(yuvfilename, 'rb') as fp:
        seekPixels = startframe * H * W * 3 // 2
        fp.seek(8 * seekPixels) #跳过前startframe帧
        for i in range(totalframe):
            print(i)
            oneframe_I420 = np.zeros((H*3//2,W),np.uint8)
            for j in range(H*3//2):
                for k in range(W):
                    oneframe_I420[j,k] = int.from_bytes(fp.read(1), byteorder='little', signed=False)
            oneframe_RGB = cv2.cvtColor(oneframe_I420,cv2.COLOR_YUV2RGB_I420)
            if show:
                plt.imshow(oneframe_RGB)
                plt.show()
                plt.pause(0.001)
            if out:
                #outname = "./akiyo/"+yuvfilename[:-4]+'_'+str(startframe+i)+'.png'
                #cv2.imwrite(outname,oneframe_RGB[:,:,::-1])
                #img = cv2.imread(outname)
                #video.write(img)
                video.write(oneframe_RGB[:,:,::-1])
            arr[i] = oneframe_RGB
        video.release()
    duration = time.time()-start
    print("耗时：{}".format(duration))
    return arr

# 图片合成视频
def picvideo(path,size):
    # path = r'C:\Users\Administrator\Desktop\1\huaixiao\\'#文件路径
    filelist = os.listdir(path) #获取该目录下的所有文件名
 
    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 12
    # size = (591,705) #图片的分辨率片
    file_path = "akiyo_qcif" + str(int(time.time())) + ".mp4"#导出路径
    fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')#不同视频编码对应不同视频格式
#    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#不同视频编码对应不同视频格式
  
    video = cv2.VideoWriter( file_path, fourcc, fps, size )
 
    for item in filelist:
        if item.endswith('.png'):   #判断图片后缀是否是.png
            item = path + '/' + item 
            img = cv2.imread(item)  #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            video.write(img)        #把图片写进视频
 
    video.release() #释放

import sys

if __name__ == '__main__':
    print('程序名称为：{}，第一个参数为：{}，第二个参数为：{}'.format(sys.argv[0], sys.argv[1], sys.argv[2]))
    startframe = int(sys.argv[1])
    endframe = int(sys.argv[2])
    fps = int(sys.argv[3])
    video = yuv2rgb(r'akiyo_qcif.yuv', 176, 144, startframe, endframe, fps=fps, show=False, out=True)
    #picvideo('./akiyo',(176,144))
