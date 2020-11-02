from net import Net
import torch
import cv2
import numpy as np
# 加载模型
model = Net()
params = torch.load('./model/model_10_2_epoch.pth')
model.load_state_dict(params)
model.eval()
upload_path='./static/images/5.JPG'
# 读取图片和推断


img = cv2.imread(upload_path,cv2.IMREAD_GRAYSCALE)
print(img)

# TARGET_IMG_SIZE = 32
# img=img.resize((1,TARGET_IMG_SIZE, TARGET_IMG_SIZE))
# print(img)

arr = np.asarray(img,dtype="float32")
data_x = np.empty((1,1,32,32),dtype="float32")
data_x[0 ,:,:,:] = arr
data_x = data_x / 255
data_x = torch.from_numpy(data_x)
print(data_x)
print(data_x.shape)

with torch.no_grad():
    out = model(data_x)

# 处理out，例如进行nms和结果显示，该部分省略
print(out)