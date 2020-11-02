from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify, session
from werkzeug.utils import secure_filename
import os
import time
from datetime import timedelta
from train import train_net
from net import Net
import torch
import cv2
import numpy as np

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png','jpg','JPG','PNG','bmp'])

def is_allowed_file(filename):
    '''
    输入文件名，输出是否是允许的文件格式
    '''
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

# import lpr_locate
# from predict_lpr_h5 import show_card,predict_car_license
import json,time
app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds = 1)

@app.route('/test/',methods=['GET','POST'])
def inference():
    flag = True
    if request.method == 'POST':
        print('Post')
        flag = False
        ff = request.files['file']
        if not (ff and is_allowed_file(ff.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片格式,，仅限于png、PNG、jpg、JPG、bmp"})

        # user_input = request.form.get("mood")

        # 当前文件所在位置
        basepath = os.path.dirname(__file__)

        # 确保路径存在
        upload_path = os.path.join(basepath,'static/images', secure_filename(ff.filename))

        ff.save(upload_path)

          # 使用opencv转换图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath,'static/images','target.jpg'),img)

        # 加载模型
        model = Net()
        params = torch.load('./static/model/model_10_2_epoch.pth')
        model.load_state_dict(params)
        model.eval()
        # upload_path='./static/images/5.JPG'
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
        out = out[0,:]
        out = out.tolist()
        max_index = out.index(max(out)) # 返回最大值的索引
        result = '识别结果是：'+ str(max_index) + '\n'
        dd = dict()
        result += '网络输出:\n'
        for i in range(10):
            result += str(i) + ':' + str(out[i]) + '\n'
            dd[str(i)]=out[i]
        print(dd)
        # result += '网络输出：'+str(dd)
        # 处理out，例如进行nms和结果显示，该部分省略
        # print(out[0,:])
        return render_template("test_net.html",flag=flag,result = result)

    return render_template("test_net.html",flag=flag)

@app.route('/dataset/',methods=['GET'])
def choose_dataset():
    return render_template("dataset.html")

@app.route('/train/',methods=['GET','POST'])
def train_parameter():
    return render_template("train.html")

@app.route('/test_video/',methods=['GET','POST'])
def test_video():
    return render_template("test_video.html")

@app.route('/train_net/',methods=['GET','POST'])
def train_network():
    if request.method == 'POST':
        model_name = request.form.get('model_name')
        optimizer = request.form.get('optimizer')
        activate = request.form.get('activate')
        learning_rate = float(request.form.get('learning_rate'))
        # print(type(learning_rate))
        # print(learning_rate)
        epoch = int(request.form.get('epoch'))
        batch_size = int(request.form.get('batch_size'))
        dropout = float(request.form.get('dropout'))
        print(dropout)
        platform = 'cpu'
        train_net(model_name,learning_rate, batch_size, optimizer,epoch,platform,activate)
    return render_template("train_net.html")

if __name__=='__main__':
    # app.debug = True
    #指定浏览器渲染的文件类型，和解码格式；
    app.config['JSONIFY_MIMETYPE'] ="application/json;charset=utf-8"
    app.config['JSON_AS_ASCII'] = False
    # 为了使用session，需要设置secret key
    app.config["SECRET_KEY"] = "renyaodanjun"
    app.run(host='0.0.0.0',port=8008, debug=True)