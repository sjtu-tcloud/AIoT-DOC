from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify, session
from werkzeug.utils import secure_filename
import os
import cv2
import time
from datetime import timedelta
from model import User, Record, History


# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png','jpg','JPG','PNG','bmp'])

def is_allowed_file(filename):
    '''
    输入文件名，输出是否是允许的文件格式
    '''
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds = 1)

import lpr_locate
from predict_lpr_h5 import show_card,predict_car_license
import json,time

# 登录
@app.route('/login/',methods=['GET','POST'])
def login():
    print('login')
    print(request.method)
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get("password")
        # print("username:",username)
        # print("password:",password)
        ret = User.login(username,password)
        print(ret)
        if not ret:
            # 登录失败
            return render_template("login.html",flag=True,msg="账号或密码错误")
        else:
            ret = list(ret)
            # 在session中保存登录信息
            session['user'] = ret[0]['username']
            session['role'] = ret[0]['role']
            role = ret[0]['role']
            print(role)
            # 根据用户身份不同，导航到不同主页
            if role == 'admin':
                return redirect(url_for('adminHome'))
            else:
                return redirect(url_for('userHome'))
    return render_template("login.html",flag=False)

# 注册
@app.route('/register/',methods=['GET','POST'])
def register():
    print('register')
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get("password")
        checkPassword = request.form.get("checkPassword")
        if password != checkPassword:
            return render_template("user_register.html",flag=True,msg="请确保两次密码相同")
        user = User(username,password,"user")
        ret = user.save()
        if ret != True:
            return render_template("user_register.html",flag=True,msg=ret)
        return render_template("user_register.html",flag=True,msg="注册成功！")
    return render_template("user_register.html",flag=False) 

# 用户主页
@app.route('/user_home/',methods=['GET','POST'])
def userHome():
    return render_template("user_car_query.html")

# 管理员主页
@app.route('/admin_home/',methods=['GET','POST'])
def adminHome():
    return render_template("admin_car_list.html")

import time
import datetime
# 添加上传文件路由
@app.route('/upload/', methods=['POST','GET'])
def upload():
    print('upload')
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

        # 再次读取图片
        img = cv2.imread(os.path.join(basepath,'static/images','target.jpg'))

        # 分割图片
        print(img)
        card, parts = lpr_locate.locate(img)

        show_card(img, card, parts)
        
        # 获取车牌号
        # pruned
        time_begin = time.time()
        kind="pruned"
        P_result = predict_car_license(parts,kind)
        P_time = round(time.time() - time_begin,5)

        # baseline
        time_begin = time.time()
        kind="baseline"
        B_result = predict_car_license(parts,kind)
        B_time = round((time.time() - time_begin)*2,5) 
        
        # 将结果保存在session中
        session['B_result'] = B_result
        session['B_time'] = B_time
        session['P_result'] = P_result
        session['P_time'] = P_time

        ret = Record.query_car(P_result)
        msg = "车牌识别结果如下"
        if ret != "该车不在停车场内":
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            now = datetime.datetime.strptime( now, '%Y-%m-%d %H:%M:%S')
            start = datetime.datetime.strptime(ret[0]['enter_time'], '%Y-%m-%d %H:%M:%S')
            minutes_delta = int((now-start).seconds/60)+1
            cost = round(minutes_delta*0.1,1)
            msg += ",预计费用"+str(cost)+"元"
        return render_template('recgonize.html', flag=flag, B_result=B_result,B_time=B_time,
        P_result=P_result,P_time=P_time,msg=msg)
    
    return render_template('recgonize.html',flag=flag)

@app.route('/allow-enter/',methods=['GET','POST'])
def allowEnter():
    print("allowEnter")
    lpr = request.get_json()['lpr']
    print(lpr)
    # 从session中获取内容
    B_result = session.get('B_result')
    B_time = session.get('B_time')
    P_result = session.get('P_result')
    P_time = session.get('P_time')

    if lpr=='':
        lpr = P_result
    car = Record(lpr) 

    # 车辆进场 
    if Record.is_active(lpr):
        print("车辆已在停车场中")
        return jsonify({"error":1003,"msg":"车辆已在停车场"})
        # return render_template('upload.html',flag=False, B_result=B_result,B_time=B_time,P_result=P_result,P_time=P_time,msg=u"车辆已在停车场中") 
    car.car_enter()
    print(car)

    # 计入历史
    his = History('enter',car.__str__())
    his.save()
    print(his)

    return jsonify({"code":200,"msg":"车辆进场成功","his_type":his.his_type,"his_msg":his.__str__(),"lpr":lpr})
    # return render_template('upload.html', flag=False, B_result=B_result,B_time=B_time,P_result=P_result,P_time=P_time,msg=u"车辆进场成功")


@app.route('/allow-leave/',methods=['GET','POST'])
def allowLeave():
    print('allow_leave')
    plate_number = request.get_json()['lpr']
    print(plate_number)
    # 从session中获取内容
    B_result = session.get('B_result')
    B_time = session.get('B_time')
    P_result = session.get('P_result')
    P_time = session.get('P_time')
    if plate_number=='':
        plate_number = P_result
    print(plate_number)
    if Record.is_active(plate_number):
        # 车辆离场
        record = Record(plate_number)
        record.car_leave()
        print(record)

        # 计入历史
        his = History('leave',record.__str__())
        his.save()
        print(his)

        return jsonify({"code":200,"msg":"车辆离场成功","his_type":his.his_type,"his_msg":his.__str__(),"lpr":plate_number})
        # return "车辆离场成功"
        # return render_template('upload.html', flag=False, B_result=B_result,B_time=B_time,P_result=P_result,P_time=P_time,msg=u"车辆离场成功")
    else:
        return jsonify({"error":1004,"msg":"车辆不在停车场"})
        # return render_template('upload.html', flag=False, B_result=B_result,B_time=B_time,P_result=P_result,P_time=P_time,msg=u"车辆不在停车场中")

@app.route('/my_history/', methods=['POST','GET'])
def get_my_history():
    print('get my history')
    plate_number = request.get_json()['lpr']
    my_history = Record.query_car_records(plate_number)
    if not my_history:
        return jsonify({"code":404, "msg":"记录为空，请检查是否正确输入了车牌号"})
    
    result = list()
    
    for m in my_history:
        if m['is_active'] == False:
            temp = {'lpr':m['plate_number'],'enter_time':m['enter_time'],'leave_time':m['leave_time'],
            'is_active':0,'cost':m['cost']}
        else:
            temp = {'lpr':m['plate_number'],'enter_time':m['enter_time'],'leave_time':m['leave_time'],
            'is_active':1,'cost':m['cost']}
        result.append(temp)
    # 计算最近一次的费用
    if result[-1]['is_active'] == 0:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        now = datetime.datetime.strptime( now, '%Y-%m-%d %H:%M:%S')
        start = datetime.datetime.strptime(result[-1]['enter_time'], '%Y-%m-%d %H:%M:%S')
        minutes_delta = int((now-start).seconds/60)+1
        result[-1]['cost'] = round(minutes_delta*0.1,1)
    
    result.reverse()
    print(result[:3])
    return jsonify(result)

@app.route('/active_cars/',methods=['GET'])
def get_active_cars():
    print('get active cars')
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    now = datetime.datetime.strptime( now, '%Y-%m-%d %H:%M:%S')
    cars = Record.get_active_cars()
    abnormal = 0
    result = list()
    for car in cars:
        start = datetime.datetime.strptime(car['enter_time'], '%Y-%m-%d %H:%M:%S')
        minutes = int((now-start).seconds/60)
        cost = round(minutes*0.1,1)
        if (minutes/60)>24:
            normal -=1
            abnormal +=1
        temp = {'lpr':car['plate_number'],'enter_time':car['enter_time'],'cost':cost}
        result.append(temp)
    print(result)
    normal = len(result) - abnormal
    result.append({"normal":normal,"abnormal":abnormal})
    return jsonify(result)

@app.route('/inout_history/',methods=['GET','POST'])
def inout_history():
    return render_template("admin_inout_history.html")

@app.route('/describe/',methods=['GET'])
def describe():
    return render_template("describe.html")

@app.route('/user_describe/',methods=['GET'])
def user_describe():
    return render_template("user_describe.html")

@app.route('/history/',methods=['GET'])
def get_history():
    print('get_history')
    history=History.query_history()
    print(history.count())
    result=list()
    result.append({"his_type": "test_his", "msg":"test_msg"})
    num = 0
    for h in history:
        if num < 30:
            temp = {"his_type":h["his_type"],"msg":h["msg"]}
            num += 1
            result.append(temp)
        else:
            break
    print(result[:2])
    result.reverse()
    return jsonify(result)


if __name__=='__main__':
    # app.debug = True
    #指定浏览器渲染的文件类型，和解码格式；
    app.config['JSONIFY_MIMETYPE'] ="application/json;charset=utf-8"
    app.config['JSON_AS_ASCII'] = False
    # 为了使用session，需要设置secret key
    app.config["SECRET_KEY"] = "renyaodanjun"
    app.run(host='localhost',port=8008, debug=True)