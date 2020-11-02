#! /usr/bin/python3

import pymongo

# 连接数据库
myclient = pymongo.MongoClient("mongodb://localhost:27017/")

# 判断数据库是否存在
dblist = myclient.database_names()
if "lpr_demo" in dblist:
    print("数据库已存在")
# 创建/指定数据库，若没有查找到，则会自动创建
mydb = myclient["lpr_demo"]
# mydb.test: 下一步指定集合


class User(object):
    def __init__(self, username, password, role):
        self.username = username
        self.password = password
        self.role = role

    def save(self):
        # 创建用户
        user = {"username":self.username, "password":self.password, "role":self.role}
        coll = self.get_coll()
        # 判断username是否重复
        if coll.find({'username':self.username}).count() != 0 :
            # 不创建
            return "用户名已存在"
        id = coll.insert_one(user)
        print(id)
        # 创建成功
        return True
    
    def __str__(self):
        return "username:{},password:{},role:{}".format(self.username,self.password,self.role)

    @staticmethod
    def login(username,password):
        collection = User.get_coll()
        result = collection.find({'username':username, 'password':password})
        # for r in result:
        #     print(r)
        if result.count()==1:
            # 登录成功
            return result.clone()
        else:
            # 登录失败
            return False

    @staticmethod
    def get_coll():
        myclient = pymongo.MongoClient("localhost", 27017)
        db = myclient["lpr_demo"]
        users = db.user_collection
        return users
 
    @staticmethod
    def query_users():
        return User.get_coll().find()

import time
import datetime
class Record(object):
    def __init__(self, plate_number):
        self.plate_number = plate_number
        self.is_active = True
        self.cost = 0
        self.enter_time = "undefined"
        self.leave_time = "undefined"

    def car_enter(self):
        self.enter_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        record = {"plate_number":self.plate_number,"is_active":True,"enter_time":self.enter_time,
                "leave_time":"undefined","cost":0}
        coll = self.get_coll()
        id = coll.insert_one(record)
        print(id)
        
    def car_leave(self):
        self.is_active = False
        collection = self.get_coll()
        condition = {'plate_number':self.plate_number, 'is_active':True}
        record = collection.find_one(condition)
        self.enter_time = record['enter_time']
        # 活跃状态置为0
        record['is_active'] = False
        # 离开时间
        self.leave_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        record['leave_time'] = self.leave_time

        # 计算费用，每分钟0.1元
        enter = datetime.datetime.strptime(record['enter_time'], '%Y-%m-%d %H:%M:%S')
        leave = datetime.datetime.strptime(self.leave_time, '%Y-%m-%d %H:%M:%S')
        minutes_delta = int((leave-enter).seconds/60)+1
        cost = round(minutes_delta*0.1,1)
        self.cost = cost
        record['cost'] = cost

        # 更新
        result = collection.update(condition,record)
        print(result)

    def __str__(self):
        if self.is_active:
            return "{}：{}成功进场".format(self.plate_number,self.enter_time)
        else:
            return "{}: {}成功离场，费用为{}元".format(self.plate_number,self.leave_time,self.cost)

    @staticmethod
    def query_car(plate_number):
        collection = Record.get_coll()
        result = collection.find({'plate_number':plate_number, 'is_active':True})
        if result.count() == 1:
            return result
        else:
            return "该车不在停车场内"

    @staticmethod
    def query_car_records(plate_number):
        collection = Record.get_coll()
        result = collection.find({'plate_number':plate_number})
        if result.count() == 0:
            return False
        return result
    
    @staticmethod
    def is_active(plate_number):
        collection = Record.get_coll()
        result = collection.find({'plate_number':plate_number, 'is_active':True})
        if result.count() == 1:
            return True
        else:
            return False

    @staticmethod
    def get_coll():
        myclient = pymongo.MongoClient("localhost", 27017)
        db = myclient["lpr_demo"]
        records = db.record_collection
        return records

    @staticmethod
    def get_active_cars():
        collection = Record.get_coll()
        result = collection.find({"is_active":True})
        return result

    @staticmethod
    def get_active_car_number():
        result = Record.get_active_cars()
        return result.count()

class History(object):
    def __init__(self,his_type,msg):
        # type:enter or leave
        self.his_type = his_type
        self.msg = msg
    
    def save(self):
        his = {"his_type":self.his_type, "msg":self.msg}
        coll = self.get_coll()
        id = coll.insert_one(his)
        print(id)
    
    def __str__(self):
        return "{}:{}".format(self.his_type,self.msg)
    
    @staticmethod
    def get_coll():
        myclient = pymongo.MongoClient("localhost", 27017)
        db = myclient["lpr_demo"]
        history = db.history_collection
        return history

    @staticmethod
    def query_history():
        return History.get_coll().find()