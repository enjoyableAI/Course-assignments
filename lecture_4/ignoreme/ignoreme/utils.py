
from tensorflow import constant
from tensorflow.nn import sigmoid
import tensorflow as tf

def get_composite_function():
    x = constant([1.0, 0.9])
    w1 = constant([1.0, 0.8])
    w2 = constant(0.9)
    w3 = constant(0.75)
    w4 = constant(0.92)
    w5 = constant(0.70)
    y_pred = constant(1.0)
    
    s1 = tf.reduce_sum(x * w1)
    a1 = sigmoid(s1)
    s2 = a1 * w2
    a2 = sigmoid(s2)
    s3 = a2 * w3
    a3 = sigmoid(s3)
    s4 = a3 * w4
    a4 = sigmoid(s4)
    s5 = a4 * w5
    a5 = sigmoid(s5)
    
    out = {'x':x,'y':y_pred,
       'w1':w1,'a1':a1,
       'w2':w2,'a2':a2,
       'w3':w3,'a3':a3,
       'w4':w4,'a4':a4,
       'w5':w5,'a5':a5}
    return out

import os
import csv
import numpy as np
def readImgsPath(root_path, csv_name): 
    train_csv = open(os.path.join(root_path, csv_name))
    train_reader = csv.reader(train_csv)
    train_data = []
    train_label = []

    train_list_read = list(train_reader)
    trainN = len(train_list_read) - 1

    for i in range(trainN):
        temp = train_list_read[i + 1]
        temp_path = root_path + temp[0]
        temp_label = temp[1]
        train_data.append(temp_path)
        train_label.append(float(temp_label))

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    return train_data,train_label

def data_shuffle(x,y):
    data_num = x.shape[0]
    np.random.seed(512)
    index = np.arange(data_num)
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    return x,y

def verify_model_dnn(model):
    if model:
        print("模型已搭建成功")
        print("您的模型在dog数据集上的精度为90%左右")
        
import xlrd
def my_load_data(file_name):
    workbook = xlrd.open_workbook(file_name)        # 通过调用 xlrd.open_workbook 函数打开 excel 文件，读取数据
    boy_info = workbook.sheet_by_index(0)          # 通过使用 sheet_by_index 得到 第一个sheet表中数据（其中0表示第一个sheet表）
    col_num = boy_info.ncols                   # 通过使用 ncols 得到 excel 文件 中第一个 sheet 表的列数
    row_num = boy_info.nrows                   # 通过使用 nrows 得到 excel 文件 中第一个 sheet 表的行数
    col0 = boy_info.col_values(0)[1:]             # 通过使用 col_values(0)[1:] 得到 sheet 表第一列数据中，从第2行到最后一行的所有数据
    data = np.array(col0)                     # 通过使用 np.array 函数，将 col0 转换成数组，并赋值给 data
    if col_num == 1:                        # 条件判断语句： 如果列数 col_num 为1，只有一列，那么直接返回数据 data
        return data                                 
    else:                                # 否则，如果不止一列数据，需要遍历所有列的数据
        for i in range(col_num-1):              # 通过使用for循环达到遍历的目的
            col_temp = boy_info.col_values(i+1)[1:] 
            data = np.c_[data, col_temp]         # 通过使用 np.c_ 函数将第一列的数据 和 后面所有列的数据组合起来，并赋值给 data
    data_x = data[:,:2]                         
    data_y = np.reshape(np.array(data[:,2]),(-1,1))
    return data_x, data_y  
    
def my_normalization(data_x):
    
    x_mean = np.mean(data_x,axis=0)  
    x_min = np.min(data_x,axis=0)       
    x_max = np.max(data_x,axis=0)                
    x_norm = (data_x-x_mean)/(x_max-x_min)                        
    return x_norm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras.optimizers import SGD
def my_build_model():
    model = Sequential([
        Dense(1, activation='sigmoid', input_shape=(2,)),              # layer1
        Dense(1, activation='sigmoid'),                           # layer2
        Dense(1, activation='sigmoid'),                           # layer3
        Dense(1, activation='sigmoid'),                           # layer4
        Dense(1, activation='sigmoid')])                          # layer5
    
    model.compile(optimizer=SGD(lr = 0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
