#!/usr/bin/env python
# coding: utf-8

################################################### section 2 本堂课的实例 #####################################
#### problem 1 加载数据集
'''
函数功能：读取数据集中样本，并进行可视化
'''
import csv
import tensorflow as tf
import matplotlib.pyplot as plt

def readImgs():

    ################################ 读取csv文件 #################################

    '''
    要求：根据preparation中的train.csv文件路径，读取并解析csv文件
    提示：
        本考点涉及路径编写、csv文件打开、csv文件解析，请参考preparation
    '''
    csv_file_path = '''***'''                  # 编写csv数据文件的路径
    csv_file ='''***'''                        # 打开csv文件
    csv_data = '''***'''                       # 解析csv文件数据     第1处待补全
    list_read = list(csv_data)                                        # 将读取的数据转化为列表格式

    ############################# 读取图片，并保存 ##############################
    sample_1_info= list_read[1]                               # 取出第1个样本（sample1）的图像路径和标签

    '''
    要求：根据csv文件中的信息（样本图像路径和标签），调用tensorflow.io内函数，读取并解析第1个样本
    提示：
        （1）sample_1_path中储存的图片路径不完整，需要使用字符串拼接补全至根目录的路径。字符串拼接示例如下：
         a = 'Hello' 
         b = ' world'
         c = a + b
         通过此操作后，变量c的内容为：'Hello world'
        （2）本考点涉及文件路径、图片文件打开、图片数据解析
        （3）tf.io.decode_jpeg指定channels=1时为灰度图像
    '''
    sample_1_path = '''***'''            # 获取sample1的图片路径
    sample_1_label = '''***'''           # 获取sample1的标签
    sample_1_file = '''***'''            # 根据sample1的图片路径，打开图片文件
    sample_1_data = '''***'''            # 解析图片文件数据，获取灰度图像    第2处待补全

    ############################### 图像显示 #####################################
    img_show = tf.reshape(sample_1_data,(227,227))
    plt.imshow(img_show,cmap='gray')
    plt.show()
    print('该图的标签为：{}'.format(sample_1_label))

    return img_show,sample_1_label
# test
img,label = readImgs()

# expectation
# 该图的标签为0

################################################### section 3 初识梯度消失 #####################################
#### problem 2 梯度可视化
'''
函数功能：模型迭代训练过程的数据可视化
'''

from ignoreme.ignoreme.utils import my_load_data
from ignoreme.ignoreme.utils import my_normalization
from ignoreme.ignoreme.utils import my_build_model

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

def grad_show(data_file_name):
    
    data_x, data_y = my_load_data(data_file_name)
    data_x = my_normalization(data_x)
    
    model = Sequential([
        Dense(1, activation='sigmoid', input_shape=(2,)),         # layer1
        Dense(1, activation='sigmoid'),                           # layer2
        Dense(1, activation='sigmoid'),                           # layer3
        Dense(1, activation='sigmoid'),                           # layer4
        Dense(1, activation='sigmoid')])                          # layer5
    
    model.compile(optimizer=SGD(lr = 0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    

    '''
    要求：请参考preparation中TensorBoard的介绍，调用TensorBoard，构造tensorboard的回调函数，且：
        （1）每个epoch保存一次模型参数等数据
        （2）保存到'./model_tensorboard_1'路径下
    '''
    tb_call = '''***'''          # 第3处待补全

    model.fit(x=data_x,
              y=data_y,
              batch_size=50,
              epochs=0,
              verbose=0,
              callbacks=[tb_call])
    return model    

# test
data_file_name = './ignoreme/ignoreme/data.xls'
model = grad_show(data_file_name)
print('tensorboard配置成功')

# expectation
# tensorboard配置成功

################################################### section 4 梯度消失详解 #####################################
#### problem 3 计算w的更新量

'''
函数功能：计算layer1中参数w1的更新量
'''

from ignoreme.ignoreme.utils import get_composite_function
def cal_delta_w1():
    
    out_funtion = get_composite_function()      # 获取当前网络各个神经元的参数和输出
    
    x, y = out_funtion['x'], out_funtion['y']
    w1, a1 = out_funtion['w1'], out_funtion['a1']     # 获取layer1的相关变量
    w2, a2 = out_funtion['w2'], out_funtion['a2']     # 获取layer2的相关变量
    w3, a3 = out_funtion['w3'], out_funtion['a3']     # 获取layer3的相关变量
    w4, a4 = out_funtion['w4'], out_funtion['a4']     # 获取layer4的相关变量
    w5, a5 = out_funtion['w5'], out_funtion['a5']     # 获取layer5的相关变量
   
    '''
    要求：请按照preparation中的模型结构和w更新量的计算公式，推导出layer1中w11的更新量
    提示：请参考layer2的w更新量的计算
    '''
    delta_w11 = '''***'''       # 第4处待补全
    
    return delta_w11

# test
delta_w11 = cal_delta_w1()
print("delta_w11的输出为：{}".format(delta_w11.numpy()))

# expectation
# delta_w11的输出为：-0.000255573

################################################### section 5 梯度爆炸的解决 #####################################
#### problem 4 掌握batch normalization

'''
函数功能：对输入数据进行batch norm操作
输入：
    a：输入的数据
    gamma：参数
    beta：参数
输出：
    batch norm之后的数据
'''

import tensorflow as tf
def my_batch_norm(a, gamma, beta): 
    
    epsilon = 1.0e-8                          # 定义较小的值，防止分母为0
    
    '''
    要求：请调用tensorflow中相关函数，实现对输入数据a进行batch normalization操作
    '''
    a_mean = '''***'''         # 计算输入数据的均值
    a_var = '''***'''          # 计算输入数据的方差
    a_norm = '''***'''         # 计算输入数据的归一化
    a_out = '''***'''          # 计算batch norm后的输出   第5处待补全

    return a_out
#test
input_data = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
output_data = my_batch_norm(input_data, 0.01, 0.1)
print(output_data.numpy())

# expectation
# [0.08433301 0.08781456 0.09129612 0.09477767 0.09825923 0.10174078
#  0.10522233 0.10870388 0.11218544 0.11566699]

################################################### section 6 DNN的欢喜冤家 #####################################
#### problem 5 实现mini-batch的数据流
'''
函数功能：生成mini-batch对应的数据流，以随着训练进度，逐步将数据集加载并送到网络中
输入：
    root_path：训练集的根目录
    csv_name：样本数据保存的csv文件名称
    batchsize：每个mini-batch的样本数量
输出：
    一个batch的数据（x,y）
'''
import numpy as np
from ignoreme.ignoreme.utils import readImgsPath
from ignoreme.ignoreme.utils import data_shuffle
import tensorflow as tf

def gen_flow_data(root_path, csv_name, batchsize):
    
    x, y = readImgsPath(root_path, csv_name)  # 获取csv数据文件中的样本图片路径（x）和对应的标签（y）
    
    m = x.shape[0]                            # 样本集中样本的总数
    n = batchsize                             # 将batchsize的大小赋值给n
    num = 0                                   # 记录当前待处理的图片编号，将图片每batch size张为一组
    
    while True:  
        
        out_x = []                            # 定义当前这个mini-batch的输出列表out_x，以保存样本数据
        out_y = []                            # 定义当前这个mini-batch的输出列表out_y，以保存标签数据
        
        '''
        要求：请参考preparation中的mini batch生成的流程，为下面while循环添加合适的条件语句，使得函数每次生成batchsize个样本
        '''
        while False:       #  False 为第6处待补全
            if num == m:                # 若当前已处理到最后一张图片了，则混洗数据，并重新开始从0计数
                num = 0
                x, y = data_shuffle(x, y)     # 数据混洗，将原数据顺序打乱，重新排序            
            temp_img = tf.io.read_file(x[num])    
            temp_img = tf.io.decode_jpeg(temp_img,channels=1)     # 通过第i张图片的路径，读取该张图片
            out_x.append(temp_img)                                # 保存图像到样本数据列表中
            out_y.append(y[num])                                   # 保存标签到标签数据列表中
            num = num + 1

        out_x = tf.reshape(out_x, (-1, 227*227))   # 将x图像数据进行flatten操作，使每个样本为行向量
        out_y = tf.reshape(out_y,(-1,1))           # 将y标签整理维度为m行1列
        
        yield (out_x, out_y)

# test
import matplotlib.pyplot as plt

data_root_path = './ignoreme'
csv_name = 'train.csv'
batch_size = 2
train_gen = gen_flow_data(data_root_path, csv_name, batch_size)

mini_batch_x, mini_batch_y = next(train_gen)
print("mini-batch的数据维度为：\n x:{} \n y:{}".format(mini_batch_x.shape, mini_batch_y.shape))
train_gen.close()

# expectation
# mini-batch的数据维度为：
# x:(2, 51529)
# y:(2, 1)

#### problem 6 实现数据增强
'''
函数功能：对输入的一个样本数据进行数据增强
输入：
    img_path：样本图像的路径
    img_label：样本图像的标签
输出：
    数据增强之后的图像
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def data_augmentation(img_path,img_label):

    ################################### 读取图像数据 ################################
    img_data = tf.io.read_file(img_path)
    img_data = tf.io.decode_jpeg(img_data,channels=1)       # 通过图像路径，读取该张图片

    ################################ 图像数据增强规则 ##############################
    '''
    要求：请参考preparation中的ImageDataGenerator介绍，补全数据增强规则为：随机旋转度数范围为±10°
    '''
    da = ImageDataGenerator('''***''')     # 第7处待补全

    ##################################### 数据增强 #################################
    img_data = tf.reshape([img_data],(1,227,227,1))
    new_img_data, new_img_label = next(da.flow(img_data,img_label,batch_size=1))

    ##################################### 图像显示 #################################
    print('原图像：')
    img_data = tf.reshape(img_data,(227,227))
    plt.imshow(img_data,cmap='gray')
    plt.show()

    print('数据增强图像：')
    new_img_data = tf.reshape(new_img_data,(227,227))
    plt.imshow(new_img_data,cmap='gray')
    plt.show()

    return img_data, new_img_data

# test
import matplotlib.pyplot as plt

img_path = './ignoreme/dataSet/test_000001.jpg'
img_label = [0.0]
img_data, new_img_data = data_augmentation(img_path,img_label)

# expectation
# 两张图像


################################################### section 7 过拟合的解决 #####################################
#### problem 7 熟悉learning rate decay
'''
函数功能：定义learning rate随着epoch的变化规则
输入：
    epoch：迭代次数
    lr：当前学习率
输出：
    更新之后的下一个epoch的学习率
'''

from tensorflow.keras import callbacks

def lr_scheduler(epoch, lr):  

    '''
    要求：请按照preparation中的learning rate decay公式，实现lr的回调函数，且：decay_rate = 0.9
    提示：虽然每次回调时均会传入epoch，但是我们此处并没有用到
    '''
    lr =  '''***'''                 # 第8处待补全
    lr_epochs.append(lr)                                     # 保存新的学习率
    return lr                                                # 返回新的学习率

lr_decay = callbacks.LearningRateScheduler(lr_scheduler)     # learning rate decay
lr_epochs = []                                               # 用于保存每次迭代训练中的学习率
 
# test
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np

import matplotlib.pyplot as plt

epoch_num = 100                                   # 定义当前模型训练的迭代次数
model = Sequential([Dense(10)])                   # 搭建模型
model.compile(SGD(lr=0.1), loss='mse')            # 配置模型
model.fit(np.arange(50).reshape(2, 25), np.ones(2),         
          epochs=epoch_num, callbacks=[lr_decay], verbose=0)  # 模型训练

plt.plot(range(epoch_num), lr_epochs, '-o')       # 画图生成epoch与lr的关系曲线
plt.xlabel("epoch")
plt.ylabel("learning rate")
plt.show()

# expectation
# 一张图像

################################################### section 8 回顾和总结 #####################################
#### problem 8 DNN解决狗狗分类问题
'''
函数功能：DNN拟合dog数据集，包括加载数据、模型搭建、模型配置、回调函数配置、模型训练和模型评估
输入：
    train_csv：训练数据的csv文件名
    test_csv：测试数据的csv文件名
输出：
    模型
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def dnn_train(train_csv, test_csv):

    ############################### 分批装载数据 ##################################
    batchSize = 32
    root_path = './ignoreme'
    train_gen = gen_flow_data(root_path, train_csv, batchSize)   # 训练集进行数据增强
    test_gen = gen_flow_data(root_path, test_csv, batchSize)         # 测试集不进行数据增强

    ############################### 模型搭建  ###################################

    '''
    要求：
        构造由全连接层构成的5层神经网络，层与层之间添加batch-norm，其中layer1已给出，请补全layer2~layer5，且：
        （1）layer2，256个神经元，采用relu为激活函数，无input_shape；
        （2）layer3，128个神经元，采用relu为激活函数，无input_shape；
        （3）layer4，128个神经元，采用relu为激活函数，无input_shape；
        （4）layer5，1个神经元，采用sigmoid为激活函数，无input_shape；
    提示：可参考layer1的模型搭建方法
    '''
    model = Sequential([
        Dense(256, activation='relu', input_shape=(227*227,)), # layer1
        BatchNormalization(),                                  # batch norm
        '''***''',                         # layer2
        '''***''',                         # batch norm
        '''***''',                         # layer3
        '''***''',                         # batch norm
        '''***''',                         # layer4
        '''***''',                         # batch norm
        '''***'''])                        # layer5      # 第9处待补全

    ############################## 模型配置 #####################################
    model.compile(optimizer=SGD(lr = 0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    ############################# 回调函数配置 ##################################
    '''
    要求：
        调用TensorBoard、EarlyStopping和ReduceLROnPlateau，实现其在训练过程中的回调，且：
        （1）TensorBoard的log保存路径为'./model_tensorboard_2'，每1个epoch保存一次模型参数等数据；
        （2）EarlyStopping的监控指标为'loss'，若连续5个epoch的loss下降幅度小于0.0001，则停止训练；
        （3）ReduceLROnPlateau的监控指标为'loss'，若连续5个epoch的loss下降幅度小于0.0001，则lr = lr * 0.2；
    提示：以上各函数的使用方法，请参考preparation
    '''
    tb_call = '''***'''
    es_call = '''***'''
    lr_call = '''***'''     # 第10处待补全

    ################################ 模型训练 ####################################
    train_num = 370
    history = model.fit(train_gen,
                      steps_per_epoch=(train_num//batchSize),
                      epochs=0,
                      verbose=0,
                      callbacks=[tb_call,es_call,lr_call])

    train_gen.close()
    ################################ 模型评估 ####################################
    loss, acc = model.evaluate(test_gen, steps=1, verbose=0)
    test_gen.close()

    return model
# test
from ignoreme.ignoreme.utils import verify_model_dnn
train_csv = 'train.csv'
test_csv = 'test.csv'
model = dnn_train(train_csv, test_csv)
verify_model_dnn(model)

# expectation
# 模型已搭建成功
# 您的模型在dog数据集上的精度为90%左右