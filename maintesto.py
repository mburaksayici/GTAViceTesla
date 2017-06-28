# -*- coding: utf-8 -*-
from __future__ import print_function
import collect_data
import time
import numpy as np
import grabscreen
import tflearn 
import cv2
import directkeys
import getkeys

import numpy as np

import tensorflow as tf
import tflearn


input =np.load('slopedatanew.npy'), 
output =np.load('outdatanew.npy')
input = np.array(input).reshape(-1,320,240,1)  # .reshape(320,240,1000)
output = np.array(output)
a = np.array([])
for i in range(3000):
    a = np.append(a,output[i][0])
output = a.reshape(3000,9)
#Building network, taking dimensions into account
""" Hard to train, 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

network = input_data(shape=[None, 320,240,1])
conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
pool1_3_3 = local_response_normalization(pool1_3_3)
conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,3, activation='relu', name='conv2_3_3')
conv2_3_3 = local_response_normalization(conv2_3_3)
pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

# merge the inception_3a__
inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

#merge the inception_3b_*
inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')


inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')

inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')

inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')


inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')

inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')

inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')

inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')


inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')


inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')

inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')


inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')

pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
pool5_7_7 = dropout(pool5_7_7, 0.4)
loss = fully_connected(pool5_7_7, 9,activation='softmax')
network = regression(loss, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.005) # 0.001 idi.
model = tflearn.DNN(network, checkpoint_path='model_googlenet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(input, output, n_epoch=20, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=128, snapshot_step=200,
          snapshot_epoch=False, run_id='googlenet_oxflowers17')
"""
"""
Good

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 320,240,1])
dense1 = tflearn.fully_connected(input_layer, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
dense3 = tflearn.fully_connected(dropout2, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout3 = tflearn.dropout(dense3, 0.8)

dense4 = tflearn.fully_connected(dropout3, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout4 = tflearn.dropout(dense4, 0.8)


dense5 = tflearn.fully_connected(dropout4, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout5 = tflearn.dropout(dense5, 0.8)


dense6 = tflearn.fully_connected(dropout5, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout6  = tflearn.dropout(dense6, 0.8)

dense7 = tflearn.fully_connected(dropout6, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout7  = tflearn.dropout(dense7, 0.8)


dense8 = tflearn.fully_connected(dropout7, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout8  = tflearn.dropout(dense8, 0.8)


dense9 = tflearn.fully_connected(dropout8, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout9  = tflearn.dropout(dense9, 0.8)

dense10 = tflearn.fully_connected(dropout9, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout10  = tflearn.dropout(dense10, 0.8)

dense11 = tflearn.fully_connected(dropout10, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout11  = tflearn.dropout(dense11, 0.8)


softmax = tflearn.fully_connected(dropout11, 9, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(input,output, n_epoch=100, validation_set=0.1,
          show_metric=True, run_id="dense_model")
"""

"""  Failed
# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 320,240,1])
dense1 = tflearn.fully_connected(input_layer, 4096, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 4096, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)

softmax = tflearn.fully_connected(dropout2, 9, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.001, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(input,output, n_epoch=10, validation_set=0.1,
          show_metric=True, run_id="dense_model")
"""
input_layer = tflearn.input_data(shape=[None, 320,240,1])
dense1 = tflearn.fully_connected(input_layer, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.7)

dense2 = tflearn.fully_connected(dropout1, 2000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
dense3 = tflearn.fully_connected(dropout2, 1000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout3 = tflearn.dropout(dense3, 0.8)

dense4 = tflearn.fully_connected(dropout3, 1000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout4 = tflearn.dropout(dense4, 0.8)
"""

dense5 = tflearn.fully_connected(dropout4, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout5 = tflearn.dropout(dense5, 0.8)


dense6 = tflearn.fully_connected(dropout5, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout6  = tflearn.dropout(dense6, 0.8)

dense7 = tflearn.fully_connected(dropout6, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout7  = tflearn.dropout(dense7, 0.8)


dense8 = tflearn.fully_connected(dropout7, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout8  = tflearn.dropout(dense8, 0.8)


dense9 = tflearn.fully_connected(dropout8, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout9  = tflearn.dropout(dense9, 0.8)

dense10 = tflearn.fully_connected(dropout9, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout10  = tflearn.dropout(dense10, 0.8)

dense11 = tflearn.fully_connected(dropout10, 3000, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout11  = tflearn.dropout(dense11, 0.8)
"""
softmax = tflearn.fully_connected(dropout4, 9, activation='softmax')

adam = tflearn.Adam(learning_rate=0.005, beta1=0.99)
#sgd = tflearn.SGD(learning_rate=0.05, lr_decay=0.96, decay_step=10000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=adam, metric=top_k,
                         loss='categorical_crossentropy')


# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(input,output, n_epoch=165, validation_set=0.1,
          show_metric=True, run_id="dense_model")
# Regression using Adagrad with learning rate decay and Top-3 accuracy
# Training

input("Press enter to start Gta")
input("pres1")
input("pres2")
input("pres3")
def thatsover():
    for i in list(range(2))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    
    while (True):
        
        last_time = time.time()
        if not paused:
            # 640*480 windowed mode
            
            screen = grabscreen.grab_screen(region=(0, 40, 640, 480))

            screen, lines = collect_data.process_img(screen)
            screen = cv2.resize(screen, (320,240))
            # screen = np.array(screen).reshape(320,240)  #  screen.reshape(-1,320,240,1)
            cv2.imshow('window', screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            prediction = model.predict([screen.reshape(320,240,1)])[0]
            b = np.array([])
            print(a)
            for i in range(9):
                if prediction[i] <=0.4:
                    b = np.append(b,[0])
                    print("1")
                elif prediction[i] >= 0.4:
                    
                    b = np.append(b,[1])
                    print("0")
            print(b)
            b = b.tolist()
            if b == collect_data.w:
                directkeys.w()
                print("w")
                
            elif b == collect_data.a:
                directkeys.a()
                print("a")                
            elif b == collect_data.s:
                directkeys.s()
                print("s")
                
            elif b == collect_data.d:
                directkeys.d()
                print("d")
            elif b == collect_data.nk:
                directkeys.nokey()
                print("nokey")
                
            elif b == collect_data.sa:
                directkeys.sa()
                print("sa")
            elif b == collect_data.sd:
                directkeys.sd()
                print("sd")
                
            elif b == collect_data.wa:
                directkeys.wa()
                print("wa")
                
            elif b == collect_data.wd:
                directkeys.wd()
                print("wd")
            else:
                print("something wrong")
            
        keys = getkeys.key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
        print('Loop took {} seconds'.format(time.time() - last_time))
      
      
      
thatsover()



"""
# -*- coding: utf-8 -*-
import collect_data
import time
import directkeys
from getkeys import key_check
import numpy as np
import grabscreen
import tflearn 
import cv2

import numpy as np

import tensorflow as tf
import tflearn


input, output = np.load('slopedata.npy'), np.load('outdata.npy')
input = np.concatenate((input,[0]), axis=0)
input = np.array(input)
input = np.delete(input, (2501),axis=0)
input = input.reshape((2501,1)) 
output = np.array(output).reshape(2501,9)
#Building network, taking dimensions into account
input_layer = tflearn.input_data(shape=[None, 1])
hidden1 = tflearn.fully_connected(input_layer,1205,activation='ReLU', regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(hidden1,0.8)

hidden2 = tflearn.fully_connected(dropout1,1205,activation='ReLU', regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(hidden2,0.8)
softmax = tflearn.fully_connected(dropout2,9,activation='softmax')

# Regression with SGD
sgd = tflearn.SGD(learning_rate=0.01,lr_decay=0.96, decay_step=1000)
top_k=tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax,optimizer=sgd,metric=top_k,loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(input,output,n_epoch=1,show_metric=True, run_id='dense_model')

print("jej")




def thatsover():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while (True):
        
        last_time = time.time()
        if not paused:
            # 640*480 windowed mode
            
            screen = grabscreen.grab_screen(region=(0, 40, 640, 480))

            screen, lines = collect_data.process_img(screen)
            sums = collect_data.just_slope(lines)
            cv2.imshow('window', screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
            reordered = np.array([])
            print("nope till for lloopp")
            a = model.predict([[sums]])
            b = np.array([])
            print(a)
            for i in range(9):
                if a[0][i] <=0.4:
                    b = np.append(b,[0])
                    print("1")
                else:
                    
                    b = np.append(b,[1])
                    print("0")
            print(reordered)
            if reordered == collect_data.w:
                directkeys.w()
                
            elif reordered == collect_data.a:
                directkeys.a()
                
            elif reordered == collect_data.s:
                directkeys.s()
                
            elif reordered == collect_data.d:
                directkeys.d()
                
            elif reordered == collect_data.nk:
                directkeys.nokey()
                
            elif reordered == collect_data.sa:
                directkeys.sa()
                
            elif reordered == collect_data.sd:
                directkeys.sd()
                
            elif reordered == collect_data.wa:
                directkeys.wa()
                
            elif reordered == collect_data.wd:
                directkeys.wd()
                
            print(reordered)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
        print('Loop took {} seconds'.format(time.time() - last_time))
        
        
                cv2.imshow('window', screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            

"""