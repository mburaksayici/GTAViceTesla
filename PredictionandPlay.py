# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 22:40:56 2017

@author: Hp
"""

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
for i in range(1000):
    a = np.append(a,output[i][0])
output = a.reshape(1000,9)


input_layer = tflearn.input_data(shape=[None, 320,240,1])
dense1 = tflearn.fully_connected(input_layer, 512, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)

dense2 = tflearn.fully_connected(dropout1, 128, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)

softmax = tflearn.fully_connected(dropout2, 9, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.5, lr_decay=0.96, decay_step=10000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')


# Training

#
#model.load("modelim2.tfl.index")
#model.load("modelim2.tfl")
model = tflearn.DNN(net, tensorboard_verbose=0)


model.load("./modelim2.tfl")
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


