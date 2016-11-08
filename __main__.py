import os
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import cv2
import numpy as np 

import model


shape = (1000, 1000)
scale = 2

OUTPUT_DIR = '/home/ubuntu/data/game_of_life/'+str(shape[0])+'x'+str(shape[1])+'/'
try:
    os.makedirs(OUTPUT_DIR)
except:
    pass

with tf.Session() as session:
    initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)
    
    board = tf.placeholder(tf.int32, shape=shape, name='board')
    board_update = tf.py_func(model.update_board, [board], [tf.int32])
    
    initial_board_values = session.run(initial_board)
    board_value = session.run(board_update, feed_dict={board: initial_board_values})[0]
    
    for i in range(100):
        [board_value] = session.run(board_update,feed_dict={board:board_value})

        frame = 255 * np.reshape(board_value,shape)
        
        frame = cv2.resize(frame.astype(np.uint8),None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        fn_frame = OUTPUT_DIR + str(i) + '.png'
        cv2.imwrite(fn_frame,frame)
        print 'wrote',fn_frame

