import serial
import numpy as np
import zmq
import tensorflow as tf
import logging
from threading import Thread

port = "6000"
# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:%s" % port)
topicfilter = "presence"
socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)


feature_columns = [tf.contrib.layers.real_valued_column("", dimension=303)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, 
                                            hidden_units=[10, 20, 10],
                                            n_classes=2,
                                            model_dir="/tmp/V8_model")
j = 0
v = 0
presence = True
topicfilter = "image"
ser = serial.Serial('/dev/cu.usbmodem1421', 9600)
read_serial = []
set_1 = []
data = np.array([],dtype=np.int32,ndmin=2)
training_data = np.array([],dtype=np.int32,ndmin=2)
training_targets = np.array([])


def get_messages():
    while True:
        string = socket.recv()
        topic, message_data = string.split()
        presence = True;
        
def V8_mega_main():
    while True:
        read_string = ser.readline().decode('utf-8').replace("b'", '')[:-4].replace('.','').split(',')
        raw_data = [int(i) for i in read_string]
        set_1 = np.append(set_1,raw_data)
        if v == 5:
            data = np.array(set_1,dtype=np.int32,ndmin=2)
            print(data.shape)
            if j == 0:
                axis = 1
            else:
                axis = 0
            training_data = np.concatenate((training_data,data), axis=axis)
            training_data = training_data.astype(int)
            print("This is the training data: \n")
            print(training_data)
            set_1 = []
            v = 0
            j += 1
            if presence == True:
                training_targets = np.append(training_targets,[1])
                presence = False
                
            else:
                 training_targets = np.append(training_targets,[0])
            training_targets = training_targets.astype(int)
        if j == 10:
            #train next bach
            logging.getLogger().setLevel(logging.INFO)
            classifier.fit(x=training_data,y=training_targets,steps=200)
            training_data = np.array([])
            training_targets = np.array([])
            j = 0
        v += 1


t = Thread(target=get_messages)
t.start()
t2 = Thread(target=V8_mega_main)
t2.start()

        
