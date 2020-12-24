###Import
import h5py
import numpy as np
import os
from PIL import Image, ImageOps
import pandas as pd
from datetime import datetime 
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices [0],True)
import random

###Preprocessing
class preprocess:
    def __init__(self):
        pass
    

        
    @staticmethod
    def load_data(path):
        data = pickle.load(open(path, 'rb'),encoding = 'latin1')
        image_data = data['data']
        label_data = np.array(data['fine_labels'])
        p_label_data = np.array(data['coarse_labels'])
        return image_data,label_data,p_label_data
    
    @staticmethod
    def load_desp(path):
        desp = pickle.load(open(path, 'rb'),encoding = 'latin1')
        print(desp)

    @staticmethod
    def to_data(value,binary = False):
        image_list = []
        for index in range(value.shape[0]):
            image = Image.merge("RGB",(Image.fromarray(value[index][0]),Image.fromarray(value[index][1]),Image.fromarray(value[index][2])))
            if binary:
                image = image.convert('L')
            #print(np.array(image).shape)
            image_list.append(np.array(image))
        image = np.array(image_list)
        return image
    
    @staticmethod
    def reshape_to_image(value,channel = 3, size = 32):
        image = value.reshape(value.shape[0],channel,size,size)
        return image
    
    @staticmethod
    def reshape_to_raw(value):
        image = value.reshape(value.shape[0],1024)
        return image
    
    @staticmethod
    def normalization(value):
        return value.astype('float32') / 255
    
    @staticmethod
    def mean_pixel(value):
        return value - np.mean(value,axis=0)
      
    @staticmethod
    def seg(value):
        for index in range(value.shape[0]):
            for a in range(value[index].shape[0]):
                for b in range(value[index].shape[1]):
                    if value[index][a][b]>=np.mean(test):
                        value[index][a][b] = 0
                    else:
                        pass
                        #test[a][b] = 255
        return value
   
    @staticmethod
    def one_hot_encode(value,classes):
        label_list = []
        for label in value:
            sublist = [0 for ind in range(classes)]
            sublist[int(label)]=1
            label_list.append(sublist)
        return np.array(label_list).astype('float32')

    @staticmethod
    def one_hot_decode(value):
        label_list = []
        for label in value:
            ind_lab = np.where(label==1)[0][0]
            label_list.append(ind_lab)
        return np.array(label_list)
        
    @staticmethod
    def zstd(value):
        # Z-score standardization
        s = np.repeat(np.std(value,axis=1,ddof=1).reshape(value.shape[0],1),value.shape[1], axis =1)
        x = value - np.repeat(np.mean(value,axis=1).reshape(value.shape[0],1),value.shape[1], axis =1)
        return x/s


    @staticmethod
    def check_distribution(label):
        # check for label distribution
        dic = {}
        for a in label:
            try:
                dic[a]+=1
            except:
                dic[a] = 0
        return dic
    
    @staticmethod
    def train_test_split(data, ratio=0.7):
        total_number = data.shape[0]
        bound_number = int(total_number * ratio)
        return data[:bound_number],data[bound_number:]
    
    @staticmethod
    def k_fold(value,number=10):
        lists= []
        sub_list = []
        total_number = int(value.shape[0])
        each_bin = total_number/number
        for loop in range(number):
            target=value[int(each_bin*loop):int(each_bin*(loop+1))]
            train1=value[:int(each_bin*loop)]
            train2 = value[int(each_bin*(loop+1)):]
            train = np.append(train1, train2, 0)
            lists.append([train,target])
        return lists 
 
 ###Initlizer
 class HeNormal(initializers.VarianceScaling):
    # refer to tensorflow.keras.initializers.HeNormal for more detail
    def __init__(self, seed=None):
        super(HeNormal, self).__init__(
            scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)

    def get_config(self):
            return {'seed': self.seed}




###model
class RIR():
    def __init__(self,classes,epoch = 82, shape = (32,32,3), Lr = 0.001, optm='SGD',rotating = 0.2, dropout=0,regularizer = regularizers.l2(0.0001)):
        self.classes = classes
        self.epoch = epoch
        self.shape = shape
        self.dropout = dropout
        self.regularizer = regularizer
        self.rotating = 0.2
        self.current = 0
        if optm == 'SGD':
            opt = SGD(lr=Lr,momentum=0.9)
        self.opt = opt
        self.model = self.rirsetting(self.classes, self.shape, self.opt, self.dropout,self.regularizer,rotating = self.rotating)
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
    
    def Conv2d_BN(self, x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu')(x)
        x = BatchNormalization(axis=3)(x)
        return x
    
    def resnet_init(self, inpt, nb_filter, kernel_size, strides=(1, 1),padding='same'):
        x1a = self.Conv2d_BN(inpt[0], nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding=padding)
        x1b = self.Conv2d_BN(inpt[0], nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding=padding)
        x1c = self.Conv2d_BN(inpt[1], nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding=padding)
        x1d = self.Conv2d_BN(inpt[1], nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding=padding)
        shortcut = self.Conv2d_BN(inpt[0], nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        r = add([x1a, x1c, shortcut])
        t = add([x1b, x1d])
        return r,t        

    def identity_Block(self, inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
        x = self.resnet_init(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = self.resnet_init(x, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        if with_conv_shortcut:
            shortcut_r = self.Conv2d_BN(inpt[0], nb_filter=nb_filter,strides=(strides[0]*2, strides[1]*2),  kernel_size=kernel_size)
            shortcut_t = self.Conv2d_BN(inpt[1], nb_filter=nb_filter, strides=(strides[0]*2, strides[1]*2), kernel_size=kernel_size)
            r = add([x[0], shortcut_r])
            t = add([x[1], shortcut_t])
            return r,t
        else:
            r = add([x[0], inpt[0]])
            t = add([x[1], inpt[1]])
            return r,t

    def lr_callback(loop,opt):
        if loop >= 42:
            opt = SGD(lr=Lr*0.1,momentum=0.9)
        elif loop >= 62:
            opt = SGD(lr=(Lr*0.1)*0.1,momentum=0.9)
        elif loop >= 100:
            opt = SGD(lr=Lr*0.1*0.1*0.1*2,momentum=0.9)
        elif loop >= 130:
            opt = SGD(lr=Lr*0.1*0.1*0.1,momentum=0.9)
        elif loop >= 160:
            opt = SGD(lr=Lr*0.1*0.1*0.1*0.1,momentum=0.9)
        return opt
    
    def rirsetting(self,classes, shape, opt, dropout, regularizer,rotating = 0.2, callback = lr_callback):
        #xavier = initializers.GlorotNormal()
        #orth = initializers.Orthogonal()
        MSR = HeNormal()
        inpt = Input(shape=shape)
        data_augmentation = tf.keras.Sequential([
          layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
          layers.experimental.preprocessing.RandomRotation(rotating),
        ])
        x = data_augmentation(inpt)
        x = ZeroPadding2D((3, 3))(x)
        
        # conv1
        x = self.Conv2d_BN(x, nb_filter=96, kernel_size=(3, 3), padding='valid')

        x = self.identity_Block((x,x), nb_filter=96, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=96, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=192, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = self.identity_Block(x, nb_filter=192, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=192, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=384, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = self.identity_Block(x, nb_filter=384, kernel_size=(3, 3))
        x,_ = self.identity_Block(x, nb_filter=384, kernel_size=(3, 3))
        
        x = self.Conv2d_BN(x, nb_filter=100, kernel_size=(1,1))
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout)(x)
        x = Dense(classes, activation='softmax',kernel_initializer = MSR, kernel_regularizer=regularizer)(x)
        
        self.model = Model(inputs=inpt, outputs=x)
        opt = callback(self.current,opt)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return self.model
    

        
    def fitting(self, train_data, train_label,x_val,y_val,batch_size = 16):
        for loop in range(self.epoch):
            self.current = loop+1
            print('epoch '+str(loop+1))
            self.model.fit(train_data, train_label,validation_data=(x_val, y_val),batch_size=batch_size)
            self.acc.append(self.model.history.history['accuracy'])
            self.val_acc.append(self.model.history.history['val_accuracy'])
            self.loss.append(self.model.history.history['loss'])
            self.val_loss.append(self.model.history.history['val_loss'])

    def load_mod(self,name):
        self.model = load_model(name)
    
    def save_model(self,name):
        self.model.save(name)
        
        
    def plot(self):
        N = np.arange(1, self.epoch+1)
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(N, self.loss, label='train_loss')
        plt.plot(N, self.val_loss, label='val_loss')
        plt.plot(N, self.acc, label='train_acc')
        plt.plot(N, self.val_acc, label='val_acc')
        plt.title('Training loss and Accuracy')
        plt.xlabel('epoch #')
        plt.ylabel('loss/accuracy')
        plt.ylim((0, 10))
        plt.legend()
        plt.show()
        
    def prob_predict(self, test_data):
        result = self.model.predict(test_data)
        return result
    
    def predicting(self, test_data):
        label_list = []
        result = self.model.predict(test_data)
        for label in result:
            lab = [0 for a in range(result.shape[1])]
            lab[np.where(label==np.max(label))[0][0]]=1
            label_list.append(lab)
        return np.array(label_list)
    
    

    @staticmethod
    def confusion_matrix(result,test_label):
        # confusion matrix is a important tool to evaluate the performance of matrix
        # here we use dataframe to plot the matrix
        labels = []
        for label in test_label:
            if label not in labels:
                labels.append(label)
        labels = sorted(labels)
        matrix = pd.DataFrame(columns=labels,index=labels)
        shapes = test_label.shape[0]
        for ind in range(shapes):
            # calc prediction by count in loop
            if pd.isnull(matrix.loc[int(result[ind]),test_label[ind]]):
                matrix.loc[int(result[ind]),test_label[ind]] = 1
            else:
                matrix.loc[int(result[ind]),test_label[ind]] += 1
        return matrix.fillna(0)

    @staticmethod
    def classification_report(result,test_label):
        # classification_report is working for precision, recall, f1-score and support sample
        # here we focus on macro avg value since there is almost balanced data by checking tools we used before
        # but we also leave mirco avg and weighted avg here for different sets
        # we design a similar ploting with current main method such as sklearn
        
        labels = []
        for label in test_label:
            if label not in labels:
                labels.append(label)
        labels = sorted(labels)
        matrix = pd.DataFrame(columns=labels,index=labels)
        shapes = test_label.shape[0]
        for ind in range(shapes):
            # calc prediction by count in loop
            if pd.isnull(matrix.loc[int(result[ind]),test_label[ind]]):
                matrix.loc[int(result[ind]),test_label[ind]] = 1
            else:
                matrix.loc[int(result[ind]),test_label[ind]] += 1
        matrix = matrix.fillna(0)
        report = None
        indexs=['mirco avg', 'macro avg', 'weighted avg']
        indexs[0:0] = labels
        report = pd.DataFrame(columns=['precision','recall','f1-score','support'],
                              index=indexs)
        for indexs in range(len(labels)):
            # precision we use TP/(TP+FP)
            precision = matrix.loc[indexs,indexs]/matrix.iloc[indexs].sum()
            # recall we use TP/(TP+FN)
            recall = matrix.loc[indexs,indexs]/matrix[indexs].sum()
            report.loc[indexs,'support'] = matrix[indexs].sum()
            report.loc[indexs,'precision'] = round(precision,2)
            report.loc[indexs,'recall'] = round(recall,2)
            # f1 we use 2*(precision*recall)/(precision+recall)
            report.loc[indexs,'f1-score'] = round((2*precision*recall/(precision+recall)),2)

        ind_sum_TP = 0  
        ind_sum_FP = 0
        ind_sum_FN = 0
        for ind_sum in range(len(labels)):
            ind_sum_TP += matrix.loc[ind_sum,ind_sum]
            for ind_sub_sum in range(len(labels)):
                if ind_sub_sum != ind_sum:
                    ind_sum_FP += matrix.loc[ind_sum,ind_sub_sum]
                    ind_sum_FN += matrix.loc[ind_sub_sum,ind_sum]
        # mirco avg for precision and recall we use as following
        mir_P = ind_sum_TP/(ind_sum_TP+ind_sum_FP)
        mir_R = ind_sum_TP/(ind_sum_TP+ind_sum_FN)
        report.loc['mirco avg','precision'] = round(mir_P,2)
        report.loc['mirco avg','recall'] = round(mir_R,2)
        report.loc['mirco avg','f1-score'] = round(2*mir_P*mir_R/(mir_P+mir_R),2)
        report.loc['mirco avg','support'] = report.loc[labels,'support'].sum()

        # macro avg we get avg of previous count
        report.loc['macro avg','precision'] = round(report.loc[labels,'precision'].sum()/len(labels),2)
        report.loc['macro avg','recall'] = round(report.loc[labels,'recall'].sum()/len(labels),2)
        report.loc['macro avg','f1-score'] = round(report.loc[labels,'f1-score'].sum()/len(labels),2)
        report.loc['macro avg','support'] = report.loc[labels,'support'].sum()

        weighted_total_precision = 0
        weighted_total_recall = 0
        weighted_total_f1 = 0
        # we count weighted value based on distribution of samples
        for ind_sum in range(len(labels)):
            total_support = report.loc[labels,'support'].sum()
            weighted_total_precision += report.loc[ind_sum,'precision']*(report.loc[ind_sum,'support']/total_support)
            weighted_total_recall += report.loc[ind_sum,'recall']*(report.loc[ind_sum,'support']/total_support)
            weighted_total_f1 += report.loc[ind_sum,'f1-score']*(report.loc[ind_sum,'support']/total_support)

        report.loc['weighted avg','precision'] = round(weighted_total_precision,2)
        report.loc['weighted avg','recall'] = round(weighted_total_recall,2)
        report.loc['weighted avg','f1-score'] = round(weighted_total_f1,2)
        report.loc['weighted avg','support'] = report.loc[labels,'support'].sum()
        return report   


#Processing with data
# for ResNet
train_x,train_y,_ = preprocess.load_data('./cifar-100-python/train')
test_x,test_y,_ = preprocess.load_data('./cifar-100-python/test')
train_x = preprocess.reshape_to_image(train_x)
test_x = preprocess.reshape_to_image(test_x)
train_x = preprocess.to_data(train_x)
test_x = preprocess.to_data(test_x)
#print(image.shape)
train_x = preprocess.normalization(train_x)
test_x = preprocess.normalization(test_x)
#print(image[0])
train_x = preprocess.mean_pixel(train_x)
test_x = preprocess.mean_pixel(test_x)
classes = max(train_y)+1
train_y = preprocess.one_hot_encode(train_y,classes)
test_y = preprocess.one_hot_encode(test_y,classes)

# parameter RIR
model = RIR(classes,epoch=100, Lr = 0.1,rotating = 0.4, dropout=0.5)
model.fitting(train_x,train_y,test_x,test_y) 
model.plot()
result = model.predicting(test_x)
decode_result = preprocess.one_hot_decode(result)
decode_test_label = preprocess.one_hot_decode(test_y)
print(model.classification_report(decode_result,decode_test_label))

model.save_model('RIR')

#Import Model
model2 = ResNet(classes, Lr = 0.007, dropout=0.4)
model2.load_mod('RIR')
