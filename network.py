import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers.core import Flatten,Dense,Dropout
from keras.layers.convolutional import Conv2D,ZeroPadding2D
from keras.layers.pooling import MaxPooling2D,GlobalMaxPooling2D
from keras.layers import Input
from keras.activations import relu
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import DenseNet121,Xception
from keras.utils import plot_model
import os

def Multimodel(cnn_weights_path=None,all_weights_path=None,class_num=5,regular_param=0
	,cnn_no_vary=False):
	'''
	获取densent121,xinception并联的网络
	此处的cnn_weights_path是个列表是densenet和xception的卷积部分的权值
	'''
	dir=os.getcwd()
	input_layer=Input(shape=(224,224,3))
	
	dense=DenseNet121(include_top=False,weights=None,input_tensor=input_layer,
		input_shape=(224,224,3))
	xception=Xception(include_top=False,weights=None,input_tensor=input_layer,
		input_shape=(224,224,3))
	#res=ResNet50(include_top=False,weights=None,input_shape=(224,224,3))

	if cnn_no_vary:
		for i,layer in  enumerate(dense.layers):
			dense.layers[i].trainable=False
		for i,layer in enumerate(xception.layers):
			xception.layers[i].trainable=False
		#for i,layer in enumerate(res.layers):
		#	res.layers[i].trainable=False
	if cnn_weights_path!=None:
		dense.load_weights(cnn_weights_path[0])
		xception.load_weights(cnn_weights_path[1])

	#print(dense.shape,xception.shape)
	#对dense_121和xception进行全局最大池化
	top1_model=GlobalMaxPooling2D(input_shape=(7,7,1024),data_format='channels_last')(dense.output)
	top2_model=GlobalMaxPooling2D(input_shape=(7,7,1024),data_format='channels_last')(xception.output)
	#top3_model=GlobalMaxPool2D(input_shape=res.output_shape)(res.outputs[0])
	
	print(top1_model.shape,top2_model.shape)
	#把top1_model和top2_model连接起来
	t=keras.layers.Concatenate(axis=1)([top1_model,top2_model])
	#第一个全连接层
	top_model=Dense(units=512,activation="relu",
		kernel_regularizer=keras.regularizers.l1(l=regular_param))(t)
	top_model=Dropout(rate=0.5)(top_model)
	top_model=Dense(units=class_num,activation="softmax",
		kernel_regularizer=keras.regularizers.l1(l=regular_param))(top_model)
	
	model=Model(inputs=input_layer,outputs=top_model)

	#加载全部的参数
	if all_weights_path:
		model.load_weights(all_weights_path)
	return model

def Dense121(weights_path=None,cnn_no_vary=False):
	model=DenseNet121(include_top=False,weights=None,input_shape=(224,224,3))
	if cnn_no_vary:
		for i,layer in enumerate(model.layers):
			model.layers[i].trainable=False

	if weights_path!=None:
		model.load_weights(weights_path)

	return model

def Dense121_all(cnn_weights_path=None,all_weights_path=None,class_num=5,regular_param=0,
	cnn_no_vary=False):
	'''
	获取整个的DenseNet121
	'''
	cnn_model=Dense121(cnn_weights_path,cnn_no_vary)
	
	'''
	top_model=Flatten(input_shape=cnn_model.output_shape[1:])(cnn_model.outputs[0])	
	top_model=Dense(units=512,activation="relu")(top_model)
	top_model=Dropout(rate=0.5)(top_model)
	top_model=Dense(units=512,activation="relu")(top_model)
	top_model=Dropout(rate=0.5)(top_model)
	top_model=Dense(units=class_num,activation="softmax")(top_model)
	'''
	
	top_model=GlobalMaxPooling2D(input_shape=cnn_model.output_shape)(cnn_model.outputs[0])
	top_model=Dense(units=512,activation="relu",kernel_regularizer=
		keras.regularizers.l1(l=regular_param))(top_model)
	top_model=Dropout(0.5)(top_model)
	top_model=Dense(units=class_num,activation="softmax",kernel_regularizer=
		keras.regularizers.l1(l=regular_param))(top_model)
	model=Model(inputs=cnn_model.inputs,outputs=top_model)
	if all_weights_path:
		model.load_weights(all_weights_path)
	return model


def VGG_16(weights_path=None,cnn_no_vary=False):
	'''
	获取VGG16的卷积层部分
	'''
	model=Sequential()

	model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
	model.add(Conv2D(64,(3,3),activation="relu"))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64,(3,3),activation="relu"))
	model.add(MaxPooling2D((2,2),strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128,(3,3),activation="relu"))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128,(3,3),activation="relu"))
	model.add(MaxPooling2D((2,2),strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256,(3,3),activation="relu"))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256,(3,3),activation="relu"))
	model.add(ZeroPadding2D((1,1)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256,(3,3),activation="relu"))
	model.add(MaxPooling2D((2,2),strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512,(3,3),activation="relu"))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512,(3,3),activation="relu"))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512,(3,3),activation="relu"))
	model.add(MaxPooling2D((2,2),strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512,(3,3),activation="relu"))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512,(3,3),activation="relu"))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512,(3,3),activation="relu"))
	model.add(MaxPooling2D((2,2),strides=(2,2)))
	'''
	model.add(Flatten())
	model.add(Dense(4096,activation="relu"))
	model.add(Dropout(rate=0.5))
	model.add(Dense(4096,activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(2,activation="softmax"))
	'''
	if weights_path:
		model.load_weights(weights_path)
	if cnn_no_vary:
		for layer in layers:
			layer.trainable=False
	return model

def VGG16_all(cnn_weights_path=None,all_weights_path=None,class_num=5,cnn_no_vary=False):
	'''
	获取VGG16的整个网络
	'''
	model=VGG_16(cnn_weights_path,cnn_no_vary)
	top_model=Sequential()
	top_model.add(Flatten(input_shape=model.output_shape[1:]))
	top_model.add(Dense(512,activation="relu"))
	top_model.add(Dropout(rate=0.5))
	top_model.add(Dense(512,activation="relu"))
	top_model.add(Dropout(rate=0.5))
	top_model.add(Dense(class_num,activation="softmax"))
	model.add(top_model)

	if all_weights_path:
		model.load_weights(all_weights_path)

	return model


if __name__=="__main__":
	weights_path=["./densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
	"./xception_weights_tf_dim_ordering_tf_kernels_notop.h5"]
	model=Multimodel(cnn_weights_path=weights_path,class_num=6)
	plot_model(model,to_file="./model.png")

