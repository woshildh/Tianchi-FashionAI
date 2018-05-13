import network
import tensorflow as tf
import keras
import time
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint,CSVLogger,TensorBoard,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import os
import keras.backend.tensorflow_backend as KTF
import keras.backend as K
#设置使用GPU的比例
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.55
session=tf.Session(config=config)

KTF.set_session(session)

class TrainParams():
	"""
	definite train params
	"""
	def __init__(self):
		
		#实际训练中的参数
		self.batch_size=32
		self.epochs=300
		self.steps_per_epoch=500
		self.class_num=None
		self.target_size=(224,224)
		self.validation_steps=300

		#是否数据增强
		self.data_augment=False

		#载入权重模式
		self.is_start=True
		self.cnn_weights_path=None
		self.all_weights_path=None

		#记录训练信息的csv路径
		self.logger_path=None

		#保存权重的路径，不指定的话会存成时间戳的格式
		self.save_weights_path="./weights/%s.h5"%(time.time())

		#训练集路径
		self.train_data_path=None
		#验证集路径
		self.validation_data_path=None
		#初始学习率
		self.start_lr=0.01
		#初始化名字
		self.name=None

		#是否展示模型
		self.show=False

		#是否固定卷积层权重
		self.cnn_no_vary=False
		#学习方法
		self.learn_method="sgd"

		#定义网络名字
		self.net_name="Dense"
		#定义l1正则化的参数
		self.regular_param=0

def all_classes_metrics(y_true,y_pred):
	'''
	定义一个自己的测评函数，输出每个类别的准确率
	'''
	class_num=y_true.shape[1].value
	#将y_pred转成one_hot形式
	y2=K.argmax(y_pred,axis=1)
	y3=K.one_hot(y2,class_num)

	#对y_true和y_pred做一个与操作
	value_dict={}




def train(train_params):
	'''
	定义训练函数
	'''
	if train_params.net_name=="Dense":
		model=network.Dense121_all(train_params.cnn_weights_path,train_params.all_weights_path,
			train_params.class_num,train_params.regular_param,train_params.cnn_no_vary)
	elif train_params.net_name=="Vgg":
		model=network.VGG16_all(train_params.cnn_weights_path,train_params.all_weights_path,
			train_params.class_num,train_params.cnn_no_vary)
	else:
		model=network.Multimodel(train_params.cnn_weights_path,train_params.all_weights_path,
			train_params.class_num,train_params.regular_param,train_params.cnn_no_vary)

	if train_params.show:
		model.summary()

	sgd=SGD(lr=train_params.start_lr,momentum=0.9,nesterov=True)
	adam=Adam()

	if train_params.learn_method=="sgd":
		model.compile(optimizer=sgd,metrics=["acc"],loss="categorical_crossentropy")
	else:
		model.compile(optimizer=adam,metrics=["acc"],loss="categorical_crossentropy")

	#定义checkpoint
	checkpoint=ModelCheckpoint(filepath=train_params.save_weights_path,verbose=1,
		save_best_only=True,save_weights_only=True)
	logger=CSVLogger(filename=train_params.logger_path,append=True)
	board=TensorBoard(log_dir="./tfboard/%s/"%(train_params.name),write_graph=True,write_images=True)
	lr_reduce=ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=20, verbose=1, mode='max')

	#数据生成
	if train_params.data_augment==False:
		
		train_datagen=ImageDataGenerator(rescale=1.0/255)
		train_generator=train_datagen.flow_from_directory(train_params.train_data_path,
			batch_size=train_params.batch_size,target_size=train_params.target_size,class_mode="categorical")
		
		validation_datagene=ImageDataGenerator(rescale=1.0/255)
		validation_generator=validation_datagene.flow_from_directory(train_params.validation_data_path,
			batch_size=train_params.batch_size,target_size=train_params.target_size,class_mode="categorical")

	else:
		train_datagen=ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
		train_generator=train_datagen.flow_from_directory(train_params.train_data_path,
			batch_size=train_params.batch_size,target_size=train_params.target_size,class_mode="categorical")
		
		validation_datagene=ImageDataGenerator(rescale=1.0/255)
		validation_generator=validation_datagene.flow_from_directory(train_params.validation_data_path,
			batch_size=train_params.batch_size,target_size=train_params.target_size,class_mode="categorical")

	model.fit_generator(train_generator,steps_per_epoch=train_params.steps_per_epoch,epochs=train_params.epochs,
		validation_data=validation_generator,validation_steps=train_params.validation_steps,
		verbose=1,callbacks=[checkpoint,logger,board,lr_reduce])


if __name__=="__main__":
#first,definite your param
	params=TrainParams()
	params.net_name="Dense"
	params.epochs=10
	params.start_lr=0.0001
	params.data_augment=True
	params.batch_size=8
	params.learn_method="sgd"
	params.cnn_no_vary=False
	
	'''
	#训练sleeve
	params.class_num=9
	params.steps_per_epoch=2000
	params.name="sleeve_length_labels"
	params.logger_path="./logger/sleeve_dense.csv"
	params.all_weights_path="./weights/sleeve_weights_best.h5"
	params.save_weights_path="./weights/sleeve_weights_best.h5"
	params.train_data_path="./ali_data/fashionAI_attributes_train_20180222/base/Images/sleeve_length_labels"
	params.validation_data_path="./ali_data/fashionAI_attributes_validate_20180222/Images/sleeve_length_labels"
	params.regular_param=0.00005
	train(params)

	#训练neckline
	params.class_num=10
	params.steps_per_epoch=3000
	params.name="neckline_design_labels"
	params.logger_path="./logger/neckline_dense.csv"
	params.all_weights_path="./weights/neckline_weights_best.h5"
	params.save_weights_path="./weights/neckline_weights_best.h5"
	params.train_data_path="./ali_data/fashionAI_attributes_train_20180222/base/Images/neckline_design_labels"
	params.validation_data_path="./ali_data/fashionAI_attributes_validate_20180222/Images/neckline_design_labels"
	params.regular_param=0.00005
	train(params)
	'''

	#训练coat_length
	params.class_num=8
	params.steps_per_epoch=3000
	params.name="coat_length_labels"
	params.logger_path="./logger/coat_dense.csv"
	params.all_weights_path="./weights/coat_weights_best.h5"
	params.save_weights_path="./weights/coat_weights_best.h5"
	params.train_data_path="./ali_data/fashionAI_attributes_train_20180222/base/Images/coat_length_labels"
	params.validation_data_path="./ali_data/fashionAI_attributes_validate_20180222/Images/coat_length_labels"
	params.regular_param=0.00005
	train(params)

	#训练collar
	params.class_num=5
	params.steps_per_epoch=2000
	params.name="collar_design_labels"
	params.logger_path="./logger/collar_dense.csv"
	params.all_weights_path="./weights/collar_weights_best.h5"
	params.save_weights_path="./weights/collar_weights_best.h5"
	params.train_data_path="./ali_data/fashionAI_attributes_train_20180222/base/Images/collar_design_labels"
	params.validation_data_path="./ali_data/fashionAI_attributes_validate_20180222/Images/collar_design_labels"
	params.regular_param=0.00005
	train(params)
	
	#训练lapel
	params.class_num=5
	params.steps_per_epoch=2000
	params.name="lapel_design_labels"
	params.logger_path="./logger/lapel_dense.csv"
	params.all_weights_path="./weights/lapel_weights_best.h5"
	params.save_weights_path="./weights/lapel_weights_best.h5"
	params.train_data_path="./ali_data/fashionAI_attributes_train_20180222/base/Images/lapel_design_labels"
	params.validation_data_path="./ali_data/fashionAI_attributes_validate_20180222/Images/lapel_design_labels"
	params.regular_param=0.00005
	train(params)
	
	#训练neck
	params.class_num=5
	params.steps_per_epoch=1500
	params.name="neck_design_labels"
	params.logger_path="./logger/neck_dense.csv"
	params.all_weights_path="./weights/neck_weights_best.h5"
	params.save_weights_path="./weights/neck_weights_best.h5"
	params.train_data_path="./ali_data/fashionAI_attributes_train_20180222/base/Images/neck_design_labels"
	params.validation_data_path="./ali_data/fashionAI_attributes_validate_20180222/Images/neck_design_labels"
	params.regular_param=0.00005
	train(params)
	
	#训练pant
	params.class_num=6
	params.steps_per_epoch=1600
	params.name="pant_length_labels"
	params.logger_path="./logger/pant_length.csv"
	params.all_weights_path="./weights/pant_weights_best.h5"
	params.save_weights_path="./weights/pant_weights_best.h5"
	params.train_data_path="./ali_data/fashionAI_attributes_train_20180222/base/Images/pant_length_labels"
	params.validation_data_path="./ali_data/fashionAI_attributes_validate_20180222/Images/pant_length_labels"
	params.regular_param=0.00005
	train(params)
			
	#训练skirt
	params.class_num=6
	params.steps_per_epoch=1600
	params.name="skirt_length_labels"
	params.logger_path="./logger/skirt_dense.csv"
	params.all_weights_path="./weights/skirt_weights_best.h5"
	params.save_weights_path="./weights/skirt_weights_best.h5"
	params.train_data_path="./ali_data/fashionAI_attributes_train_20180222/base/Images/skirt_length_labels"
	params.validation_data_path="./ali_data/fashionAI_attributes_validate_20180222/Images/skirt_length_labels"
	params.regular_param=0.00005
	train(params)


