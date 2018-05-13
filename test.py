import tensorflow as tf
import keras
import network
from PIL import Image
import numpy as np
import os,csv

class testParams():
	"""
	定义测试部分的参数
	"""
	def __init__(self):
		#要测试的类别名称
		self.class_name=None
		#要写入的csv的路径
		self.answer_path=None
		#要读取的问题的csv路径
		self.question_path=None
		#要加载全部权重的路径，我默认的权重命名方式为class_name_best_weights.h5
		self.all_weights_path=None
		#类别名称对应类别数的字典
		self.name_map_classnum={"coat_length_labels":8,"collar_design_labels":5,"lapel_design_labels":5,
		"neck_design_labels":5,"neckline_design_labels":10,"pant_length_labels":6,"skirt_length_labels":6,
		"sleeve_length_labels":9}
		#使用的网络的名称
		self.net_name="Dense"
		#测试集图片的路径
		self.test_image_path="./ali_data/fashionAI_attributes_test_a_20180222/rank/Images/"

def get_models(test_params):
	'''
	获取所有的模型并加载参数
	'''
	if test_params.net_name=="Dense":
		model_dict={
	"coat_length_labels":network.Dense121_all(all_weights_path=test_params.all_weights_path+"coat_weights_best.h5",
		class_num=test_params.name_map_classnum["coat_length_labels"]),

	"collar_design_labels":network.Dense121_all(all_weights_path=test_params.all_weights_path+"collar_weights_best.h5",
		class_num=test_params.name_map_classnum["collar_design_labels"]),
	
	"lapel_design_labels":network.Dense121_all(all_weights_path=test_params.all_weights_path+"lapel_weights_best.h5",
		class_num=test_params.name_map_classnum["lapel_design_labels"]),
	
	"neck_design_labels":network.Dense121_all(all_weights_path=test_params.all_weights_path+"neck_weights_best.h5",
		class_num=test_params.name_map_classnum["neck_design_labels"]),
	
	"neckline_design_labels":network.Dense121_all(all_weights_path=test_params.all_weights_path+"neckline_weights_best.h5",
		class_num=test_params.name_map_classnum["neckline_design_labels"]),
	
	"pant_length_labels":network.Dense121_all(all_weights_path=test_params.all_weights_path+"pant_weights_best.h5",
		class_num=test_params.name_map_classnum["pant_length_labels"]),
	
	"skirt_length_labels":network.Dense121_all(all_weights_path=test_params.all_weights_path+"skirt_weights_best.h5",
		class_num=test_params.name_map_classnum["skirt_length_labels"]),
	
	"sleeve_length_labels":network.Dense121_all(all_weights_path=test_params.all_weights_path+"sleeve_weights_best.h5",
		class_num=test_params.name_map_classnum["sleeve_length_labels"])
	}
		return model_dict

def test_only_coat(test_params):
	'''
	仅仅测试coat_length
	'''
	model_binary=network.Dense121_all(all_weights_path=test_params.all_weights_path+"coat_binary_weights_best.h5",
		class_num=2)
	model=network.Dense121_all(all_weights_path=test_params.all_weights_path+"coat_weights_best.h5",
		class_num=test_params.name_map_classnum["coat_length_labels"])
	print("所有模型获取完毕...")
	with open(test_params.question_path,"r",encoding="utf-8") as file1:
		csvReader=csv.reader(file1)
		with open(test_params.answer_path,"a",encoding="utf-8") as file2:
			csvWriter=csv.writer(file2)
			for line in csvReader:
				l0=line[0]
				l1=line[1]
				image_path=os.path.join(test_params.test_image_path,l1,l0.split("/")[-1])
				img=Image.open(image_path,"r")
				img=img.resize((224,224))
				img=np.expand_dims(img,axis=0)
				img=img/255
				res=model_binary.predict(img,batch_size=1)[0]
				if res[0]>res[1]:
					res=[str(res[0]),'0.00','0.00','0.00','0.00','0.00','0.00','0.00']
				else:
					res2=model.predict(img,batch_size=1)[0]
					res=['0.00']+list(res)
				l2=";".join(res)
				content=[l0,l1,l2]
				csvWriter.writerow(content)
				print(count,l1)
				count+=1

def test_one(test_params):
	'''
	仅仅测试一个模型
	'''
	print("正在测试的是%s,共有%d"%(test_params.class_name,
		test_params.name_map_classnum[test_params.class_name]))
	
	if test_params.net_name=="Multimodel":
		model=network.Multimodel(class_num=test_params.name_map_classnum[test_params.class_name],
			all_weights_path=test_params.all_weights_path)
		print("模型加载完毕....")
	elif test_params.net_name=="Dense":
		model=network.Dense121_all(class_num=test_params.name_map_classnum[test_params.class_name],
			all_weights_path=test_params.all_weights_path)
		print("模型加载完毕....")
	else:
		model=network.VGG16_all(class_num=test_params.name_map_classnum[test_params.class_name],
			all_weights_path=test_params.all_weights_path)
		print("模型加载完毕....")
	count=0
	with open(test_params.question_path,"r",encoding="utf-8") as file1:
		csvReader=csv.reader(file1)
		with open(test_params.answer_path,"a",encoding="utf-8") as file2:
			csvWriter=csv.writer(file2)
			for line in csvReader:
				l0=line[0]
				l1=line[1]
				if l1!=test_params.class_name:
					continue
				image_path=os.path.join(test_params.test_image_path,l1,l0.split("/")[-1])
				img=Image.open(image_path,"r")
				img=img.resize((224,224))
				img=np.expand_dims(img,axis=0)
				img=img/255
				res=model.predict(img,batch_size=1)[0]
				res=list(res)
				for i in range(len(res)):
					res[i]=str(res[i])
				l2=";".join(res)
				content=[l0,l1,l2]
				csvWriter.writerow(content)
				print(count,l1)
				count+=1

def test(test_params):
	'''
	进行测试并且将结果写到csv中
	'''
	count=0
	model_dict=get_models(test_params)
	print("所有模型获取完毕...")
	with open(test_params.question_path,"r",encoding="utf-8") as file1:
		csvReader=csv.reader(file1)
		with open(test_params.answer_path,"w",encoding="utf-8") as file2:
			csvWriter=csv.writer(file2)

			for line in csvReader:
				l0=line[0]
				l1=line[1]
				image_path=os.path.join(test_params.test_image_path,l1,l0.split("/")[-1])
				img=Image.open(image_path,"r")
				img=img.resize((224,224))
				img=np.expand_dims(img,axis=0)
				img=img/255.0
				res=model_dict[line[1]].predict(img,batch_size=1)[0]
				res=list(res)
				for i in range(len(res)):
					res[i]=str(res[i])
				l2=";".join(res)
				content=[l0,l1,l2]
				csvWriter.writerow(content)
				print(count,l1)
				count+=1

if __name__=="__main__":
	params=testParams()
	params.net_name="Dense"
	#params.class_name="neckline_design_labels"
	params.test_image_path="./ali_data/z_rank/Images/"
	params.all_weights_path="./weights/"
	params.answer_path="./answer.csv"
	params.question_path="./question.csv"
	#print(params.name_map_classnum)
	test(params)
	