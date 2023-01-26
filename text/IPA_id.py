import email
import os
from re import T
import sys
from espeakng import ESpeakNG

f=open("./text/IPA.txt",'r',encoding='UTF-8')
ipa=f.read()
f.close()
symbols=ipa.split()


class text2vec():
	def __init__(self) -> None:
		try :
			f=open("./text/IPA.txt",'r',encoding='UTF-8')
		except:
			raise Exception("No IPA File")
		ipa=f.read()
		f.close()
		self.ipa=list(ipa)
		#print(self.ipa)

	def text2ipa(self,text,id):
		if id =='english':
			id='en-us'
		#print(text,id)
		esng = ESpeakNG()
		esng.voice = id
		ipa_data= esng.g2p (text, ipa=2)
		#print(ipa_data)
		return ipa_data

	def ipa2vec(self,data):
		result=[]
		for i in data:
			try :
				result.append(self.ipa.index(i))
			except :
				with open('./what.txt','w', encoding= 'UTF-8')  as f:
					f.write(i)
		return result

def text_to_sequence(text,id):
	c=text2vec()
	data=c.text2ipa(text,id)
	data=c.ipa2vec(data)
	
	return data

def IPA_set(text,id):
	c=text2vec()
	data=c.text2ipa(text,id)
	return data



if __name__=='__main__':
	text_path = '/media/caijb/data_drive/data/ge_test.txt'
	with open(text_path,'r',encoding='UTF-8') as f:
		data = f.read().split('\n')
	for i in data:
		i = i.split('|')
		_, _, lang, text = i[0], i[1], i[2], i[3]
		result = text_to_sequence(text,lang)



