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
		print(text,id)
		esng = ESpeakNG()
		esng.voice = id
		ipa_data= esng.g2p (text, ipa=2)
		return ipa_data

	def ipa2vec(self,data):
		result=[]
		for i in data:
			result.append(self.ipa.index(i))
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
	print(text_to_sequence("이상한 나라의 엘리스는 정말로 재밋지만 기괴한 이야기도 많이 들어 있는 거 같아\n",'ko'))
