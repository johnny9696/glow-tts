""" from https://github.com/keithito/tacotron """
import os
import sys
from jamo import h2j, j2hcj
#change by the language type
from text.korean import symbols,last_letter



#symbol에서 가져온 걸 숫자로 변환 
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_last_letter_to_id = {s: i+len(symbols) for i, s in enumerate(last_letter)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
_id_to_last_letter = {i+len(symbols): s for i, s in enumerate(last_letter)}


def text_to_sequence(text,cleaner_names,dictionary=None):
	_text=text
	seq=[]
	seq=_word_to_id(_text)
	return seq

def _word_to_id(text):
	result=[]
	for i in range(len(text)):
		temp_text=numeric_cleaner(text[i])
		try:
			jamo_str=j2hcj(h2j(temp_text))
			if len(jamo_str)<3:
				result=result+_letters_to_id(jamo_str)
			else:
				result=result+_letters_to_id(jamo_str[:2])
				result=result+_last_letter_id(jamo_str[2])
		except:
			print(text[i])
	return result

def _letters_to_id(text):
	return [_symbol_to_id[s] for s in text]

def _last_letter_id(text):
	return [_last_letter_to_id[s] for s in text]

def numeric_cleaner(num):
	if num =="1":
		return "일"
	elif num =="2":
		return "이"
	elif num =="3":
		return "삼"
	elif num =="4":
		return "사"
	elif num =="5":
		return "오"
	elif num =="6":
		return "육"
	elif num =="7":
		return "칠"
	elif num =="8":
		return "팔"
	elif num =="9":
		return "구"
	elif num =="0":
		return "이"
	return num
if __name__=="__main__":
	print(text_to_sequence("동해물과 백두산이 말랐다.",None))