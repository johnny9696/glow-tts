from cgitb import text
import os
from re import T
import sys



_pad        = '_'
_punctuation = '!\'(),.:;?`´ '
_special = '-^'
with open('./text/IPA.txt', 'r',encoding='UTF-8') as f: 
    _letters = f.read()

letter_=[_pad] + list(_special) + list(_punctuation) + list(_letters)


def german_cleaner(text):
    text=text.lower()
    text=text.replace('ü','ue')
    text=text.replace('ö','oe')
    text=text.replace('ä','ae')
    text=text.replace('ß','ss')
    return text

def spanish_cleaner(text):
    text=text.lower()
    text=text.replace('ñ','ne')
    text=text.replace('í','i^')
    text=text.replace('á','a^')
    text=text.replace('é','e^')
    text=text.replace('ó','o^')
    text=text.replace('ú','u^')
    text=text.replace('ü','ue')
    text=text.replace('¡','i')
    return text

def english_cleaner(text):
    text=text.lower()
    return text

def franch_cleaner(text):
    text=text.lower()
    text=text.replace('æ','eu')
    text=text.replace('à','aee')
    text=text.replace('ù','uee')
    text=text.replace('è','eee')
    text=text.replace('â','a^')
    text=text.replace('ê','e^')
    text=text.replace('î','i^')
    text=text.replace('ô','o^')
    text=text.replace('û','u^')
    text=text.replace('ë','e')
    text=text.replace('ï','i')
    text=text.replace('ü','u')
    text=text.replace('ç','ch')
    text=text.replace('é','e')
    return text

def erase(text):
    for i in text:
        if i not in letter_:
            text=text.replace(i,'')
    return text

def text_to_sequence(text,id):
    if id=='es':
        text=spanish_cleaner(text)
    elif id=='du':
        text=german_cleaner(text)
    elif id=='fr':
        text=franch_cleaner(text)
    elif id=='english':
        text=english_cleaner(text)
    else:
        print('wrong langual ID')
        sys.exit()
    text=erase(text)
    data=[]
    for i in text:
        try:
            data.append(letter_.index(i))
        except:
            data.append(len(letter_)+1)

    return data

if __name__ =='__main__':
    print(text_to_sequence('Hello My name is ','english'))
