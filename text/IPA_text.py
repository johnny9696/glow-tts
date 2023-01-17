import os
import sys
from IPA_id import IPA_set

f_path='/media/caijb/data_drive/data/new_all.txt'
s_path='/media/caijb/data_drive/data/'
f=open(f_path,'r')
data=f.read()
f.close()
data=data.split('\n')
new_data=[]
ipa=[]
text_data=[]
for i in data:
    print(i)
    i_=i.split('|')
    lang=i_[2]
    text=i_[3]
    #print(text,lang)
    data=IPA_set(text,lang)
    print(data)
    for j in data:
        if j not in ipa:
            ipa.append(j)
    text_data.append(i+'\n')
    new_data.append(i+'\n')
    print(ipa)
new_data=''.join(new_data)
ipa=''.join(ipa)
print(new_data)

#f=open(s_path+'new_all.txt','w',encoding='UTF-8')
#f.write(new_data)
#f.close()  
f=open('./IPA.txt','w',encoding='UTF-8')
f.write(ipa)
f.close()

