import glob
import sys
import os
#change the save path and audio path for own envs
sv_path="/media/caijb/data_drive/data/"
txt_path="./txt/"
audio_path="/media/caijb/data_drive/data/vctk/wav48_silence_trimmed/"

txt_folder_list=os.listdir(txt_path)
w_f=open("./vctk_text.txt","w",newline="\n")
for i in txt_folder_list:
    txt_list=os.listdir(txt_path+i)
    sid=i[1:]
    audio_name=audio_path+i+"/"
    for j in txt_list:
        name1=j[:-4]+"_mic1.flac"
        name2=j[:-4]+"_mic2.flac"
        a_path=audio_name+name1
        b_path=audio_name+name2
        f=open(txt_path+i+"/"+j,"r")
        data=f.read()
        f.close()
        a_data=a_path+"|"+sid+"|"+data
        b_data=b_path+"|"+sid+"|"+data
        w_f.write(a_data)
        w_f.write(b_data)
w_f.close()

