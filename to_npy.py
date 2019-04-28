import os
from PIL import Image
import numpy as np
import cv2

root_dir='/media/ustb/Dataset/biovid/flow_dataset/original_flow/flow_train'
count=0
print('start')
for subdir1 in os.listdir(root_dir):
    if subdir1 != '6':
        subdir=root_dir+'/'+subdir1
        target_sub = '/media/ustb/Dataset/biovid/flow_dataset/original_flow/flow_train_npy/' +subdir1
        if not os.path.exists(target_sub):
            os.makedirs(target_sub)
        for video_path1 in os.listdir(subdir):
            count += 1
            print(count)
            image_npy=[]
            video_path=subdir+'/'+video_path1
            target=target_sub+'/'+video_path1+'.npy'
            files = os.listdir(video_path)
            files.sort(key=lambda x: int(x[:-4]))
            for frame1 in files:
                if frame1 != '138.jpg':
            	    frame=os.path.join(video_path,frame1)
            	    image_tmp=Image.open(frame)
            	    image_tmp = image_tmp.resize((32, 32))
            	    image_tmp=np.array(image_tmp)
            	    image_tmp=cv2.cvtColor(image_tmp,cv2.COLOR_BGR2RGB)
            	    image_npy.append(image_tmp)
            image_npy=np.array(image_npy)
            np.save(target,image_npy)
