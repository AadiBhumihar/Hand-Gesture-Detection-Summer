#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: bhumihar
"""

from PIL import Image
import os

def ppm_to_jpeg(path):
    root = path    
    for path, subdirs, files in os.walk(root):
        for name in files:
            img = os.path.join(path, name);
            im = Image.open(img)
            im = im.convert('RGB')
            im.save(path+'/'+name.rsplit('.', 1)[0]+'.jpeg')
            os.remove(path+'/'+name)

train_path= '/home/bhumihar/Programming/Python/opencv/sample/project/Marcel-train'        
test_path= '/home/bhumihar/Programming/Python/opencv/sample/project/Marcel-test'
val_path = '/home/bhumihar/Programming/Python/opencv/sample/Project/Marcel-val'

#ppm_to_jpeg(train_path)
ppm_to_jpeg(val_path)