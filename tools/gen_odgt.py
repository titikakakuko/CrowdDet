import os
import sys
import math
import argparse

import numpy as np
import glob, cv2, torch

# img path
img_names = sorted(glob.glob('C:/Users/yinuowang3/Downloads/FOV02/20190416/19/*.jpg'))
filePath = 'C:/Users/yinuowang3/Downloads/FOV02/20190416/19/'

# for i,j,k in os.walk(filePath):
#     for img_name in k:
#         print(img_name)
#         break

path_list = os.listdir(filePath)
print(path_list[0])

with open('odgt.txt','a') as f:
    for file_name in path_list:
        f.write('{"ID": "' + file_name.split(".")[0] + '", "gtboxes": []}\n')
        # print(file_name)
f.close()
