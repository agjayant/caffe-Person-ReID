
# coding: utf-8

# In[ ]:

import data_augment as da
import sys
import os
import numpy as np
import random


# In[ ]:

#folder =  sys.argv[1]
folder =  '/home/jayant/vision/PETA/VIPeR/VIPeR_check/'
images_list = os.listdir(folder)
os.chdir(folder)

for img in images_list:
    image = img[0:-8]
    os.rename(img, image + '_01.jpg')
    src = image + '_01.jpg'
    da.mymirror(src, image + '_02.jpg')
    shift = 10*random.uniform(-1,1)
    da.translate(src, image + '_03.jpg',shift,shift)
    da.mymirror(image + '_03.jpg' , image + '_04.jpg')
    shift = 10*random.uniform(-1,1)
    da.translate(src, image + '_05.jpg',shift,shift)
    da.mymirror(image + '_05.jpg' , image + '_06.jpg')

