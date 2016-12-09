
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import math
# get_ipython().magic(u'matplotlib inline')
import Image
# Make sure that caffe is on the python path:
caffe_root = '/users/gpu/agjayant1/caffe-PersonReID/' # Expected to be in rank_scripts/

import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import sys
# source_folder = '/home/jayant/vision/PETA/'

import os
import h5py
import fileinput


# In[3]:

image_data = []


# In[4]:

#folder = '/home/jayant/vision/PETA/TownCentre/alex_size/'
folder = '/users/gpu/agjayant1/PETA/alex_size/'

Town_images_list = os.listdir(folder)

TC_images_list= []
for item in Town_images_list:
    TC_images_list.append((int(item.split('_')[0]),item))
TC_images_list.sort()


for item in TC_images_list:
    image= item[1]
    #print image
    pic = Image.open(folder+image)
    #A = caffe.io.load_image(folder+image)
    A = np.array(pic,dtype='f4')
    A = A.transpose((2,0,1))
    #B= np.array(A,dtype='f4')
    image_data.append(A)
#     image_data.append((image,A))


# In[5]:

label_data_town=[]
for line in fileinput.input('/users/gpu/agjayant1/PETA/TownCentre_atr.txt'):
    label_data_town.append(line[:-2].split(' '))


# In[6]:

#for item in label_data_GRID:
#    print item[1:107]
# check = 6
# print label_data_town[check]
# sum1=0
# for item in label_data_town[check][1:107]:
#     sum1=sum1+ int(item)
# print 'sum =',sum1
# i=1
# for item in label_data_town[check][1:107]:
#     if item =='1' :
#         print i
#     i=i+1


# In[7]:

label = []


# In[8]:

for item in TC_images_list:
    label.append(item[0])


# In[9]:

attributes = []


# In[10]:

i=0
for lab in label:
#     print lab
    if int(label_data_town[i][0]) == lab:
            attributes.append(label_data_town[i][1:107])
    else:
            i = i+1
            attributes.append(label_data_town[i][1:107])


# In[11]:

label = np.array(label,dtype='f4')
attributes = np.array(attributes,dtype='f4')


# In[ ]:

h = h5py.File("/users/gpu/agjayant1/PETA/townCentre_training.hdf5", 'w' )
dset1 = h.create_dataset("data", data=image_data)
dset2 = h.create_dataset("attributes", data=attributes)
dset3 = h.create_dataset("label",  data=label)
h.close()

