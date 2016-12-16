
# coding: utf-8

# In[2]:



# In[2]:

import numpy as np
#import matplotlib.pyplot as plt
import math
#get_ipython().magic(u'matplotlib inline')

# Make sure that caffe is on the python path:
caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import fileinput

import os


# In[9]:

##test block


# In[ ]:

caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Net(sys.argv[1], #path to deploy prototxt,
                sys.argv[2], #path to learned model,
                caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(caffe_root + 'rank_scripts/query_128x128_market.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# In[ ]:

# In[5]:
image_data = []
label = []
attributes = []
#******************************
#Forward Pass for  Test Folder
#******************************
test_folder = '/users/gpu/agjayant1/Market-1501-v15.09.15/bounding_box_trainx2/'
# test_folder = '/users/gpu/agjayant1/Market-1501-v15.09.15/query/'
images_list = os.listdir(test_folder)

print "Setup Done. Starting Forward Pass for Test Images"

# In[6]:

num_images = len(images_list)
BatchSize = 100
j=0

while j < 5000:
    net = caffe.Net(sys.argv[1], # deploy prototxt,
                sys.argv[2], # model,
                caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]


    # set net to batch size
    net.blobs['data'].reshape(BatchSize,3,227,227)
    i = 0
    k = j
    while j < num_images and i < BatchSize:
        test_image = caffe.io.load_image(test_folder+ images_list[j])
        net.blobs['data'].data[i] = transformer.preprocess('data', test_image)
        i = i + 1
        j = j + 1

    out = net.forward()
    i=0
    while k < num_images and i < BatchSize:
        pic = Image.open(test_folder+images_list[k])
        A = np.array(pic,dtype='f4')
        A = A.transpose((2,0,1))
        image_data.append(A)
        a = out['fc8_a'][i]
        b =  sorted(range(len(a)), key=lambda x: a[x])[-20:]
        d = [0] * 106
        D= np.array(d)
        D[b] = 1
        attributes.append(D)
        label.append(int(item.split('_')[0]))
        i = i + 1
        k = k + 1


# In[7]:

print "**************************************************"
print "Forward Pass Completed for all Test Images: Part 1"
print "**************************************************"
label = np.array(label,dtype='f4')
attributes = np.array(attributes,dtype='f4')
h = h5py.File("/users/gpu/agjayant/Market-1501-v15.09.15/market_training1.h5", 'w' )
dset1 = h.create_dataset("data", data=image_data)
dset2 = h.create_dataset("attributes", data=attributes)
dset3 = h.create_dataset("label",  data=label)
h.close()

print "**************************************************"
print "Saved for all Images: Part 1"
print "**************************************************"


# In[ ]:

image_data = []
label = []
attributes = []
while j < 10000:
    net = caffe.Net(sys.argv[1], # deploy prototxt,
                sys.argv[2], # model,
                caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]


    # set net to batch size
    net.blobs['data'].reshape(BatchSize,3,227,227)
    i = 0
    k = j
    while j < num_images and i < BatchSize:
        test_image = caffe.io.load_image(test_folder+ images_list[j])
        net.blobs['data'].data[i] = transformer.preprocess('data', test_image)
        i = i + 1
        j = j + 1

    out = net.forward()
    i=0
    while k < num_images and i < BatchSize:
        pic = Image.open(test_folder+images_list[k])
        A = np.array(pic,dtype='f4')
        A = A.transpose((2,0,1))
        image_data.append(A)
        a = out['fc8_a'][i]
        b =  sorted(range(len(a)), key=lambda x: a[x])[-20:]
        d = [0] * 106
        D= np.array(d)
        D[b] = 1
        attributes.append(D)
        label.append(int(item.split('_')[0]))
        i = i + 1
        k = k + 1


# In[7]:

print "******************************************"
print "Forward Pass Completed for all Test Images : 2"
print "******************************************"
label = np.array(label,dtype='f4')
attributes = np.array(attributes,dtype='f4')
h = h5py.File("/users/gpu/agjayant/Market-1501-v15.09.15/market_training2.h5", 'w' )
dset1 = h.create_dataset("data", data=image_data)
dset2 = h.create_dataset("attributes", data=attributes)
dset3 = h.create_dataset("label",  data=label)
h.close()

print "**************************************************"
print "Saved for all Images: Part 2"
print "**************************************************"


# In[ ]:

image_data = []
label = []
attributes = []
while j < num_images:
    net = caffe.Net(sys.argv[1], # deploy prototxt,
                sys.argv[2], # model,
                caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]


    # set net to batch size
    net.blobs['data'].reshape(BatchSize,3,227,227)
    i = 0
    k = j
    while j < num_images and i < BatchSize:
        test_image = caffe.io.load_image(test_folder+ images_list[j])
        net.blobs['data'].data[i] = transformer.preprocess('data', test_image)
        i = i + 1
        j = j + 1

    out = net.forward()
    i=0
    while k < num_images and i < BatchSize:
        pic = Image.open(test_folder+images_list[k])
        A = np.array(pic,dtype='f4')
        A = A.transpose((2,0,1))
        image_data.append(A)
        a = out['fc8_a'][i]
        b =  sorted(range(len(a)), key=lambda x: a[x])[-20:]
        d = [0] * 106
        D= np.array(d)
        D[b] = 1
        attributes.append(D)
        label.append(int(item.split('_')[0]))
        i = i + 1
        k = k + 1


# In[7]:

print "******************************************"
print "Forward Pass Completed for all Test Images:3"
print "******************************************"
label = np.array(label,dtype='f4')
attributes = np.array(attributes,dtype='f4')
h = h5py.File("/users/gpu/agjayant/Market-1501-v15.09.15/market_training3.h5", 'w' )
dset1 = h.create_dataset("data", data=image_data)
dset2 = h.create_dataset("attributes", data=attributes)
dset3 = h.create_dataset("label",  data=label)
h.close()

print "**************************************************"
print "Saved for all Images: Part 3"
print "**************************************************"

