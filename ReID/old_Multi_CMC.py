
# coding: utf-8

# In[1]:

import numpy as np
#import matplotlib.pyplot as plt
import math
#get_ipython().magic(u'matplotlib inline')

# Make sure that caffe is on the python path:
caffe_root = '../' 
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import fileinput
#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

import os


# In[2]:

#getDiff Implementation
def getDiff( vector1, vector2 ):
    sum=0

    for j in range(4096):
        diff= vector1[j]-vector2[j]
        diff=diff*diff
        sum=sum+diff
        
    return math.sqrt(sum)


# In[3]:

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(caffe_root +  'rank_scripts/multi/deploy_multinet_wa_1.prototxt',
                    caffe_root + 'rank_scripts/multi/models103/_iter_10000.caffemodel',
                    caffe.TEST)# input preprocessing: 'data' is the name of the input blob == net.inputs[0]


transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(caffe_root + 'rank_scripts/market_query_256x256.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# In[4]:

#Rank Vector Setup
num_rank = 21

#source of query folder
query_folder = sys.argv[1]
images_list = os.listdir(query_folder)


# In[12]:

images_features = {}

for image in images_list:
    net = caffe.Net(caffe_root +  'rank_scripts/multi/deploy_multinet_wa_1.prototxt',
                    caffe_root + 'rank_scripts/multi/models103/_iter_10000.caffemodel',
                    caffe.TEST)# input preprocessing: 'data' is the name of the input blob == net.inputs[0]

    # set net to batch size
    net.blobs['data'].reshape(1,3,227,227)

    query_image_path = query_folder + image
    query_image = caffe.io.load_image(query_image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', query_image)
    out = net.forward()
    images_features[image]=out['fc7'][0]


# In[18]:

#print images_features['0026_c2s1_001626_00.jpg']


# In[19]:

file1 = open('multi_i-LID_cmc_103_10000.txt','w')

images_set= []

for line in fileinput.input('probe_i-LID.txt'):
	images_set.append(line[:-1])

for image_q in images_set:
    
    vector_query = images_features[image_q]
    #plt.figure(figsize=(3,3))
    #plt.imshow(query_image)

    #Paired list to hold (diff,imagePath)

    Rank_list= []

    #print images_list

    for image in images_list:

        vector_new=images_features[image]
        diff = getDiff(vector_query, vector_new)

        #add the pair (diff,image) to the list
        Rank_list.append((diff,image))

    #sort the list based on diff
    Rank_list.sort()

    NewRankList = Rank_list[0:num_rank]

    file1.write(image_q)
    file1.write(',')
    
    for item in NewRankList:
        file1.write(item[1])
        
        if item is NewRankList[len(NewRankList)-1]:
            file1.write('\n')
        else:
            file1.write(',')
    
file1.close()

