
# coding: utf-8

# In[2]:

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


# In[3]:

#getDiff Implementation
def getDiff( vector1, vector2 ):
    sum=0
    for j in range(4096):
        diff= vector1[j]-vector2[j]
        diff=diff*diff
        sum=sum+diff
        
    return math.sqrt(sum)


# In[4]:

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(caffe_root +  'rank_scripts/multi/deploy_multinet.prototxt',
                caffe_root + 'rank_scripts/models103/_iter_500.caffemodel',
                caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(caffe_root + 'rank_scripts/query_128x128_market.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# In[5]:

#Rank Vector Setup
num_rank = 6

#source of query folder
query_folder = sys.argv[1]
images_list = os.listdir(query_folder)


# In[6]:

num_images = len(images_list)
BatchSize = 50
j=0
images_features = {}
while j < num_images:
    net = caffe.Net(caffe_root +  'rank_scripts/multi/deploy_multinet.prototxt',
                    caffe_root + 'rank_scripts/models103/_iter_500.caffemodel',
                    caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]


    # set net to batch size
    net.blobs['data'].reshape(BatchSize,3,227,227)
    i = 0 
    k = j
    while j < num_images and i < BatchSize:
        query_image_path = 'rank_scripts/check_images/' + images_list[j]
        query_image = caffe.io.load_image(caffe_root + query_image_path)
        net.blobs['data'].data[i] = transformer.preprocess('data', query_image)
        i = i + 1 
        j = j + 1 
            
    out = net.forward()
    i=0
    while k < num_images and i < BatchSize:
        images_features[images_list[k]]=out['fc7'][i]
        i = i + 1
        k = k + 1


# In[7]:

images_features_1 = {}

for image in images_list:
    net = caffe.Net(caffe_root +  'rank_scripts/multi/deploy_multinet.prototxt',
                    caffe_root + 'rank_scripts/models103/_iter_500.caffemodel',
                    caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]

    # set net to batch size
    net.blobs['data'].reshape(50,3,227,227)

    query_image_path = 'rank_scripts/check_images/' + image
    query_image = caffe.io.load_image(caffe_root + query_image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', query_image)
    out = net.forward()
    images_features_1[image]=out['fc7'][0]


# In[10]:


# In[6]:

#print images_features['0026_c2s1_001626_00.jpg']
images_set= []
  
for line in fileinput.input('query_set.txt'):
    images_set.append(line[:-1])


# In[7]:

file1 = open('market_cmc_38300.txt','w')

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

        #remove the last element if more than 'num_rank'
        #if len(Rank_list) > num_rank :
         #   Rank_list.remove(Rank_list[len(Rank_list)-1])
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

