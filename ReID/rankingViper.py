
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
    for j in range(len(vector1)):
        diff= vector1[j]-vector2[j]
        diff=diff*diff
        sum=sum+diff

    return math.sqrt(sum)


# In[4]:

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


# In[5]:

#******************************
#Forward Pass for  Test Folder
#******************************
test_folder = '/users/gpu/agjayant1/PETA/VIPeR/test/'
# test_folder = '/users/gpu/agjayant1/Market-1501-v15.09.15/query/'
images_list = os.listdir(test_folder)

print "Setup Done. Starting Forward Pass for Test Images"

# In[6]:

num_images = len(images_list)
BatchSize = 100
j=0
images_features = {}
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
        images_features[images_list[k]]=out['fc7'][i]
        # images_features[images_list[k]]=out['featAtr'][i]
        i = i + 1
        k = k + 1


# In[7]:

print "******************************************"
print "Forward Pass Completed for all Test Images"
print "******************************************"
#******************************
#Forward Pass for  Query Folder
#******************************
query_folder = '/users/gpu/agjayant1/PETA/VIPeR/query/'
# images_set= []

images_set = os.listdir(query_folder)
# for line in fileinput.input('queryMarketSingle.txt'):
    # images_set.append(line[:-1])


print "Setup Done. Starting Forward Pass for Query Images"

# In[6]:

num_images = len(images_set)
j=0
while j < num_images:
    net = caffe.Net(sys.argv[1], # deploy prototxt,
                sys.argv[2], # model,
                caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]


    # set net to batch size
    net.blobs['data'].reshape(BatchSize,3,227,227)
    i = 0
    k = j
    while j < num_images and i < BatchSize:
        query_image = caffe.io.load_image(query_folder+ images_set[j])
        net.blobs['data'].data[i] = transformer.preprocess('data', query_image)
        i = i + 1
        j = j + 1

    out = net.forward()
    i=0
    while k < num_images and i < BatchSize:
        images_features[images_set[k]]=out['fc7'][i]
        # images_features[images_set[k]]=out['featAtr'][i]
        i = i + 1
        k = k + 1


print "******************************************"
print "Forward Pass Completed for all Query Images"
print "******************************************"

num_rank = 6

file1 = open(sys.argv[3],'w')

ProbeLength = len(images_set)
ProbeCount = 1
for image_q in images_set:

    vector_query = images_features[image_q]

    #Paired list to hold (diff,imagePath)

    Rank_list= []

    minDis = ( getDiff(vector_query , images_features[images_list[0]]), images_list[0])
    minDisNext = ( getDiff(vector_query , images_features[images_list[0]]), images_list[0])
    #print images_list

    for image in images_list:

        vector_new=images_features[image]
        diff = getDiff(vector_query, vector_new)

        #add the pair (diff,image) to the list
        # Rank_list.append((diff,image))
        if diff < minDis[0]:
            minDisNext = minDis
            minDis = (diff, image)
        elif diff < minDisNext[0]:
            minDisNext = (diff, image)

        #sort the list based on diff
    # Rank_list.sort()
    Rank_list.append(minDis)
    Rank_list.append(minDisNext)

        #remove the last element if more than 'num_rank'
        #if len(Rank_list) > num_rank :
         #   Rank_list.remove(Rank_list[len(Rank_list)-1])
    # NewRankList = Rank_list[0:num_rank]
    NewRankList = Rank_list

    file1.write(image_q)
    file1.write(',')

    for item in NewRankList:
        file1.write(item[1])

        if item is NewRankList[len(NewRankList)-1]:
            file1.write('\n')
        else:
            file1.write(',')
    print "RankList Written for Image: ",ProbeCount,'/',ProbeLength
    ProbeCount = ProbeCount + 1

file1.close()

