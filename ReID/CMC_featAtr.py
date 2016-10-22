
# coding: utf-8

# In[4]:

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


# In[22]:

def getDiff( vector1, vector2 ):
    feat=0
    for j in range(4096):
        diff= vector1[0][j]-vector2[0][j]
        diff=diff*diff
        feat=feat+diff

    atr= 0
    for j in range(106):
        diff= vector1[1][j]-vector2[1][j]
        diff=diff*diff
        atr=atr+diff

    return math.sqrt(0.01*feat + atr)


# In[7]:

# In[4]:

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(caffe_root +  'rank_scripts/split7/deploy_split7_atr.prototxt',
                caffe_root + 'rank_scripts/split7/models500/_iter_30000.caffemodel',
                caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]
# net = caffe.Net(caffe_root +  'ReID/split7/deploy_split7_atr.prototxt',
                # caffe_root + 'ReID/split7/models500/_iter_30000.caffemodel',
                # caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(caffe_root + 'rank_scripts/query_128x128_market.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# In[16]:

# In[5]:

#source of query folder
query_folder = sys.argv[1]
# query_folder = caffe_root + 'ReID/check_images/'
images_list = os.listdir(query_folder)


# In[17]:

# In[6]:

num_images = len(images_list)
BatchSize = 100
# BatchSize = 9
j=0
images_features = {}
while j < num_images:
    net = caffe.Net(caffe_root +  'rank_scripts/split7/deploy_split7_atr.prototxt',
               caffe_root + 'rank_scripts/split7/models500/_iter_30000.caffemodel',
               caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]
    # net = caffe.Net(caffe_root +  'ReID/split7/deploy_split7_atr.prototxt',
                # caffe_root + 'ReID/split7/models500/_iter_30000.caffemodel',
                # caffe.TEST)# input preprocessing: 'data'is the name of the input blob == net.inputs[0]


    # set net to batch size
    net.blobs['data'].reshape(BatchSize,3,227,227)
    i = 0
    k = j
    while j < num_images and i < BatchSize:
        query_image = caffe.io.load_image(query_folder+ images_list[j])
        net.blobs['data'].data[i] = transformer.preprocess('data', query_image)
        i = i + 1
        j = j + 1

    out = net.forward()
    i=0
    while k < num_images and i < BatchSize:
        #images_features[images_list[k]]=out['fc7'][i]
        images_features[images_list[k]]=(out['fc7'][i],out['fc8_a'][i])
        i = i + 1
        k = k + 1


# In[18]:

num_rank = 6

#print images_features['0026_c2s1_001626_00.jpg']
images_set= []

for line in fileinput.input('Town_probeSet.txt'):
   images_set.append(line[:-1])

# for line in fileinput.input('Town_probe_check.txt'):
    # images_set.append(line[:-1])


# In[24]:

# In[7]:

file1 = open('town_cmc_500_30000.txt','w')
# file1 = open('town_cmc_500_30000_check.txt','w')


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

