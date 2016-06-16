
# coding: utf-8

# In[8]:

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


# In[9]:

#getDiff Implementation
def getDiff( vector1, vector2 ):
    sum=0
    for i in range(50):
        for j in range(1024):
            diff= vector1[i][j]-vector2[i][j]
            diff=diff*diff
            sum=sum+diff
        
    return math.sqrt(sum)


# In[10]:

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(caffe_root +  'examples/_temp/unsup_net_deploy.prototxt',
                    caffe_root + 'rank_scripts/models2/_iter_100.caffemodel',
                    caffe.TEST)# input preprocessing: 'data' is the name of the input blob == net.inputs[0]

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'rank_scripts/market_train_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# In[12]:

#Rank Vector Setup
num_rank = 6

#source of query folder
query_folder = sys.argv[1]
images_list = os.listdir(query_folder)


# In[18]:

file1 = open('market_cmc.txt','w')

with open(sys.argv[2]) as fp:	
	for line in fp:
	    net = caffe.Net(caffe_root +  'examples/_temp/unsup_net_deploy.prototxt',
	                    caffe_root + 'rank_scripts/models2/_iter_100.caffemodel',
	                    caffe.TEST)# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
	
	    # set net to batch size of 100
	    net.blobs['data'].reshape(100,3,64,64)

	    image_q = line[:-1]	
	    query_image = caffe.io.load_image( query_folder + image_q )
	    net.blobs['data'].data[...] = transformer.preprocess('data', query_image)
	    out = net.forward()
	    vector_query = out['fc7']
	    #plt.figure(figsize=(3,3))
	    #plt.imshow(query_image)
	
	    #Paired list to hold (diff,imagePath)
	
	    Rank_list= []
	
	    #print images_list
	
	    for image in images_list:
	        new_net = caffe.Net(caffe_root +  'examples/_temp/unsup_net_deploy.prototxt',
	                    caffe_root + 'rank_scripts/models2/_iter_100.caffemodel',
	                    caffe.TEST)
	        new_net.blobs['data'].reshape(100,3,64,64)
	        new_net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(query_folder + image))
	        output=new_net.forward()
	        vector_new=output['fc7']
	        diff = getDiff(vector_query, vector_new)
	
	        #add the pair (diff,image) to the list
	        Rank_list.append((diff,image))
	
	        #sort the list based on diff
	        Rank_list.sort()
	
	        #remove the last element if more than 'num_rank'
	        if len(Rank_list) > num_rank :
	            Rank_list.remove(Rank_list[len(Rank_list)-1])
	
	    file1.write(image_q)
	    file1.write(',')
	    
	    for item in Rank_list:
	        file1.write(item[1])
	        
	        if item is Rank_list[len(Rank_list)-1]:
	            file1.write('\n')
	        else:
	            file1.write(',')
	    
file1.close()
