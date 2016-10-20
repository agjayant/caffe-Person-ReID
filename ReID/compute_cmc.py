
# coding: utf-8

# In[14]:

import sys
import fileinput
#import matplotlib
#import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
import numpy as np

data = []

for line in fileinput.input(sys.argv[1]):
    data.append(line[:-1].split(','))
    
#for item in data:
#   print item
    
#print data


# In[9]:

num_rank = len(data[0])-2
print num_rank


# In[10]:

Rank_cnt = [0]* num_rank

for item in data:
    query_id = int(item[0][0:4])
    
    for i in range(2,len(item)):
        match_id = int(item[i][0:4])
        
        if match_id == query_id :
            for j in range(i,len(item)):
                Rank_cnt[j-2] = Rank_cnt[j-2]+1
            break

print Rank_cnt


# In[11]:

x  = range(1,num_rank+1)
y = []

for rank in Rank_cnt:
    y.append(rank*100*1.0/len(data))
    
print y


# In[6]:

#plt.plot(x,y)
#plt.xticks(np.arange(1,num_rank+1,1))
#plt.savefig('temp.png')

