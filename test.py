from numpy import genfromtxt
import os
import numpy as np

currentpath = os.getcwd()
training_data = genfromtxt(currentpath+'/data/train.csv',delimiter=',',skip_header=1, max_rows=100)
# split traning data into two parts: inputs and tag test_data
# print(training_data[:,1:])
# # print(training_data[:,0])
# x3 = zip(training_data[:,1:],training_data[:,0])
# x3 = list(x3)
# np.random.shuffle(x3)
# mini_batches = [
#     x3[k:k+10]
#     for k in range(0,len(x3),10)
# ]
# i=0
# for x,y in mini_batches[0]:
#     print(i)
#     print(x)
#     print(y)
#     i+=i
# mini_batches = [
#     x3[k:k+3]
#     for k in range(0,len(list(x3)),3)
# ]
# print(mini_batches)
a = np.random.randn(10,1)
print(np.argmax(a))
print(a)