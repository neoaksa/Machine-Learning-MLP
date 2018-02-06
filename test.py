from numpy import genfromtxt
import os
import numpy as np


def normalizationX(x):
    return x / 255

currentpath = os.getcwd()
training_file = genfromtxt(currentpath+'/data/train.csv',delimiter=',',skip_header=1, max_rows=100)
training_input = normalizationX(training_file[0::2, 1:])  # normalization data
# # split traning data into two parts: inputs and tag test_data
# # print(training_data[:,1:])
# # # print(training_data[:,0])
# # x3 = zip(training_data[:,1:],training_data[:,0])
# # x3 = list(x3)
# # np.random.shuffle(x3)
# # mini_batches = [
# #     x3[k:k+10]
# #     for k in range(0,len(x3),10)
# # ]
# # i=0
# # for x,y in mini_batches[0]:
# #     print(i)
# #     print(x)
# #     print(y)
# #     i+=i
# # mini_batches = [
# #     x3[k:k+3]
# #     for k in range(0,len(list(x3)),3)
# # ]
# # print(mini_batches)
# a = np.random.randn(10,1)
# print(np.argmax(a))
# print(a)

t = np.arange(0,784).reshape(28,28)
print(t)
t=t[4:24, 4:24]
print(len(t[1]))


p = np.arange(0,10)
print(p[0:5])
# #print(t.flatten())
# n = training_input[:, t.flatten()]
# print(n[1])




