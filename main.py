import mlp
from numpy import genfromtxt
import numpy as np
import os

def main():
    # load data from csv
    currentpath = os.getcwd()
    training_file = genfromtxt(currentpath+'/data/train.csv',
                delimiter=',',
                skip_header=1,
                max_rows=4000)
    # split training data into training and validation, 50 : 25
    training_input = normalizationX(training_file[0::2,1:])     # normalization data
    training_input = np.transpose(training_input)               # transpose
    training_input = np.split(training_input,len(training_input[0]), axis=1)
    training_tag = [YtoOutput(y) for y in training_file[0::2,0]]    # construct tag for output node
    training_data = list(zip(training_input,training_tag))
    validation_input = normalizationX(training_file[1::4,1:])   # normalization data
    validation_input = np.transpose(validation_input)            # transpose
    validation_input = np.split(validation_input,len(validation_input[0]), axis=1)
    validation_tag = [YtoOutput(y) for y in training_file[1::4,0]]  # construct tag for output node
    validation_date = list(zip(validation_input,validation_tag))
    # test_data_file = genfromtxt(currentpath+'/data/test.csv',
    #             delimiter=',',
    #             skip_header=1,
    #             max_rows=100)

    # construct mlp object
    # defalut: lr=learning rate=0.3, momentum=0.5, size=None, epoch=100, load=load weights from npz file=0
    # 784 inputs, 1 hidden layer with 50 nueros, 10 output.
    # in this case, we can only can hidden layer of mlp structure
    size = [784, 50, 10]
    amlp = mlp.mlp(0.5, 0.3, size, 200,0)
    amlp.miniBatch(training_data,50, validation_date)
    # save weights
    np.save('weights', amlp.weights)
    np.save('biases', amlp.biases)
    np.save('size',size)

# handle output lable to 10 nodes
def YtoOutput(y):
    output = np.zeros(shape=(10,1))
    output[int(y)] = 1.0
    return output

# data normalization inputs
def normalizationX(x):
    return x / 255

if __name__ == '__main__':
    main()
