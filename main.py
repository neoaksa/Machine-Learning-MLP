import mlp
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    whitecells()
    #training()
    #testing()

def testing():
    # load data from csv
    currentpath = os.getcwd()
    testing_file = genfromtxt(currentpath + '/data/test_data.csv',
                               delimiter=',',
                               skip_header=1)
    testing_label = genfromtxt(currentpath + '/data/test_labels.csv',
                               delimiter=',',
                               skip_header=1)
    testing_input = normalizationX(testing_file)  # normalization data
    testing_input = np.transpose(testing_input)  # transpose
    testing_input = np.split(testing_input, len(testing_input[0]), axis=1)
    testing_tag = [YtoOutput(y) for y in testing_label]  # construct tag for output node
    testing_data = list(zip(testing_input, testing_tag))

    # construct mlp object
    # lr=learning rate=0.3,
    # momentum=0.5,
    # size=None,
    # epoch=100,
    # load=1 load trained nuero
    # save=0 do not save neuro
    amlp = mlp.mlp(None, None, None, None, 1, 0)
    np.savetxt('test_result.csv', amlp.perdict(testing_input), delimiter=',')
    evaluate = amlp.evaluate(testing_data)
    print("Correct in Test: {0} % ".format(evaluate/len(testing_label)*100))


# training code
def training():
    # load data from csv
    currentpath = os.getcwd()
    training_file = genfromtxt(currentpath + '/data/train.csv',
                               delimiter=',',
                               skip_header=1,
                               max_rows=4000)
    # split training data into training and validation, 50 : 25
    training_input = normalizationX(training_file[0::2, 1:])  # normalization data
    training_input = np.transpose(training_input)  # transpose
    training_input = np.split(training_input, len(training_input[0]), axis=1)
    training_tag = [YtoOutput(y) for y in training_file[0::2, 0]]  # construct tag for output node
    training_data = list(zip(training_input, training_tag))
    validation_input = normalizationX(training_file[1::4, 1:])  # normalization data
    validation_input = np.transpose(validation_input)  # transpose
    validation_input = np.split(validation_input, len(validation_input[0]), axis=1)
    validation_tag = [YtoOutput(y) for y in training_file[1::4, 0]]  # construct tag for output node
    validation_date = list(zip(validation_input, validation_tag))
    # test_data_file = genfromtxt(currentpath+'/data/test.csv',
    #             delimiter=',',
    #             skip_header=1,
    #             max_rows=100)

    # construct mlp object
    # lr=learning rate=0.3,
    # momentum=0.5,
    # size=None,
    # epoch=100,
    # load=load weights from npz file=0
    # save=save new module structure into npy files
    # 784 inputs, 1 hidden layer with 50 nueros, 10 output.
    # in this case, we can only modify hidden layer of mlp structure
    size = [784, 50, 10]
    amlp = mlp.mlp(0.5, 0.3, size, 200, 0, 1)
    amlp.miniBatch(training_data, 50, validation_date)

# handle output lable to 10 nodes
def YtoOutput(y):
    output = np.zeros(shape=(10,1))
    output[int(y)] = 1.0
    return output

# data normalization inputs
def normalizationX(x):
    return x / 255

# analysis white cells
def whitecells():
    # load data from csv
    # load data from csv
    currentpath = os.getcwd()
    training_file = genfromtxt(currentpath + '/data/train.csv',
                               delimiter=',',
                               skip_header=1,
                               max_rows=4000)
    meanarray = np.mean(training_file[:,1:],axis=0)
    meanarray = meanarray.reshape(28,28)
    plt.imshow(meanarray, cmap='hot', interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    main()
