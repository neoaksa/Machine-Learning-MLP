import mlp
from numpy import genfromtxt
import numpy as np
import os
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def main():
     # whitecells()
     # training()
      testing()

def testing():
    # load data from csv
    currentpath = os.getcwd()
    testing_file = genfromtxt(currentpath + '/data/test_data.csv',
                               delimiter=',',
                               skip_header=1)
    testing_label = genfromtxt(currentpath + '/data/test_labels.csv',
                               delimiter=',',
                               skip_header=1)
    testing_input = normalizationX(cutwhitespace(testing_file))  # normalization data
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
    # print(amlp.perdict(testing_input))
    # print(testing_label)
    # type(testing_input)
    print("Correct in Test: {0} % ".format(evaluate/len(testing_label)*100))
    # plot a confusion_matrix for testing result
    pred = amlp.perdict(testing_input)
    cmat = confusion_matrix(testing_label, pred)
    print(cmat)
    df_cm = pd.DataFrame(cmat, range(10), range(10))
    sn.set(font_scale=1.4)
    # feed confusion_matrix to heatmap
    sn.heatmap(df_cm,annot=True,annot_kws={"size": 12},fmt='g',cmap='Blues',vmax=25)
    plt.show()

# training code
def training():
    # load data from csv
    currentpath = os.getcwd()
    training_file = genfromtxt(currentpath + '/data/train.csv',
                               delimiter=',',
                               skip_header=1,
                               max_rows=4000)
    # split training data into training and validation, 50 : 25
    training_input = normalizationX(cutwhitespace(training_file[0::2, 1:]))  # normalization data
    training_input = np.transpose(training_input)  # transpose
    training_input = np.split(training_input, len(training_input[0]), axis=1)
    training_tag = [YtoOutput(y) for y in training_file[0::2, 0]]  # construct tag for output node
    training_data = list(zip(training_input, training_tag))
    validation_input = normalizationX(cutwhitespace(training_file[1::4, 1:]))  # normalization data
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
    # 784 inputs(may cut witespace), 1 hidden layer with 50 nueros, 10 output.
    # in this case, we can only modify hidden layer of mlp structure
    size = [len(training_input[0]), 50, 10]
    amlp = mlp.mlp(0.9, 0.3, size, 200, 0, 1)
    amlp.miniBatch(training_data, 50, validation_date)

# handle output lable to 10 nodes
def YtoOutput(y):
    output = np.zeros(shape=(10,1))
    output[int(y)] = 1.0
    return output

# data normalization inputs
def normalizationX(x):
    return x / 255

# visualize the white space according to the mean of each cells in all samples
def whitecells():
    # load data from csv
    currentpath = os.getcwd()
    training_file = genfromtxt(currentpath + '/data/train.csv',
                               delimiter=',',
                               skip_header=1,
                               max_rows=40000)
    meanarray = np.mean(training_file[:,1:],axis=0)
    meanarray = meanarray.reshape(28,28)
    plt.imshow(meanarray, cmap='hot', interpolation='nearest')
    plt.show()

# according to whitecells(), cut whitespace
def cutwhitespace(input):
    ref = np.arange(0, 784).reshape(28, 28)
    ref = ref[4:24, 4:24]
    input = input[:,ref.flatten()]
    return input

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
