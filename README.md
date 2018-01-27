# Handwritten-Digit-Recognition-with-a-Multilayer-Perceptron
* this module is used for handwritten digit recognitionn, which consists of
  * mini batch alorithms
  * forward alorithms
  * back ward alorithms
  * save and load structure
  * plot error
* folder structure
  * /main.py    # main entry
  * /mlp.py     # algorithms
  * /data       # training and test data
  * /document   # document for reference
  *   /figure_1.png   #error_plot
  * /weight.py  # save weights
  * /size.py    # save mlp structure
  * /biases.py  # save biases weights
  
* functions
  
  * mlp.__init__  #inital mlp object
    * lr: learning rate
    * momntum: nabla w montum
    * size: structure of mlp. e.g [784, 50, 10] denotes 784 inputs, 1 hidden layer with 50 nueros, 10 output.
    * epoch: max epch number
    * load: 0= without load 1=load from npy files
    * save: 0= without save 1=save to npy files
    
  * miniBatch     # run mlp with training or test data
    * traning data
    * mini_batch_size
    * test_data: optional 
  
  * perdict       # perform perdict
    * test_data: input data should be [n,[m,1] let n=number of samples, m=features
