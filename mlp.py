import numpy as np

class mlp:
    def __init__(self, lr=0.3, momentum=0.5, size=None, epoch=100, load=0):
        self.learningRate = lr  # learning Rate
        self.maxEpoch = epoch   # epoch number
        self.size = size        # network structure
        self.momentum = momentum # moentum
        self.num_layers = len(size) # the num of layers
        # for a MLP, the weights matrix should reverse the order of num of neuros in each layer
        # the biases weights matrix should be num of neuros in the hidden layer + output layer
        # e.g. for a n*m1*m2*k network, weigths for each layer = m1*n , m2*m1 , k*m2
        # biases weights = m1*1 , m2*1, k*1
        if load == 1:
            self.weights = np.load("weights.npy")
            self.biases = np.load("biases.npy")
        else:
            # initial the weight
            self.weights = [np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])]
            # initial the bais
            self.biases = [np.random.randn(y, 1) for y in size[1:]]
        # save previous nabla weights
        self.nabla_b_p = [np.zeros(b.shape) for b in self.biases]
        self.nabla_w_p = [np.zeros(w.shape) for w in self.weights]

    # forward network, a is input vecotr [n,1]
    def forward(self, a):
        # loop weights matrix and biases matrix in each layer
        # we can perform the forward network to the output layer
        for b,w in zip(self.biases, self.weights):
            a = self.sigmode(np.dot(w,a)+b)
        return a

    #  miniBatch
    def miniBatch(self, training_data, mini_batch_size,  test_data):
        if test_data:
            n_test = len(test_data)
        # loop epochs
        for j in range(self.maxEpoch):
            # shuffle the data, this step is nesscary for minibatch
            np.random.shuffle(training_data)
            # split training_data to len(training_data)/batch size the minibatch
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0,len(training_data),mini_batch_size)
            ]
            #print(training_data[0])
            # for each mini batch we update weights
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            # evaluate
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    # update mini batch
    def update_mini_batch(self, mini_batch):
        # set all bias and weights as zero
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # x in input vector , y is true value for caculating error
        for x,y in mini_batch:
            # caculate drivative
            delta_nabla_b, delta_nabla_w =  self.backprop(x,y)
            # cumulate all derivative which output in each minibatch
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
        # update weights and bias by nabla weights and momentum weights
        self.weights = [w - (self.learningRate/len(mini_batch))*nw - (self.momentum/len(mini_batch))*pw for w,nw,pw in zip(self.weights, nabla_w, self.nabla_w_p)]
        self.biases = [b-(self.learningRate/len(mini_batch))*nb - (self.momentum/len(mini_batch))*pb for b,nb,pb in zip(self.biases, nabla_b,self.nabla_b_p)]
        # save to momentum weights matrix
        self.nabla_b_p = nabla_b
        self.nabla_w_p = nabla_w

    def backprop(self, x, y):
        # empty matrix for storing
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # forward
        # initial first layer output as x
        activation = x
        # store the value of neuros
        activations = [x]
        # zs for storing value before active function
        zs = []
        # save value zs and after sigmod value activetion in forward
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmode(z)
            activations.append(activation)

        #  drivative of error in output layer = y(1-y)(t-y)
        delta = self.cost_derivative(activations[-1],y) * self.sigmode_prime(zs[-1])
        # bias=1, nabla weight of bias in the last layer is Ey
        nabla_b[-1] = delta
        # nabla weight matrix of output layer = D_error_outputlayer * output_value_previous
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-2]))
        # caculate hidden layer drivative
        for l in range(2, self.num_layers):
            z = zs[-l]      # input of last layer
            sp = self.sigmode_prime(z)       #  drivative of sigmode
            # D_error_hidden = weight_NextLayer * D_Error_NextLayer * f'(value of current node)
            delta = np.dot(np.transpose(self.weights[-l+1]), delta)*sp
            # nabla weights
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))
        return (nabla_b,nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward(x)),np.argmax(y)) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    # define sigmode as activate function
    def sigmode(self,x):
        return 1.0/(1.0 + np.exp(-x))

    #define softmax as activate function
    def softmax(self,inputs):
        return np.exp(inputs)/float(sum(np.exp(inputs)))

    # derivative of sigmode activate function
    def sigmode_prime(self,x):
        return self.sigmode(x) * (1 - self.sigmode(x))
