"""
A Neural Network framework to be used in the fitting/learning process.

Created by © Rodrigo Carvalho 2019
Mainteined by © Rodrigo Carvalho
"""

import numpy as np
import os

class NN:
    """
    A Neural Network implementation to solve regression and classification problems.

    """

    def __init__(self):
        print(
            "Model Initialized\nCall .model() to define your NN and .fit(x,y) to start the training process\nIt is possible to load a trained NN with load_network().\n"
        )
        pass

    ###################################
    #####          TOOLs         ######
    ###################################

    def onehot(self, arr):
        """
        Transoform input array into OneHotEncode format
        """
        a = np.array(arr)
        b = np.zeros((len(a), max(a) + 1))
        b[np.arange(len(a)), a] = 1
        return b

    def shuffle(self, a, b):
        """
        Shuffle a and b arrays together
        """
        c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
        a2 = c[:, : a.size // len(a)].reshape(a.shape)
        b2 = c[:, a.size // len(a) :].reshape(b.shape)
        np.random.shuffle(c)
        return a2, b2

    def model(self, units, activation, loss, metrics):
        """
        Define the NN model details
        > units: tuple or list to specify dimensionality of W(l) weight matrices for each layer l
                 the last layer must have same dimension of n of classes
        > activation: tuple or list of strings to specify activation function to be used in each output h(l). Options are relu, softmax, sigmoid and linear.
        > loss: string to specify type of loss function to be used during the training process. Options are crossentropy, mae and mse.
        > Metrics: number used to evaluations. Options are accuracy, crossentropy, mae or mse.
        """
        self.start = 0
        self.layers = len(units)
        self.units = units
        self.a_type = ["linear"] + activation
        self.loss_type = loss
        self.metrics_type = metrics
        print("--------------------------------------------------------------")
        print(" - Number of layers: ", self.layers)
        print(" - Number of h-units", self.units)
        print(" - Activation for each layer: ", activation)
        print(" - Loss function: ", self.loss_type)
        print(" - Metrics: ", self.metrics_type)

    ###################################
    #####  ACTIVATION FUNCTIONS  ######
    ###################################

    def sigmoid(self, x):
        """
        Compute the sigmoid of x
        """
        # x=np.array(xx)
        s = lambda a: 1 / (1 + np.exp(-a))
        return s(x)

    def relu(self, x):
        """
        Compute the relu of x
        """
        xx = np.array(x)
        xx[xx < 0] = 0
        return xx

    def softmax(self, x):
        """
        Compute the softmax
        """
        # x=np.array(xx)
        nm = np.exp(x - np.max(x))
        sm = np.sum(nm, axis=1)
        temp = []
        for i in range(nm.shape[-1]):
            temp.append(nm[:, i] / sm)
        return np.array(temp).T

    def softmax_derivatives(self, x):
        s = np.reshape(x, (-1, 1))
        return np.diagflat(s) - np.dot(s, s.T)

    ###################################
    #######   LOSS FUNCTIONS   ########
    ###################################

    def cross_entropy(self, x, y, n):
        """
        Compute the cross entropy from a softmax output (x)
        y is the labels
        data should be in onehot format
        """
        loss = -1.0 * np.sum(np.multiply(y, np.log(x))) / n
        return loss

    def mae(self, x, y, n):
        """
        Compute the mae for y and x arrays
        """
        loss = np.sum(np.abs(np.subtract(y, x))) / n
        return loss

    def mse(self, x, y, n):
        """
        Compute the mse for y and x arrays
        """
        loss = np.sum(np.subtract(y, x) ** 2) / (2 * n)
        return loss

    def accuracy(self, y, n):
        """
        Compute accuracy for a one-hot format labels
        """
        tot = 0.0
        for i in range(n):
            tot += 0.0 if np.argmax(self.h[-1][i]) == np.argmax(y[i]) else 1.0
        return 1.0 - tot / n

    def loss(self, x, y, n, mode):
        if mode == "crossentropy":
            return self.cross_entropy(x, y, n)
        elif mode == "mae":
            return self.mae(x, y, n)
        elif mode == "mse":
            return self.mse(x, y, n)
        else:
            print("Error with loss function options!!")

    def metrics(self, x, y, n, mode):
        if mode == "crossentropy":
            return self.cross_entropy(x, y, n)
        elif mode == "mae":
            return self.mae(x, y, n)
        elif mode == "mse":
            return self.mse(x, y, n)
        elif mode == "accuracy":
            return self.accuracy(y, n)
        elif mode == None:
            pass
        else:
            print("Error with metrics options!!")

    ###################################
    #######    PROPAGATION     ########
    ###################################

    def activation(self, x, activation):
        if activation == "sigmoid":
            return self.sigmoid(x)
        elif activation == "relu":
            return self.relu(x)
        elif activation == "softmax":
            return self.softmax(x)
        elif activation == "linear":
            return np.array(x)
        else:
            print("Error with activation function specification!!")

    def act_derivative(self, x, activation):
        # x=np.array(xx)
        if activation == "sigmoid":
            """
            Compute the derivative of the sigmoid
            """
            t = self.sigmoid(x)
            ds = t * np.subtract(1.0, t)
            return ds
        elif activation == "relu":
            """
            Compute the derivative of the relu
            """
            ds = np.zeros_like(x)
            ds[np.where(x > 0)] = 1.0
            return ds
        elif activation == "linear":
            """
            Compute the derivative of linear activation
            """
            return np.ones_like(x)
        else:
            print("Error with activation function specification")

    def forward(self, x):
        """
        Process the forward propagation. Return the loss value and generate z=wx+b and h=actv(z)
        """
        # start forward process
        z = []  # z=wx+b
        h = []  # h=activation(z)
        z.append(x)
        h.append(x)
        for l in range(self.layers):
            z.append(
                np.matmul(self.w[l], h[l].reshape(len(h[l]), -1, 1)).reshape(
                    len(h[l]), -1
                )
                + self.b[l]
            )
            h.append(self.activation(z[l + 1], self.a_type[l + 1]))

        # global variables h,z to be used in backpropg
        self.h = h
        self.z = z
        return

    def backward(self, x, y):
        """
        Process the backpropagation and compute gradients
        """
        # start parameters
        final_grad = []  # gradients/w
        final_grad_b = []  # gradients/b
        phi = []  # recurrent terms in backpropagation

        ### LAST LAYER (L)
        # first term | derivative of Loss/softmax/output
        if self.loss_type == "crossentropy" or self.loss_type == "mse":
            phi.append(np.subtract(self.h[-1], y))
        elif self.loss_type == "mae":
            ttt = np.subtract(self.h[-1], y)
            phi.append(ttt / np.abs(ttt))

        a = self.h[-2].reshape(self.n, 1, -1)
        b = phi[0].reshape(self.n, -1, 1)
        # for first layer
        grad = np.matmul(b, a)
        final_grad.append(np.sum(grad, axis=0) / self.n)
        final_grad_b.append(np.sum(phi[0], axis=0) / self.n)

        if self.layers > 1:
            # loop for L-1 layers
            for l in range(1, self.layers):
                d_actv = self.act_derivative(self.z[-l - 1], self.a_type[-l - 1])
                a = self.w[-l]
                b = phi[l - 1].reshape(self.n, -1, 1)
                c = np.matmul(a.T, b).reshape(self.n, -1)
                d = np.multiply(c, d_actv)
                phi.append(d)
                d = d.reshape(self.n, -1, 1)
                e = self.h[-l - 2].reshape(self.n, 1, -1)
                grad = np.matmul(d, e)
                final_grad.append(np.sum(grad, axis=0) / self.n)
                final_grad_b.append(np.sum(phi[l], axis=0) / self.n)

        temp = []
        temp2 = []
        for i in range(1, len(final_grad) + 1):
            temp.append(np.array(final_grad[-i]))
            temp2.append(np.array(final_grad_b[-i]))

        # final gradient in correct order
        grad_w = temp
        grad_b = temp2
        return grad_w, grad_b

    ###################################
    #######         FIT        ########
    ###################################

    def fit(
        self,
        x,
        y,
        epochs=100,
        lr=0.1,
        verbose=True,
        dc=True,
        test=None,
        n_batches=1,
        shuffle=True,
        init_rand=True,
    ):
        """
        Execute the training process, initializing and updating the network parameters
        At the end of fitting process, a dictionary will be returned with error and accuracy for each epoch

        >>>>> OPTIONS
        epochs: number of epochs to process the training
        n_batches: total number of batches (integer)
        shuffle: True/False to shuffle data at each epoch
        lr: learning rate. Float between 0,1
        dc: learning rate decay of type 1/x
        verbose: True/False to print results at each epoch
        test: input list [X_test,Y_test] to validation
        """
        # initialize variables
        keep_x = np.array(x)
        keep_y = np.array(y)
        # history
        self.loss_history = []
        self.acc_history = []
        it_index = []
        ep_index = []

        # setting validation data
        if test is not None:
            # initialize test variables
            self.x_test = np.array(test[0])
            self.y_test = np.array(test[1])
            self.n_test = np.shape(self.y_test)[0]
            # test history
            self.loss_history_test = []
            self.acc_history_test = []

        ## initiating iterations
        for it in range(epochs):
            # print(lr)
            # shuffle the data at each iteration
            if shuffle == True:
                x, y = self.shuffle(keep_x, keep_y)
            else:
                x = keep_x
                y = keep_y

            # split the shuffled data at each iteration to get batches
            x = np.split(x, n_batches)
            y = np.split(y, n_batches)

            ## initiating batches
            cost = 0.0
            acc = 0.0
            for b_idx in range(n_batches):
                it_index.append(it * b_idx)
                self.x = x[b_idx]
                self.y = np.array(y[b_idx])
                self.n = np.shape(self.y)[0]

                ### STARTING WEIGHTS
                if self.start == 0:
                    # start parameters
                    w = []
                    b = []
                    if init_rand == False:
                        w.append(np.ones((self.units[0], self.x.shape[-1])))
                        b.append(np.ones((self.units[0])))
                        for l in range(self.layers - 1):
                            w.append(np.ones((self.units[l + 1], self.units[l])))
                            b.append(np.ones((self.units[l + 1])))
                    elif init_rand == "rand":
                        w.append(np.random.rand(self.units[0], self.x.shape[-1]))
                        b.append(np.random.rand(self.units[0]))
                        for l in range(self.layers - 1):
                            w.append(np.random.rand(self.units[l + 1], self.units[l]))
                            b.append(np.random.rand(self.units[l + 1]))
                    else:
                        w.append(
                            np.random.normal(
                                0, scale=0.01, size=(self.units[0], self.x.shape[-1])
                            )
                        )
                        b.append(np.random.normal(0, scale=0.01, size=(self.units[0])))
                        for l in range(self.layers - 1):
                            w.append(
                                np.random.normal(
                                    0,
                                    scale=0.01,
                                    size=(self.units[l + 1], self.units[l]),
                                )
                            )
                            b.append(
                                np.random.normal(
                                    0, scale=0.01, size=(self.units[l + 1])
                                )
                            )
                    self.w = w
                    self.b = b
                    self.start = 1
                ### END WEIGHTS

                # initial values
                self.forward(self.x)
                grad_w, grad_b = self.backward(self.x, self.y)

                ######################
                # starting optimization
                for i in range(len(grad_w)):
                    self.w[i] = self.w[i] - lr * grad_w[i]
                    self.b[i] = self.b[i] - lr * grad_b[i]

                # update cost and accuracy
                self.forward(self.x)
                cost = self.loss(self.h[-1], self.y, self.n, self.loss_type)
                acc = self.metrics(self.h[-1], self.y, self.n, self.metrics_type)

                # acc=acc/n_batches
                # cost=cost/n_batches
                if test is not None:
                    if b_idx == n_batches - 1:
                        self.forward(self.x_test)
                        cost_test = self.loss(
                            self.h[-1], self.y_test, self.n_test, self.loss_type
                        )
                        acc_test = self.metrics(
                            self.h[-1], self.y_test, self.n_test, self.metrics_type
                        )
                        print(
                            "%2d Batches, Epoch %2d -- Loss: %.3f Metric: %.3f  |-|  Test_Loss: %.3f | Metric: %.3f"
                            % (b_idx + 1, it + 1, cost, acc, cost_test, acc_test),
                            end="\r",
                        ) if verbose == True else None
                        self.loss_history.append(cost)
                        self.acc_history.append(acc)
                        self.loss_history_test.append(cost_test)
                        self.acc_history_test.append(acc_test)
                        ep_index.append(it * b_idx)
                    else:
                        print(
                            "%2d Batches, Epoch %2d -- Loss: %.3f Metric: %.3f"
                            % (b_idx + 1, it + 1, cost, acc),
                            end="\r",
                        ) if verbose == True else None
                        self.loss_history.append(cost)
                        self.acc_history.append(acc)
                else:
                    print(
                        "%2d Batches, Epoch %2d -- Loss: %.3f Metric: %.3f"
                        % (b_idx + 1, it + 1, cost, acc),
                        end="\r",
                    ) if verbose == True else None
                    self.loss_history.append(cost)
                    self.acc_history.append(acc)

            # jump one line
            print("")

            # decay
            lr = lr * (1 - 2 * lr / (it + 1)) if dc == True else lr

        if test is not None:
            dic = {
                "Iterations": it_index,
                "Epochs": ep_index,
                "Loss": self.loss_history,
                "Metrics": self.acc_history,
                "TestLoss": self.loss_history_test,
                "TestMetrics": self.acc_history_test,
            }
        else:
            dic = {
                "Iterations": it_index,
                "Epochs": ep_index,
                "Loss": self.loss_history,
                "Metrics": self.acc_history,
            }
        return dic

    def get_w(self):
        """
        Get parameters after training
        """
        return self.w, self.b

    def evaluate(self, xx, yy):
        """
        Return the output values of the NN for input x data and print scores accordling to the supplied y data.
        """
        x = np.array(xx)
        y = np.array(yy)
        # initialize variables
        n = len(x)
        self.forward(x)
        cost = self.loss(self.h[-1], y, n, self.loss_type)
        # Print scores
        print("\nLoss: ", cost)
        acc = self.metrics(self.h[-1], y, n, self.metrics_type)
        print("Metric: ", acc, "\n")
        return self.h[-1]

    def predict(self, xx):
        """
        Return the actual outputs ot the NN for the input x data.
        """
        x = np.array(xx)
        self.forward(x)
        return self.h[-1]

    def save_network(self):
        """Save the network parameters after training. Parameters will be saved in the folder nn_parameters.

        Returns
        -------
        Nothing

        """
        import pickle
        import gzip
        import shutil

        os.mkdir("nn_parameters")
        os.chdir("nn_parameters")

        # Network
        data = [self.layers, self.units, self.a_type]
        with open("net", "wb") as filename:
            pickle.dump(data, filename)
        with open("net", "rb") as filename:
            with gzip.open("net.gz", "wb") as f:
                shutil.copyfileobj(filename, f)
        os.remove("net")

        # W
        with open("w", "wb") as filename:
            pickle.dump(self.w, filename)
        with open("w", "rb") as filename:
            with gzip.open("w.gz", "wb") as f:
                shutil.copyfileobj(filename, f)
        os.remove("w")

        # b
        with open("b", "wb") as filename:
            pickle.dump(self.b, filename)
        with open("b", "rb") as filename:
            with gzip.open("b.gz", "wb") as f:
                shutil.copyfileobj(filename, f)
        os.remove("b")
        os.chdir("..")

    def load_network(self):
        """Load the network parameters from folder nn_parameters. After load, it is possible to call the predict and get_w method.

        Returns
        -------
        Nothing

        """
        if os.path.exists("nn_parameters") == False:
            return print("Folder does not exist!")

        import pickle
        import gzip

        os.chdir("nn_parameters")

        # Network
        with gzip.open("net.gz", "rb") as filec:
            data = pickle.load(filec)
        self.layers = data[0]
        self.units = data[1]
        self.a_type = data[2]
        # W
        with gzip.open("w.gz", "rb") as filec:
            self.w = pickle.load(filec)
        # b
        with gzip.open("b.gz", "rb") as filec:
            self.b = pickle.load(filec)

        os.chdir("..")
