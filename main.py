import os
import autograd.numpy as np
from autograd import grad
import random
os.chdir('C:\\Users\\jmpark8187\\Documents\\Python Scripts\\feedforward neural network')
from read_mnist_data import read_images
from read_mnist_data import read_labels
from read_mnist_data import convert2onehot

# read data
train_images = read_images('data/train-images-idx3-ubyte.gz')
train_labels = read_labels('data/train-labels-idx1-ubyte.gz')
test_images = read_images('data/t10k-images-idx3-ubyte.gz')
test_labels = read_labels('data/t10k-labels-idx1-ubyte.gz')
train_labels = convert2onehot(train_labels)


def ReLU(x):
    # activation function
    return x * (x > 0)


def log_softmax(x):
    # log softmax prevent over/under flow
    z = x-x.max(1, keepdims=True)
    return (z - np.log(np.sum(np.exp(z), axis=1, keepdims=True)))


def cross_entropy(W0, W1, W2):
    # cost function, batch size: 600
    ran = 600 * (np.random.random_integers(1, 100) - 1)
    batch_x = train_images[ran + np.arange(600)]
    batch_y = train_labels[ran + np.arange(600)]
    layer1 = np.dot(batch_x, W0)
    layer2 = np.dot(ReLU(layer1), W1)
    layer3 = np.dot(ReLU(layer2), W2)
    return np.mean(-np.sum(batch_y * log_softmax(layer3), axis=1))


# gradient function
grad_W0 = grad(cross_entropy, argnum=0)
grad_W1 = grad(cross_entropy, argnum=1)
grad_W2 = grad(cross_entropy, argnum=2)

# weight
W0 = (2*np.random.random((785, 785)) - 1)
W1 = (2*np.random.random((785, 785)) - 1)
W2 = (2*np.random.random((785, 10)) - 1)

# training
for i in range(1000):
    W0_ = np.multiply(W0, np.random.binomial(n=1, p=0.8, size=(785, 1)))
    W1_ = np.multiply(W1, np.random.binomial(n=1, p=0.5, size=(785, 1)))
    W2_ = np.multiply(W2, np.random.binomial(n=1, p=0.5, size=(785, 1)))

    W0 -= grad_W0(W0_, W1_, W2_) * 0.01
    W1 -= grad_W1(W0_, W1_, W2_) * 0.01
    W2 -= grad_W2(W0_, W1_, W2_) * 0.01
    if i % 10 == 0:
        print('cost: ', cross_entropy(W0, W1, W2))


# evaluate
predict = []
correct_ans = 0
for i in range(len(test_labels)):
    layer1 = np.dot(test_images[i], W0)
    layer2 = np.dot(ReLU(layer1), W1)
    layer3 = np.dot(ReLU(layer2), W2)
    predict.append(np.argmax(layer3))

for i in predict:
    if predict[i] == test_labels[i]:
        correct_ans += 1

correct_ans = correct_ans/len(predict) * 100
print("accuracy", correct_ans, "%")
