import autograd.numpy as np
from autograd import grad
import struct
import random


def log_softmax(x):
    """ compute softmax """
    z = x-x.max(1, keepdims=True)
    return (z - np.log(np.sum(np.exp(z), axis=1, keepdims=True)))


def crossEntropy(weights):
    """ compute cross entropy between prediction and target """
    return -sum(sum(batch_y * log_softmax(np.dot(batch_x, weights))))


# open file
f_train_image = open('train-images.idx3-ubyte', 'rb')
f_train_label = open('train-labels.idx1-ubyte', 'rb')
f_test_image = open('t10k-images.idx3-ubyte', 'rb')
f_test_label = open('t10k-labels.idx1-ubyte', 'rb')

# data
train_images = []
train_labels = []
test_images = []
test_labels = []

# temporary vector to read file
tmp_image = []
tmp_label = []

# read first redundant bytes
f_train_image.read(16)
f_train_label.read(8)
f_test_image.read(16)
f_test_label.read(8)

# read training data
while True:
    tmp_image = f_train_image.read(784)
    tmp_label = f_train_label.read(1)
    if not tmp_image:
        break
    if not tmp_label:
        break
    train_images.append(struct.unpack(len(tmp_image)*'B', tmp_image))
    train_labels.append(int(tmp_label[0]))

train_images = np.divide(train_images, 255)
print("training read done")

# read test data
while True:
    tmp_image = f_test_image.read(784)
    tmp_label = f_test_label.read(1)
    if not tmp_image:
        break
    if not tmp_label:
        break
    test_images.append(struct.unpack(len(tmp_image)*'B', tmp_image))
    test_labels.append(int(tmp_label[0]))

test_images = np.divide(test_images, 255)
print("test read done")

# add bias term
train_images = np.append(train_images, np.ones((len(train_images), 1)), axis=1)
test_images = np.append(test_images, np.ones((len(test_images), 1)), axis=1)

# transform training labels to one-hot matrix
Train_labels = []
for i in train_labels:
    a = np.zeros(10)
    a[i] = 1
    Train_labels.append(a)

Train_labels = np.array(Train_labels)

# batch
batch_x = np.empty((0, 785))
batch_y = np.empty((0, 10))
for i in range(100):
    r = random.randint(1, train_images.shape[0]-1)
    batch_x = np.append(batch_x, [train_images[r]], axis=0)
    batch_y = np.append(batch_y, [Train_labels[r]], axis=0)

# Learning
weights = np.ones((785, 10))
print("before gradient descent: ", crossEntropy(weights))
grad_crossEntropy = grad(crossEntropy)

for i in range(500):
    weights -= 0.01 * grad_crossEntropy(weights)

print("after gradient descent: ", crossEntropy(weights))

# evaluate
predict = []
correct_ans = 0
for i in range(len(test_labels)):
    predict.append(np.argmax(np.dot(test_images[i], weights)))

for i in predict:
    if predict[i] == test_labels[i]:
        correct_ans += 1

correct_ans = correct_ans/len(predict) * 100
print("accuracy", correct_ans, "%")
