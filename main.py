import autograd.numpy as np
from autograd import grad
import os
os.chdir('C:\\Users\\jmpark8187\\Documents\\Python Scripts\\feedforward neural network')
import read_mnist_data as rd

train_images = rd.parse_images('data/train-images-idx3-ubyte.gz')
train_labels = rd.parse_labels('data/train-labels-idx1-ubyte.gz')
test_images = rd.parse_images('data/t10k-images-idx3-ubyte.gz')
test_labels = rd.parse_labels('data/t10k-labels-idx1-ubyte.gz')

train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = np.append(train_images, np.ones((len(train_images), 1)), axis=1)
test_images = np.append(test_images, np.ones((len(test_images), 1)), axis=1)
W = np.random.rand(785, 10)
train_images = train_images[0:100, :]
train_labels = train_labels[0:100]
alpha = 0.8
print("Alpha:", (alpha))

np.random.seed(1)

# (784 - 256 - 64 - 10)
W0 = (2*np.random.random((785, 256)) - 1)
W1 = (2*np.random.random((256, 64)) - 1)
W2 = (2*np.random.random((64, 10)) - 1)


layer_0 = np.zeros((1, 785))
layer_1 = np.zeros((1, 256))
layer_2 = np.zeros((1, 64))
layer_3 = np.zeros((1, 10))


def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output*(1-output)


def ANN(W0, W1, W2, layer_0):
    layer_1 = sigmoid(np.dot(layer_0, W0))
    layer_2 = sigmoid(np.dot(layer_1, W1))
    layer_3 = sigmoid(np.dot(layer_2, W2))
    return layer_3


times = 20  # 반복횟수
print("do neural")

for j in range(times):
    for i in range(len(train_images)):
        layer_0[0] = train_images[i]
        layer_1 = sigmoid(np.dot(layer_0, W0))
        layer_2 = sigmoid(np.dot(layer_1, W1))
        layer_3 = sigmoid(np.dot(layer_2, W2))
        r = np.zeros(10)       # 정답이 입력되는 부분을 0으로 초기화
        r[train_labels[i]] = 1.0        # 정답에 해당하는 위치를 1로 설정
        layer_3_error = layer_3 - r
        # print("err:",layer_3_error)
        layer_3_delta = layer_3_error*sigmoid_output_to_derivative(layer_3)
        layer_2_error = layer_3_delta.dot(W2.T)
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
        layer_1_error = layer_2_delta.dot(W1.T)
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        # layer_3_error = layer_3 - train_labels
        W2 -= alpha * np.reshape(layer_2, (-1, 1))*layer_3_delta
        W1 -= alpha * np.reshape(layer_1, (-1, 1))*layer_2_delta
        W0 -= alpha * np.reshape(layer_0, (-1, 1))*layer_1_delta
        """
        #layer_3_error = train_labels - layer_3
        W2 += alpha * (layer_2.T.dot(layer_3_delta))
        W1 += alpha * (layer_1.T.dot(layer_2_delta))
        W0 += alpha * (layer_0.T.dot(layer_1_delta))
        """
    # 1개의 입력이 끝난후 마지막 입력에 대한 에러확인
    print((j), " iterations:")
    print("Error last input :", str(np.mean(np.abs(layer_3_error))))
    # print("out:",layer_3)
    # print("ans:",r)
