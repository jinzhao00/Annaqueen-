import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

import datetime


data_set = fashion_mnist

# data_set = mnist # 这里可以切换成最普通的minst数据集

# 计算距离（欧氏距离）


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# 距离shape


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# 损失(loss)函数


def contrastive_loss(y_true, y_pred):
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


'''
    x 传进来的是对应的图像数据集
    class_indices 即对应label在train set 中的index索引
'''


def create_pairs(x, class_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(class_indices[d]) for d in range(len(class_indices))]) - 1
    for d in range(len(class_indices)):
        for i in range(n):
            z1, z2 = class_indices[d][i], class_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, len(class_indices))
            dn = (d + inc) % len(class_indices)
            z1, z2 = class_indices[d][i], class_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1.0, 0.0]
    return np.array(pairs), np.array(labels)

# 随机打乱数据


def shuffle(x, y):
    # indices = the number of images in the source data set
    index = np.arange(len(y))
    np.random.shuffle(index)
    return x[index], y[index]

# 构建模型


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Conv2D(16, kernel_size=(3, 3),
               activation='relu', kernel_regularizer=regularizers.l2(0.06), padding='same')(input)
    # kernel_regularizer=regularizers.l2(0.06),
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu',
               kernel_regularizer=regularizers.l2(0.1))(x)
    # ,kernel_regularizer=regularizers.l2(0.1)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu',
               kernel_regularizer=regularizers.l2(0.08))(x)
    # ,kernel_regularizer = regularizers.l2(0.08)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    return Model(input, x)

# 计算accuracy


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])


def showImages(x, y):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        for j in range(2):
            plt.subplot(8, 8, i*2+j+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x[i][j].reshape(
                28, 28), cmap=plt.cm.binary)
        plt.subplot(8, 8, i*2+1)
        plt.xlabel(['true', 'false'][y[i]])
    plt.show()

# 2 个元组：
# train_images, test_images: uint8 数组表示的灰度图像，尺寸为 (num_samples, 28, 28)。
# train_labels, test_labels: uint8 数组表示的数字标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)。


# 类别	描述	      中文
# 0	    T-shirt/top	 T恤/上衣
# 1	    Trouser	     裤子
# 2	    Pullover	 套头衫
# 3	    Dress	     连衣裙
# 4	    Coat	     外套
# 5	    Sandal	     凉鞋
# 6	    Shirt	     衬衫
# 7	    Sneaker	     运动鞋
# 8	    Bag	         背包
# 9	    Ankle boot	 短靴
class_names = ['top', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# num_classes = 10
# 用来标记训练用的类型（上述10类型之一）
train_classes = [0, 1, 2, 4, 5, 9]
# 用来标记测试用的类型（上述10类型之一）
test_classes = [3, 7, 8, 6]

# input image dimensions
img_rows, img_cols = 28, 28

epochs = 10
# 加载数据
(train_images, train_labels), (test_images, test_labels) = data_set.load_data()
# concat train and test data
train_images = np.concatenate((train_images, test_images))
train_labels = np.concatenate((train_labels, test_labels))

# 如果backend 用的是channels_first模式的则对图像进行处理
if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(
        train_images.shape[0], 1, img_rows, img_cols)
    test_images = test_images.reshape(
        test_images.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_images = train_images.reshape(
        train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(
        test_images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 把灰度图归一化 0 - 255 to 0 - 1
train_images, test_images = train_images.astype(
    'float32'), test_images.astype('float32')
train_images, test_images = train_images / 255.0, test_images / 255.0

input_shape = train_images.shape[1:]

# create training+test positive and negative pairs
# 从全体样本中取出训练数据集，这里用了上面定义的train_classes
class_indices = [np.where(train_labels == i)[0] for i in train_classes]

'''
这行代码统计每一个labels有多少数据
mnist 数据是不平均的 6903 7877 6990 7141 6824 6313 6876 7293 6825 6958 70000
而fashion_mnist 数据是平均的 各7000
'''
# class_count = [len(np.where(train_labels == i)[0]) for i in range(10)]

# 使用上面得到的训练数据集的indices 将对应的测试图片数据两两配对，作为最后传入孪生网络的数据
pairs, y = create_pairs(train_images, class_indices)
pairs_for_012459, label_for_012459 = pairs, y
# 取80%的数据集用来训练 另外20%作为测试集
train_y_len = round(len(y)*0.8)

# 随机打乱数据集，取前train_y_len个，若不随机打乱，则后面几种类型会取不到
shuffled_pairs, shuffled_y = shuffle(pairs, y)
train_pairs, train_y = shuffled_pairs[0:train_y_len], shuffled_y[0:train_y_len]

# 将后面20%的数据作为训练用的test sets
test_pairs, test_y = shuffled_pairs[train_y_len:], shuffled_y[train_y_len:]

# 同理取test_classes作为测试集，这里的测试集和上面训练用的test sets不一样，
class_indices = [np.where(train_labels == i)[0] for i in test_classes]
ftest_pairs, ftest_y = create_pairs(train_images, class_indices)

# class_indices = [np.where(test_labels == i)[0] for i in train_classes]
# test_pairs, test_y = create_pairs(test_images, class_indices, train_classes)

# 显示图片代码
# showImages(train_pairs, train_y)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# 生成tensorboard 图表
log_dir = "logs/fit/"  # + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# trainRMSprop
rms = RMSprop()
# 编译模型
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
# 训练模型
history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_y, batch_size=128,
                    epochs=epochs, validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_y), callbacks=[tensorboard_callback])
# 存储模型
# model.save('final.h5')

# 画图查看学习历史
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plot_train_history(history, 'loss', 'val_loss')
# plt.subplot(1, 2, 2)
# plot_train_history(history, 'accuracy', 'val_accuracy')
# plt.show()

# compute final accuracy on training and test sets
y_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
tr_acc = compute_accuracy(train_y, y_pred)
y_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
te_acc = compute_accuracy(test_y, y_pred)

# compute_accuracy(ftest_y, model.predict([ftest_pairs[:, 0], ftest_pairs[:, 1]]))
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

y_pred = model.predict([pairs_for_012459[:, 0], pairs_for_012459[:, 1]])
tr_acc = compute_accuracy(label_for_012459, y_pred)
print('* Accuracy on 012459 set: %0.2f%%' % (100 * tr_acc))

class_indices = [np.where(train_labels == i)[0] for i in range(10)]
all_pairs, all_label = create_pairs(train_images, class_indices)
y_pred = model.predict([all_pairs[:, 0], all_pairs[:, 1]])
tr_acc = compute_accuracy(all_label, y_pred)
print('* Accuracy on all set: %0.2f%%' % (100 * tr_acc))

y_pred = model.predict([ftest_pairs[:, 0], ftest_pairs[:, 1]])
tx_acc = compute_accuracy(ftest_y, y_pred)
print('* Accuracy on final_test set: %0.2f%%' % (100 * tx_acc))

