import keras
import tensorflow
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import SGD


def load_data_set():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_rows = x_train[0].shape[0]
    img_cols = x_train[1].shape[0]

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_train /= 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    num_classes = y_test.shape[1]
    num_pixels = x_train.shape[1] * x_train.shape[2]

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(0.01), metrics=['accuracy'])

    print(model.summary())


if __name__ == "__main__":
    load_data_set()
