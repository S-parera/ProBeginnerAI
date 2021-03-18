import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf 
from tensorflow.keras import datasets, layers, models, optimizers

#HYPERPARAMETERS
epochs = 20
batch_size = 128
verbose = 1
optimizer = optimizers.Adam()
validation_split = 0.95

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
n_classes = 10

def build(input_shape, n_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(20, (5,5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Conv2D(50, (5,5), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))

    return model

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

X_train = X_train.reshape((60000, 28,28,1))
X_test = X_test.reshape((10000, 28,28,1))

X_train, X_test = X_train/255.0, X_test/255.0

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

model = build(input_shape, n_classes)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
model.summary()

callbacks = [tf.keras.callbacks.TensorBoard(log_dir="./logs")]

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                    validation_split=validation_split, callbacks=callbacks)

score = model.evaluate(X_test, y_test, verbose=verbose)
print("\nTest scores: ", score[0])
print("\nTest accuracy: ", score[1])