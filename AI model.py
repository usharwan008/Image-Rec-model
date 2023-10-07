#importing reqd modules-

import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
import numpy as np
import matplotlib as plt

#Loading the CIFAR10 database-
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()            #Distributing DB for training & testing
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
"""pixel values of images are divided by 255 to reduce their values from 1-256 to 0-1 & flattened 
in order to send to the neural network"""

K = len(set(y_train))
print("number of classes:", K)                                        #Calculating total no. of output classes


# input layer of the model
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)

#Hidden layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

#output layer
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.summary()                                                                #model description

# Compiling our model
"""I have used the ADAM optimiser & sparse categorical crossentropy for parameters, 
and metrics have been set to accuracy"""

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Augmenting data & training the model-
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size

r = model.fit(train_generator, validation_data=(x_test, y_test),
			steps_per_epoch=steps_per_epoch, epochs=50)              #model trained till 50 epochs. Should produce around 95% accuracy

#mapping labels-
labels = '''airplane automobile bird cat deer dog frog horseship truck'''.split()


image_number = 0
plt.imshow(x_test[image_number])                              #displaying image


n = np.array(x_test[image_number])                            #loading image in array & reshaping
p = n.reshape(1, 32, 32, 3)

#passing img in the network for prediction and saving the predicted label-
predicted_label = labels[model.predict(p).argmax()]           


original_label = labels[y_test[image_number]]                 #loading og label

#Result-
print("Original label is {} and predicted label is {}".format(
	original_label, predicted_label))
