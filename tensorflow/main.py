#%%

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

(training_images, training_image_values), (validation_images, validation_image_values) = tf.keras.datasets.mnist.load_data()

training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
validation_images = validation_images.reshape(validation_images.shape[0], 28, 28, 1)

# Normalisen van de input images
training_images = training_images.astype('float32')
validation_images = validation_images.astype('float32')
training_images /= 255
validation_images /= 255

model = Sequential()

# Een convolution layer om de relatie tussen omliggende pixels te behouden
input_shape = (28, 28, 1)
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))

# Pool om "higher level" kenmerken te vinden
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten naar 1 dimensie
model.add(Flatten()) 

#layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(10,activation=tf.nn.softmax))


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(training_images,training_image_values)
model.evaluate(validation_images, validation_image_values)






# %%
