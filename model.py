
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files


def get_data(filename):
    
    with open(filename) as training_file:
      # Your code starts here
      data = np.loadtxt(training_file,delimiter=',',skiprows=1)
      for row in data:
        images.append(row[1:])
        labels.append(row[0])
      images = np.reshape(images,(images.shape[0],28,28))
      labels = np.array(label).astype('float')
    return images, labels

testing_images, testing_labels = get_data('sign_mnist_test.csv')
training_images, training_labels = get_data('sign_mnist_train.csv')



print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)


training_images = np.expand_dims(training_images,3)
testing_images = np.expand_dims(testing_images,3)

train_datagen = ImageDataGenerator(rescale=1/255,width_shift_range=0.2,height_shift_range=0.2, #Data augmentaion
                                   rotation_range = 40,zoom_range = 0.2)

validation_datagen = ImageDataGenerator(rescale=1/255,zoom_range = 0.2)
    

print(training_images.shape)
print(testing_images.shape)
    

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(26, activation=tf.nn.softmax)]
    )

model.compile(loss = tf.losses.sparse_categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])

model.summary()

history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=32,),
                              steps_per_epoch=len(training_images) / 32,
                              epochs=5,
                              validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),
                              validation_steps=len(testing_images) / 32)

model.evaluate(testing_images, testing_labels)
    



import matplotlib.pyplot as plt
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val+loss"]

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
