import os # functions for interacting with the operating system
import cv2  #computer vision or load images & process images
import numpy as np  # arrays
import matplotlib.pyplot as plt  #visualisation of actual digits
import tensorflow as tf

mnist = tf.keras.datasets.mnist
# loading directly from tensorflow no need to dwnld
# splitting into testing data & training data
(x_train, y_train) , (x_test , y_test) = mnist.load_data()
# x = pixcel data  y = classification (no , digit)
# normalising or scaling it down
# 0-255(grayscale pixcel)
x_train  = tf.keras.utils.normalize(x_train, axis=1)
# normalising pixcels
x_test = tf.keras.utils.normalize(x_test, axis=1)




#
# model = tf.keras.models.Sequential()
# # adding layer and flattening into one big line of 784 pixcel
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
# softmax - all neurons add up to 1

# compiling model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=10)
# # model.save('hand.model')
# model.save('hand1.model')

# model = tf.keras.models.load_model('hand.model')
model = tf.keras.models.load_model('hand1.model')

#
# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)


image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("error")
    # except Exception as e:
    #     print(f"Error: {e}")
    finally:
        image_number += 1


# image_number = 1
#
# while True:
#     image_path = f"Digits/digit{image_number}.png"
#
#     if not os.path.isfile(image_path):
#         break
#
#     try:
#         img = cv2.imread(image_path)[:, :, 0]
#         img = np.invert(np.array([img]))
#         prediction = model.predict(img)
#         print(f"This digit is probably a {np.argmax(prediction)}")
#         plt.imshow(img[0], cmap=plt.cm.binary)
#         plt.show()
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#     finally:
#         image_number += 1


# image_number = 1
# while os.path.isfile('digits/digit{}.png'.format(image_number)):
#     try:
#         img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
#         img = np.invert(np.array([img]))
#         prediction = model.predict(img)
#         print("The number is probably a {}".format(np.argmax(prediction)))
#         plt.imshow(img[0], cmap=plt.cm.binary)
#         plt.show()
#         image_number += 1
#     except:
#         print("Error reading image! Proceeding with next image...")
#         image_number += 1
















