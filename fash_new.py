# https://www.tensorflow.org/tutorials/keras/basic_classification
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure()
# plt.imshow(X_train[9])
# plt.xlabel(class_names[y_train[9]])
# plt.show()
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
	])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy : ", test_acc)

prediction = model.predict(test_images)
print(np.argmax(prediction[0]))

def plot_image(i, prediction_array, true_label, img):
	prediction_array, true_label, img = prediction_array[i], true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img, cmap=plt.cm.binary)
	predict_label = np.argmax(prediction_array)

	if(predict_label==true_label):
		color = 'blue'
	else:
		color = 'red'
	plt.xlabel("{} {:0.2f}% ({})". format(class_names[predict_label], 100*np.max(prediction_array), class_names[true_label]), color=color)

def plt_value_array(i, prediction_array, true_label):
	prediction_array, true_label = prediction_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10), prediction_array, color="#777777")
	plt.ylim([0, 1])
	predict_label = np.argmax(prediction_array)
	thisplot[predict_label].set_color('red')
	thisplot[true_label].set_color('blue')

num_row = 5
num_cols = 3

num_images = num_row*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_row))

for i in range(num_images):
	plt.subplot(num_row, 2*num_cols, 2*i+1)
	plot_image(i, prediction, test_labels, test_images)
	plt.subplot(num_row, 2*num_cols, 2*i+2)
	plt_value_array(i, prediction, test_labels)
plt.show()

img = test_images[11]
img = (np.expand_dims(img, 0))

prediction_single = model.predict(img)
print(prediction_single)
plt_value_array(0, prediction_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
np.argmax(prediction_single[0])