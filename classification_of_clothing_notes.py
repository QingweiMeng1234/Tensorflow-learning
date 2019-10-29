import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
#Part1

# this is a data set embedded in Keras.
data = keras.datasets.fashion_mnist 

#embedded in Keras. Allows you to split the data into two sets. It is recommended to use part 
#part of data for training, part for testing.
(train_images,train_labels), (test_images,test_labels) = data.load_data() 



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Reduce the images into a number between 0 to 1 for easier calculation.
train_images = train_images/255.0
test_images = test_images/255.0

# #this prints out a numpy array that represents the image      
# print(train_images[0])      

# #this shows the image
# plt.imshow(train_images[0],cmap= plt.cm.binary)
# plt.show()

#Part 2: Creating the Model

#This is the model. We first flatten the layer, meaning taking each element in the nested arrays into one big array.
#Then we use Dense, meaning that each neuron from the first layer has a surjective map on the second layer with 128 elements.
#Activation refers to the Activation function used for calculation.
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128,activation= "relu"),
    keras.layers.Dense(10,activation = "softmax")
])

#Compile the model. Different optimizers and loss function may produce different results. Accuracy tells
#how accurate the prediction is.
model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",metrics=["accuracy"])

#model.fit trains the model. Epoches: times the model sees one image
model.fit(train_images,train_labels, epochs=20)

# #Output
# test_acc,test_loss = model.evaluate(test_images,  test_labels)

# print('\nTest accuracy:', test_acc)

#Part 3 Make a Prediction

#predict receives a number or a np.array. Now we predict all test_images. We obtain a np.array of our predictions
#with a probability in each entry.
prediction = model.predict(test_images)

#Show that the prediction matches the actual stuff.
#np.argmax() gives you the index of the highest-value.
#class_names is to show the name with the corresponding position.
#print(class_names[np.argmax(prediction[0])])

#Or, we want to visualize how good the prediction and the actual image in a better way.

#A random number generator
def randomnum():
    return random.randint(9949,9999)

while True:
    a = randomnum()
    b = randomnum()
    if(a!=b):
        break

list = []
if a<b :
    list.insert(0,a)
    list.insert(1,b)
else:
    list.insert(0,b)
    list.insert(1,a)


for i in range(list[0],list[1]):
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: "+class_names[np.argmax(prediction[i])])
    plt.show()



