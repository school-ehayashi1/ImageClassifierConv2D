# Import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.mnist import load_data
import numpy as np

# Preprocessing 
X, y = load_data(return_X_y=True)
from sklearn.model_selection import train_test_split
print(X.shape, y.shape)
Xtr,ytr, Xt, yt = train_test_split(X,y, random_shuffle=True, ratio=0.7)





# Custom loss functions(Practice)
def loss(y_train, y_pred): # Takes ytrain and yPred
    return len(y_train[y_train==y_pred])/len(y_train) # Returns value or same shape array of True False,
def custact(x): # input
    return x[x>0] # return values same shape

imgclf = keras.Sequential([
    keras.layers.Conv2D(64, 7, padding="same", activation="relu"),
    keras.layers.Conv2D(128, 7, padding="same", activation="relu"),
    keras.layers.Flatten(),
    keras.layer.Dense(100, activation=custact, kernel_initializer="he_uniform"),
    keras.layers.Dense(10, activation="softmax", kernel_initializer="he_uniform")
    ])
epoch =0
try: # tries to get input of epochs
  epoch = int(input("Epochs: "))
except TypeError:# if gets an invalid type.
  raise Exception("Invalid type, must be type int.")
  
# Fit+compile
imgclf.compile(optimizer="adam", loss=loss, metrics=["accuracy"]) #compile, adam opt, loss function called loss, metric by accuracy
imgclf.fit(Xtr, ytr, epochs=epoch) # epoch= how many times it runs/trains



# Info
print(imgclf.summary()+"\n") # prints each method
print(imgclf.history()+"\n")
print(imgclf.layers+"\n")



# predict function
def predict_digit(digitsinput):
  # load function
  if not imgclf:# sets imgclf to a model if null
    imgclf = keras.models.load_model("imgclfdigits.h5") # might not work, custom functions not included???
  return imgclf.predict(digitsinput)

# save model
def save_modeldigit(): # save model in h5 file
  keras.models.save_model("imgclfdigits.h5")

