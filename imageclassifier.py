
# Import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.mnist import load_data
import numpy as np

# Preprocessing 
X, y = load_data()
from sklearn.model_selection import train_test_split
Xtr,ytr, Xt, yt = train_test_split(X,y, shuffle=True)

Xtr, ytr, Xt, yt = np.array(Xtr), np.array(ytr), np.array(Xt), np.array(yt)

print(Xtr[0].shape)

# Custom loss functions(Practice)
def loss(y_train, y_pred): # Takes ytrain and yPred
    return len(y_train[y_train==y_pred])/len(y_train) # Returns value or same shape array of True False,
def custact(x): # input
    return x[x>0] # return values same shape
Xtr = np.reshape(Xtr, (60000, 28,28,1))
imgclf = keras.Sequential([
    keras.layers.Conv2D(64, 7, padding="same", activation="relu"),
    keras.layers.Conv2D(128, 7, padding="same", activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation=custact, kernel_initializer="he_uniform"),
    keras.layers.Dense(10, activation="softmax", kernel_initializer="he_uniform")
    ])

'''
try: # tries to get input of epochs
  epoch = int(input("Epochs: "))
except ValueError:# if gets an invalid type.
  raise Exception("Invalid type, must be type int.")
  '''
# Fit+compile
imgclf.compile(optimizer="adam", loss=loss, metrics=["accuracy"]) #compile, adam opt, loss function called loss, metric by accuracy
imgclf.fit(np.array(Xtr), np.array(ytr), epochs=10) # epoch= how many times it runs/trains



# Info
print(imgclf.summary()+"\n") # prints each method
print(imgclf.history()+"\n")
print(imgclf.layers+"\n")



# predict function
def predict_digit(digitsinput):
  # load function
  if not imgclf:# sets imgclf to a model if null
    imgclf = keras.models.load_model("imgclfdigits.h5") # might not work, custom functions not included???
  return imgclf.predict(np.array(digitsinput))

# save model
def save_modeldigit(): # save model in h5 file
  keras.models.save_model("imgclfdigits.h5")




#Proto2

# Import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.mnist import load_data
import numpy as np

# Preprocessing 
X, y = load_data()
from sklearn.model_selection import train_test_split
Xtr,ytr, Xt, yt = train_test_split(X,y, shuffle=True)

Xtr, ytr, Xt, yt = np.array(Xtr), np.array(ytr), np.array(Xt), np.array(yt)

print(Xtr[0].shape)

# Custom loss functions(Practice)
def loss(y_train, y_pred): # Takes ytrain and yPred
    return len(y_train[y_train==y_pred])/len(y_train) # Returns value or same shape array of True False,
def custact(x): # input
    return x[x>0] # return values same shape
Xtr = np.expand_dims(Xtr, -1)
imgclf = keras.Sequential([
    keras.Input((1, 28,28)),
    keras.layers.Conv2D(64, 7, padding="same", activation="relu"),
    keras.layers.Conv2D(128, 7, padding="same", activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation=custact, kernel_initializer="he_uniform"),
    keras.layers.Dense(10, activation="softmax", kernel_initializer="he_uniform")
    ])

'''
try: # tries to get input of epochs
  epoch = int(input("Epochs: "))
except ValueError:# if gets an invalid type.
  raise Exception("Invalid type, must be type int.")
  '''
# Fit+compile
imgclf.compile(optimizer="adam", loss=loss, metrics=["accuracy"]) #compile, adam opt, loss function called loss, metric by accuracy
imgclf.fit(np.array(Xtr), np.array(ytr), epochs=10, batch_size=32) # epoch= how many times it runs/trains



# Info
print(imgclf.summary()+"\n") # prints each method
print(imgclf.history()+"\n")
print(imgclf.layers+"\n")



# predict function
def predict_digit(digitsinput):
  # load function
  if not imgclf:# sets imgclf to a model if null
    imgclf = keras.models.load_model("imgclfdigits.h5") # might not work, custom functions not included???
  return imgclf.predict(np.array(digitsinput))

# save model
def save_modeldigit(): # save model in h5 file
  keras.models.save_model("imgclfdigits.h5")
