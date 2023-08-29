# grader-required-cell 
import os
import tensorflow as tf
from tensorflow import keras

# load-and-inspect-the-data
# load-the-data
current_dir = os.getcwd()

# Append data
data_path = os.path.join(current_dir, "data/mnist.npz")

# discard test set
(x_test, y_test), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# normalize pixel values
x_train = x_train / 255.0

# now look at the shape of training data
data_shape = x_train.shape
print (f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

# defining callback
class myCallbacks(tf.keras.callacks.Callback):
   def on_epoch_end(self, epoch, logs={}):
      if logs.get("accuracy") > .92:
          print("Reached 92% accuracy so cancelling training!")
          self.model.stop_training = True


def parshva_mnist(x_train, y_train):
    callbacks = myCallback()

    # define model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.softmax)
      ])

  

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
  
    history = model.fit(x_train, y_train, epochs=15, callbacks=[callbacks])
    
    return history


hist = parshva.mnist(x_train, y_train)



















