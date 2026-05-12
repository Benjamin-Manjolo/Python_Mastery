import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# Fake data so the code has something to train on
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(0, 2, 1000)
X_test  = np.random.rand(200, 20)
y_test  = np.random.randint(0, 2, 200)
X_new   = np.random.rand(5, 20)

#bulid the model
model= tf.keras.Sequential([
    layers.Dense(128,activation='relu',input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

#compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#Train
history = model.fit(
    X_train,y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

#Evaluate

loss,acc = model.evaluate(X_test,y_test)
print(f"Test accuracy: {acc:3f}")

#predict
predictions = model.predict(X_new)

#save & load
model.save('my_models.keras')
model = tf.keras.models.load_model('my_models.keras')