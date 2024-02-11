import tensorflow as tf
from tensorflow.keras import layers, models



def create_NN_model():
  model = models.Sequential()
  model.add(layers.Dense(84, activation='relu'))
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dropout(0.2))
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dropout(0.2))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(29, activation='softmax'))

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

# model.fit(X_train, y_train, epochs=10)

# model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

#test_loss, test_acc = model.evaluate(X_test, y_test)
#print(f'Test accuracy: {test_acc}')

