from tensorflow.keras import layers, models

# VANILLA NN
def create_NN_model():
  model = models.Sequential()
  #model.add(layers.Normalization())
  model.add(layers.Dense(84, activation='relu'))
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dropout(0.3))
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dropout(0.3))
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dropout(0.2))
  model.add(layers.Dense(32, activation='relu'))
  model.add(layers.Dense(29, activation='softmax'))

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model