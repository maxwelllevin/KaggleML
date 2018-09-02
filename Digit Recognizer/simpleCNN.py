import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

# Here we create a sequential convolutional neural network. Keras takes care of most of the details for us.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Now we tell the model what its goal is. We use RMS prop to update the weights (like gradient descent with momentum)
# We use categorical crossentropy because we seek to have our model classify x_train to distinct categories.
# We set our metric to 'accuracy' so that we reduce the categorical crossentropy as training progresses.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Here we import the data using pandas and reshape it for our model using numpy.
raw_train = pd.read_csv('train.csv')
y_train = to_categorical(raw_train['label'].values)
raw_train = raw_train.drop('label', axis=1)
x_train = raw_train.values.reshape(-1, 28, 28, 1)
x_test = pd.read_csv('test.csv').values.reshape(-1, 28, 28, 1)

# Now we perform the training process.
model.fit(x_train, y_train, epochs=20, batch_size=512, verbose=2)

# We make predictions on the test data and save our predictions to a csv file for evaluation.
predictions = model.predict_classes(x_test)
pred_df = pd.DataFrame({'Label': predictions})
pred_df.index += 1
pred_df.to_csv('pred.csv', index_label='ImageId')
