#LSTM PROJECT: CAGLAR SUBASI, SUAT BULDANLIOGLU, ONUR KARAMAN, YAVUZ SELIM SEFUNC
# Within this file we'll analyse oil prices with LSTM model.
#importing library
import numpy as np
import pandas
import os
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model

np.set_printoptions(threshold=np.nan)

batch_size = 64 #number of batch size
epochs = 20 #number of epochs
timesteps = 10 # number of steps from past to predict future steps
loss_func = 'mae'

def get_train_length(dataset, batch_size, test_percent):
	# rebuild using end of data not start
	length = len(dataset)
	length *= 1 - test_percent
	train_length_values = []
	for x in range(int(length) - 100, int(length)):
		modulo = x % batch_size
		if(modulo == 0):
			train_length_values.append(x)
	return (max(train_length_values))

def get_test_length(dataset, batch_size, train_length):
	test_length_values = []
	for x in range(len(dataset) - 200, len(dataset) - timesteps * 2): 
		modulo = (x - train_length) % batch_size
		if(modulo == 0):
			test_length_values.append(x)
	return (max(test_length_values))

def build_model(batch_size, timesteps):
	# Initilize LSTM model with MAE Loss-Function
	inputs = Input(batch_shape=(batch_size, timesteps, 1))
	lstm_1 = LSTM(25, stateful=True, return_sequences=True)(inputs)
	lstm_2 = LSTM(25, stateful=True, return_sequences=True)(lstm_1)
	output = Dense(units=1)(lstm_2)

	regressor = Model(inputs=inputs, outputs=output)
	regressor.compile(optimizer='adam', loss=loss_func)
	regressor.summary()

	return regressor

def get_input_output_data(dataset, length, timesteps):
	# move data for input back = timesteps and output forward = timesteps
	X, y = [], []
	for i in range(timesteps, length + timesteps):
		X.append(dataset[i - timesteps:i, 0])
		y.append(dataset[i:i + timesteps, 0])

	return np.array(X), np.array(y)

#please the BrentOil.csv data
data = pandas.read_csv(os.path.join(os.path.dirname(__file__),"/Users/yavuzselimsefunc/Downloads/BrentOil.csv"))

# set train and test lengths
train_length = get_train_length(data, batch_size, 0.3)

# Adding timesteps
upper_train = train_length + timesteps * 2

test_length = get_test_length(data, batch_size, upper_train)
upper_test = test_length + timesteps * 2
testset_length = test_length - upper_train
print(upper_train, testset_length, len(data))

#set train and test datasets
data_train = data[0:upper_train]
training_set = data_train.iloc[:,1:2].values

data_test = data[upper_train:upper_test] 
test_set = data_test.iloc[:,1:2].values.astype('float32')

#Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(np.float64(training_set))
test_set_scaled = scaler.fit_transform(np.float64(test_set))

X_train, y_train = get_input_output_data(training_set_scaled, train_length, timesteps)

X_test, unused_y_test = get_input_output_data(test_set_scaled, testset_length, timesteps)

# Reshape data

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = build_model(batch_size, timesteps)

# Train model

for i in range(epochs):
	print(i + 1 ," Epoch")
	model.fit(X_train, y_train, shuffle=False, epochs=1, batch_size=batch_size)
	model.reset_states()


# predict 
predicted_data = model.predict(X_test, batch_size=batch_size)
model.reset_states()
print(predicted_data.shape)

# reshaping predicted
predicted_data = np.reshape(predicted_data, (predicted_data.shape[0], predicted_data.shape[1]))
print(predicted_data.shape)

# inverse transform
predicted_data = scaler.inverse_transform(predicted_data)

# creating y_test data
y_test = []
for j in range(0, len(predicted_data)):
    y_test = np.append(y_test, predicted_data[j, timesteps - 1])

# reshaping
y_test = np.reshape(y_test, (y_test.shape[0], 1))

show_y_test = np.reshape(y_test, (y_test.shape[0],))
show_test_dataset = data[upper_train + timesteps:].iloc[:,1:2].values.astype('float32')

X_test_show = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
X_test_show = scaler.inverse_transform(X_test_show)

# Rebuild actual X_test data to see the difference
X_test_c = []
for j in range(0, len(X_test_show)):
    X_test_c = np.append(X_test_c, X_test_show[j, timesteps - 1])

#CONCLUSION
# As a result, we've get a MAE score of 0.0695 which is very good. We can predict
# brent oil prices with an error less than a dollar. 
