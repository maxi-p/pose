from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflowjs as tfjs
import os
import numpy as np


label_map = {'idle':0,'holding_controller':1, 'texting':2, 'stretch':3, 'left_raise_full':4, 'left_raise_half':5, 'right_raise_full':6, 'right_raise_half':7, 'left_hand_on_desk':8, 'right_hand_on desk':9, 'talking_left':10, 'talking_right':11}

# print(label_map)

# Path for exported data
DATA_PATH = os.path.join("MP_Data")

# Actions
actions = np.array(['idle','holding_controller', 'texting', 'stretch', 'left_raise_full', 'left_raise_half', 'right_raise_full', 'right_raise_half', 'left_hand_on_desk', 'right_hand_on desk', 'talking_left', 'talking_right'])

# Samples
no_sequences = 30

# Frames
sequence_length = 30

# Folder start
start_folder = 0

sequences, labels = [],[]
for action in actions:
	for sequence in range(no_sequences):
		window = []
		for frame_num in range(sequence_length):
			res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
			window.append(res)
		sequences.append(window)
		labels.append(label_map[action])
print(np.array(sequences).shape)

X = np.array(sequences)

y = to_categorical(labels).astype(int)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05)


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,56)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=500, callbacks=[tb_callback])
model.save('my_model.keras')
model.summary()
