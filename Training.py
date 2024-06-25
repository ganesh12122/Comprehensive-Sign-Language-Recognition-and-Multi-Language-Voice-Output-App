from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential

# Path for exported data, numpy arrays
DATA_PATH = os.path.join(os.getcwd(), 'Data')

actions = np.array(['Hello', 'Love', 'No', 'None', 'Thankyou'])

sequences, labels = [], []
for idx, action in enumerate(actions):
    for sequence in range(1, 121):
        sequence_keypoints = np.load(os.path.join(DATA_PATH, action, f"sequence_{sequence}.npy"))
        sequences.append(sequence_keypoints)
        labels.append(idx)

X = np.array(sequences)
y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Define and compile your model with the custom optimizer
model = Sequential()
model.add(TimeDistributed(Dense(128), input_shape=(
    
    X_train.shape[1], X_train.shape[2])))
model.add((LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add((LSTM(256, return_sequences=True)))
model.add(Dropout(0.5))
model.add((LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add((LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(len(actions), activation='softmax'))

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy', 'categorical_accuracy'])

# Learning Rate Scheduling and Early Stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)
early_stop = EarlyStopping(
    monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Now, you can train your model
history = model.fit(X_train, y_train, epochs=50, batch_size=64,
                    validation_data=(X_test, y_test),
                    callbacks=[reduce_lr, early_stop])

model.save('action_detection_model.h5')
