import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import batch_generator

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_df = pd.read_csv('data/driving_log.csv',
                      names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

data_df['center'] = data_df['center'].apply(lambda x: 'data/IMG/' + x.split('\\')[-1])
data_df['left'] = data_df['left'].apply(lambda x: 'data/IMG/' + x.split('\\')[-1])
data_df['right'] = data_df['right'].apply(lambda x: 'data/IMG/' + x.split('\\')[-1])

X = data_df[['center', 'left', 'right']].values
y = data_df['steering'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)


def createModel():
    net = tf.keras.modelsSequential()

    net.add(tf.keras.layers.Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    net.add(tf.keras.layers.Conv2D(24, (5, 5), (2, 2), activation='elu'))
    net.add(tf.keras.layers.Conv2D(36, (5, 5), (2, 2), activation='elu'))
    net.add(tf.keras.layers.Conv2D(48, (5, 5), (2, 2), activation='elu'))
    net.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu'))
    net.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu'))

    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(100, activation='elu'))
    net.add(tf.keras.layers.Dense(50, activation='elu'))
    net.add(tf.keras.layers.Dense(10, activation='elu'))
    net.add(tf.keras.layers.Dense(1, activation='tanh'))

    net.compile(tf.keras.optimizers.Adam(lr=1.0e-4), loss='mse')
    return net


checkpoint = tf.keras.callbacks.ModelCheckpoint('model-{epoch:03d}.h5',
                                                monitor='val_loss',
                                                verbose=0,
                                                save_best_only=True,
                                                mode='auto')

model = createModel()
model.fit_generator(batch_generator(X_train, y_train, 40, True),
                    steps_per_epoch=2000,
                    validation_data=batch_generator(X_valid, y_valid, 40, False),
                    validation_steps=10,
                    epochs=50,
                    callbacks=[checkpoint]
                    )
