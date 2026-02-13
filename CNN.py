import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping,ReduceLROnPlateau

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(
    rescale=1/255
)

train_set =train_datagen.flow(
    x_train,
    y_train,
    batch_size=64
    )

test_set = test_datagen.flow(
    x_test,
    y_test,
    batch_size=64
)

tracker = ReduceLROnPlateau(
    monitor="val_accuracy",
    patience=3,
    min_lr=0.0001,
    factor=0.5,
    verbose=1
)

early_stoping = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

cnn =keras.Sequential()

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=32,padding="same",activation=None,input_shape=(32,32,3)))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(strides=2,pool_size=2))

cnn.add(keras.layers.Conv2D(kernel_size=3,padding="same",filters=64,activation=None))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(strides=2,pool_size=2))

cnn.add(keras.layers.Conv2D(kernel_size=3,padding="same",filters=128,activation=None))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(strides=2,pool_size=2))

cnn.add(keras.layers.GlobalAveragePooling2D())

cnn.add(keras.layers.Dense(units=256,activation="relu"))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Dropout(0.25))

cnn.add(keras.layers.Dense(units=10,activation="softmax"))

cnn.compile(optimizer="adam",metrics=["accuracy"], loss = "sparse_categorical_crossentropy")

cnn.fit(x = train_set,validation_data=test_set,callbacks=[tracker,early_stoping])
