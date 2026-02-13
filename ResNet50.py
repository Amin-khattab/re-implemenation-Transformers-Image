import keras
import tensorflow as tf
from tensorflow.keras import layers,models,mixed_precision
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from google.colab import files

policy = mixed_precision.Policy("mixed_bfloat16")
mixed_precision.set_global_policy(policy)

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode="nearest",
    rotation_range=10
)

test_datagen = ImageDataGenerator(rescale=1/255)

train_set = train_datagen.flow(x_train,y_train,batch_size=64)
test_set = test_datagen.flow(x_test,y_test,batch_size=64)

def build_pretrained_resnet():
    input_tensor = layers.Input(shape=(32,32,3))

    x = layers.UpSampling2D(size=(3,3))(input_tensor)

    base_model = ResNet50V2(include_top=False,weights="imagenet",input_tensor=x)
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=256,activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=10,activation="softmax",dtype="float32")(x)

    return models.Model(input_tensor,outputs)

resnet_pt = build_pretrained_resnet()

tracker = ReduceLROnPlateau(
    patience=3,
    monitor="val_accuracy",
    factor=0.5,
    verbose=1
)

early_stop = EarlyStopping(
    patience=10,
    verbose=1,
    monitor="val_accuracy",
    restore_best_weights=True
)

resnet_pt.compile(optimizer=keras.optimizers.Adam(1e-4),loss = "sparse_categorical_crossentropy",metrics=["accuracy"])

resnet_pt.fit(x=train_set,validation_data=test_set,callbacks=[early_stop,tracker],epochs=50)
