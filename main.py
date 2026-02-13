import keras_hub
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from tensorflow.keras import mixed_precision,layers,models
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
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

def vit_pre_trained():
    tensor_input = layers.Input(shape=(32,32,3))

    x = layers.UpSampling2D(size=(7,7))(tensor_input)

    vit_backbone = keras_hub.models.ViTBackbone.from_preset(
        "vit_base_patch16_224_imagenet",
        load_weights=True
    )
    vit_backbone.trainable = False

    x = vit_backbone(x)

    # Grab the [CLS] token (represents the whole image)
    cls_token = x[:,0,:]

    x = layers.BatchNormalization()(cls_token)

    x = layers.Dense(units=256,activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=10,activation="softmax",dtype = "float32")(x)

    return models.Model(tensor_input,outputs)

trained_vit = vit_pre_trained()

tracker = ReduceLROnPlateau(
    monitor = "val_accuracy",
    patience = 3,
    factor = 0.5,
    min_lr = 1e-6,
    verbose = 1
)

early_stoping = EarlyStopping(
    monitor = "val_accuracy",
    patience = 10,
    restore_best_weights=True,
    verbose = 1
)

trained_vit.compile(optimizer = tf.keras.optimizers.Adam(1e-4),loss = "sparse_categorical_crossentropy",metrics=["accuracy"])

trained_vit.fit(epochs = 75,x = train_set,validation_data=test_set,callbacks = [early_stoping,tracker])
