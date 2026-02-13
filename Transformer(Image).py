import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    rescale=1 / 255
)

test_datagen = ImageDataGenerator(rescale=1 / 255)

test_generator = test_datagen.flow(x_test, y_test, batch_size=64)
train_generator = train_datagen.flow(x_train, y_train, batch_size=64)

# 2. Hyperparameters
input_shape = (32, 32, 3)
num_classes = 10
patch_size = 4
num_patches = (input_shape[0] // patch_size) ** 2  # 64 patches
projection_dim = 64
num_heads = 4
mlp_dim = 128
num_layers = 4


# Step 1: Patching & Projection
# first step we Convulate the Image into patch's then into a series of words
class PatchEncoder(layers.Layer):
    def __init__(self, patch_size, projection_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = layers.Conv2D(
            filters=projection_dim,
            kernel_size=patch_size,  # we make the conv2D the kernel and strides the same number because
            strides=patch_size,  # we want it to take 16 pixels lets say and then jump to the other patch of 16
            padding="valid"
        )
        self.flatten = layers.Reshape(target_shape=(-1, projection_dim))  # then flattening it to make it like words so
        # that the transformer can read it as a sentence

    def call(self, image):
        projected_patches = self.projection(image)
        return self.flatten(projected_patches)


# Step 2: CLS Token & Positional Embeddings
class ViTEmbeddings(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.cls_token = tf.Variable(
            initial_value=tf.random.normal([1, 1, projection_dim]),
            # so here we only create one cls thats why we only give it to 1 image
            trainable=True,
            name="cls_Token"
        )
        self.position_embedding = layers.Embedding(
            input_dim=1 + num_patches,
            output_dim=projection_dim
        )

    def call(self, patches):
        batch_size = tf.shape(patches)[0]

        cls_token_cast = tf.cast(self.cls_token, patches.dtype)

        cls_tokens = tf.broadcast_to(
            cls_token_cast, [batch_size, 1, self.projection_dim]
            # here we apply the cls token to be copied to the number of bach_size
        )
        # cls token dim = (batch_size,1,projection_dim)
        # patches_dim = (batch_196,dim)
        # by setting the axis to 1 the cls token becomes the first of the order
        x = tf.concat([cls_tokens, patches], axis=1)

        positions = tf.range(start=0, limit=self.num_patches + 1,
                             delta=1)  # we make this to go from 1-197/ 0 is the cls token

        return x + self.position_embedding(positions)


# Step 3: Transformer Block
class TransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads, mlp_dim):
        super().__init__()
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim
        )
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dense(projection_dim)
        ])

    def call(self, x):
        # residual Connection
        x1 = self.ln1(x)
        attention_output = self.attention(x1, x1)
        x2 = x + attention_output

        # residual Connection
        x3 = self.ln2(x2)
        mlp_output = self.mlp(x3)
        return mlp_output + x2


# Final Model Assembly
class ViTModel(tf.keras.Model):
    def __init__(self, num_classes, num_heads, patch_size, num_patches, projection_dim, mlp_dim, num_layers):
        super().__init__()
        self.patch_encoder = PatchEncoder(patch_size, projection_dim)
        self.embedding = ViTEmbeddings(num_patches, projection_dim)
        self.transformer_layers = [
            TransformerBlock(projection_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ]
        self.ln = layers.LayerNormalization(epsilon=1e-6)
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, x):
        x = self.patch_encoder(x)
        x = self.embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)

        cls_output = x[:, 0, :]
        return self.classifier(self.ln(cls_output))


# 3. Create Model
model = ViTModel(
    num_classes=num_classes,
    num_heads=num_heads,
    patch_size=patch_size,
    num_patches=num_patches,
    projection_dim=projection_dim,
    mlp_dim=mlp_dim,
    num_layers=num_layers
)

tracker = ReduceLROnPlateau(
    monitor="val_loss",
    patience=3,
    min_lr=0.0001,
    factor=0.5,
    verbose=1
)

early_stoping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 4. Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Train
print("Starting training on CIFAR-10...")
model.fit(train_generator, epochs=100, validation_data=test_generator,callbacks=[tracker,early_stoping])
