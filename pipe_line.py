import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import keras_hub

# --- 1. CONFIGURATION ---
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
image_path = "test_images"  # The folder containing your local test images

# --- 2. CUSTOM ViT COMPONENTS (FROM SCRATCH) ---

class PatchEncoder(layers.Layer):
    def __init__(self, patch_size, projection_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = layers.Conv2D(
            filters=projection_dim, kernel_size=patch_size,
            strides=patch_size, padding="valid"
        )
        self.flatten = layers.Reshape(target_shape=(-1, projection_dim))

    def call(self, image):
        projected_patches = self.projection(image)
        return self.flatten(projected_patches)

class ViTEmbeddings(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.cls_token = tf.Variable(
            initial_value=tf.random.normal([1, 1, projection_dim]),
            trainable=True, name="cls_Token"
        )
        self.position_embedding = layers.Embedding(
            input_dim=1 + num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        batch_size = tf.shape(patches)[0]
        cls_token_cast = tf.cast(self.cls_token, patches.dtype)
        cls_tokens = tf.broadcast_to(cls_token_cast, [batch_size, 1, self.projection_dim])
        x = tf.concat([cls_tokens, patches], axis=1)
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        return x + self.position_embedding(positions)

class TransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads, mlp_dim):
        super().__init__()
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dense(projection_dim)
        ])

    def call(self, x):
        x1 = self.ln1(x)
        attention_output = self.attention(x1, x1)
        x2 = x + attention_output
        x3 = self.ln2(x2)
        mlp_output = self.mlp(x3)
        return mlp_output + x2

class ViTModel(tf.keras.Model):
    def __init__(self, num_classes, num_heads, patch_size, num_patches, projection_dim, mlp_dim, num_layers):
        super().__init__()
        self.patch_encoder = PatchEncoder(patch_size, projection_dim)
        self.embedding = ViTEmbeddings(num_patches, projection_dim)
        self.transformer_layers = [TransformerBlock(projection_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        self.ln = layers.LayerNormalization(epsilon=1e-6)
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, x):
        x = self.patch_encoder(x)
        x = self.embedding(x)
        for layer in self.transformer_layers: x = layer(x)
        return self.classifier(self.ln(x[:, 0, :]))

# --- 3. ARCHITECTURE BUILDERS ---

def build_simple_cnn():
    return models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPool2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPool2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPool2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

def build_custom_vit():
    model = ViTModel(num_classes=10, num_heads=4, patch_size=4, num_patches=64, projection_dim=64, mlp_dim=128, num_layers=4)
    model(tf.zeros((1, 32, 32, 3))) # Initialize weights
    return model

def build_resnet():
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.UpSampling2D(size=(3, 3))(inputs)
    base = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_tensor=x)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return models.Model(inputs, outputs)

def build_vit_boss():
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.UpSampling2D(size=(7, 7))(inputs)
    backbone = keras_hub.models.ViTBackbone.from_preset("vit_base_patch16_224_imagenet")
    x = backbone(x)[:, 0, :]
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return models.Model(inputs, outputs)

# --- 4. THE GAUNTLET RUNNER ---

model_gauntlet = {
    "Simple CNN": ("cnn_cifar10_weights.weights.h5", build_simple_cnn()),
    "Custom ViT (Scratch)": ("vit_cifar10_weights.weights.h5", build_custom_vit()),
    "ResNet-50": ("resnet_cifar10_breakthrough.weights.h5", build_resnet()),
    "ViT-B16 Boss": ("vit_final_boss.weights.h5", build_vit_boss())
}

for name, (weight_path, model) in model_gauntlet.items():
    print(f"\n--- Testing {name} ---")
    try:
        model.load_weights(weight_path)
        for file in os.listdir(image_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = image.load_img(os.path.join(image_path, file), target_size=(32, 32))
                x = image.img_to_array(img) / 255.0
                x = np.expand_dims(x, axis=0)
                score = model.predict(x, verbose=0)
                print(f"[{file}] -> {labels[np.argmax(score)]} ({np.max(score)*100:.1f}%)")
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
