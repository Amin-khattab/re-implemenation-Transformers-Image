import tensorflow as tf
from tensorflow.keras import layers


# first step we Convulate the Image into patch's then into a series of words
class PatchEncoder(layers.Layer):
    def __init__(self,patch_size,projection_dim):
        super().__init__()
        self.patch_size = patch_size

        self.projection = layers.Conv2D(
            filters = projection_dim,
            kernel_size = patch_size,# we make the conv2D the kernel and strides the same number because
                                    # we want it to take 16 pixels lets say and then jump to the other patch of 16
            strides = patch_size,
            padding = "valid"
        )

        self.flatten = layers.Reshape(target_shape=(-1,projection_dim))# then flattening it to make it like words so
                                                                       # that the transformer can read it as a sentence

    def call(self,image):
        projected_patchs = self.projection(image)

        return self.flatten(projected_patchs)

class ViTEmbeddings(layers.Layer):
    def __init__(self,num_patchs,projection_dim):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_patchs = num_patchs

        self.cls_token = tf.Variable(
            initial_value=tf.random.normal([1,1,projection_dim]),# so here we only create one cls thats why we only give it to 1 image
            trainable=True,
            name = "cls_Token"
        )

        self.position_embedding = layers.Embedding(
            input_dim=1+num_patchs,
            output_dim=projection_dim
        )

    def call(self,patches):
        batch_size = tf.shape(patches)[0]
        cls_tokens = tf.broadcast_to(
            self.cls_token,[batch_size,1,self.projection_dim] # here we apply the cls token to be copied to the number of bach_size
        )
                                                #cls token dim = (batch_size,1,projection_dim)
                                                #patches_dim = (batch_196,dim)
                                                # by setting the axis to 1 the cls token becomes the first of the order
        x = tf.concat([cls_tokens,patches],axis=1)

        positions = tf.range(start=0,limit=self.num_patchs + 1,delta=1) # we make this to go from 1-197/ 0 is the cls token

        return x + self.position_embedding(positions)

class TrasnformerBlock(layers.Layer):
    def __init__(self,projection_dim,num_heads,mlp_dim):
        super().__init__()

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,key_dim=projection_dim
        )

        self.mlp = tf.keras.Sequential([
        layers.Dense(mlp_dim,activation=tf.nn.gelu),
        layers.Dense(projection_dim)
        ])

    def call(self,x):
        #residual Connection

        x1 = self.ln1(x)
        attention_output = self.attention(x1,x1)
        x2 = x + attention_output

        # residual Connection

        x3 = self.ln2(x2)
        mlp_output = self.mlp(x3)
        return mlp_output + x2

class ViTModel(tf.keras.Model):
    def __init__(self,num_classes,num_heads,patch_size,num_patchs,projection_dim,mlp_dim,num_layers):
        super().__init__()

        self.patch_encoder = PatchEncoder(patch_size,projection_dim)
        self.embedding = ViTEmbeddings(num_patchs,projection_dim)

        self.transfomer_layer = [
            TrasnformerBlock(projection_dim,num_heads,mlp_dim)
            for _ in range(num_layers)
        ]

        self.ln = layers.LayerNormalization(epsilon=1e-6)
        self.classifier = layers.Dense(num_classes,activation="softmax")

    def call(self,x):
        x = self.patch_encoder(x)

        x = self.embedding(x)

        for layer in self.transfomer_layer:
            x = layer(x)


        cls_output = x[:,0,:]

        return self.classifier(self.ln(cls_output))

model = ViTModel(
    num_heads=4,
    num_layers=4,
    num_patchs=196,
    num_classes=10,
    projection_dim=64,
    patch_size=16,
    mlp_dim=128
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit()
