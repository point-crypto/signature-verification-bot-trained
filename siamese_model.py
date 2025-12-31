import tensorflow as tf
from tensorflow.keras import layers, Model

def build_siamese():
    def base():
        inp = layers.Input(shape=(150,300,1))
        x = layers.Conv2D(32,(3,3),activation='relu')(inp)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64,(3,3),activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        return Model(inp, x)

    a = layers.Input((150,300,1))
    b = layers.Input((150,300,1))

    base_net = base()
    fa = base_net(a)
    fb = base_net(b)

    dist = layers.Lambda(lambda x: tf.abs(x[0]-x[1]))([fa, fb])
    out = layers.Dense(1, activation='sigmoid')(dist)

    model = Model([a,b], out)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
