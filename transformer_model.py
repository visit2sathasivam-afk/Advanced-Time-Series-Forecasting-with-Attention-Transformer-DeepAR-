import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pos = np.arange(seq_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle = pos / np.power(10000, (2*(i//2))/np.float32(d_model))
        angle[:, 0::2] = np.sin(angle[:, 0::2])
        angle[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = tf.constant(angle[None, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]


class EncoderBlock(layers.Layer):
    def __init__(self, d_model, heads, ff):
        super().__init__()
        self.mha = layers.MultiHeadAttention(heads, d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff, activation="relu"),
            layers.Dense(d_model)
        ])
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x):
        attn, scores = self.mha(x, x, return_attention_scores=True)
        x = self.ln1(x + attn)
        ffn = self.ffn(x)
        return self.ln2(x + ffn), scores


def build_transformer(input_shape, future_steps):
    inp = layers.Input(shape=input_shape)

    x = layers.Dense(64)(inp)
    x = PositionalEncoding(input_shape[0], 64)(x)

    enc, scores = EncoderBlock(64, 4, 128)(x)

    x = layers.GlobalAveragePooling1D()(enc)
    x = layers.Dense(future_steps * input_shape[-1])(x)
    out = layers.Reshape((future_steps, input_shape[-1]))(x)

    model = tf.keras.Model(inp, out)
    model.attention_scores = scores
    return model
