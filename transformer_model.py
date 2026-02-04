import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_len, d_model):
        super().__init__()
        pos = np.arange(sequence_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.cast(angle_rads[np.newaxis, ...], tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, : tf.shape(x)[1], :]


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False, return_attention=False):
        attn_output, scores = self.att(
            inputs, inputs, return_attention_scores=True
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        if return_attention:
            return out2, scores
        return out2


def build_transformer(input_shape, future_steps):
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(64)(inputs)
    x = PositionalEncoding(input_shape[0], 64)(x)

    transformer_block = TransformerBlock(64, 4, 128)
    x = transformer_block(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(future_steps * input_shape[-1])(x)
    outputs = layers.Reshape((future_steps, input_shape[-1]))(x)

    model = tf.keras.Model(inputs, outputs)
    return model
