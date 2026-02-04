import matplotlib.pyplot as plt
import numpy as np
from transformer_model import TransformerBlock


def plot_attention(model, X_sample):
    for layer in model.layers:
        if isinstance(layer, TransformerBlock):
            _, scores = layer(X_sample, return_attention=True)
            attn = scores[0].numpy()
            plt.imshow(attn.mean(axis=0))
            plt.colorbar()
            plt.title("Attention Heatmap")
            plt.xlabel("Input Time Step")
            plt.ylabel("Input Time Step")
            plt.show()
            break

plot_attention(transformer_model, X_test[:1])

