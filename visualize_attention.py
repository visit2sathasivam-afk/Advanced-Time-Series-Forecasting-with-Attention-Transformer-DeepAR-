import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess


X_train, X_test, y_train, y_test, _ = load_and_preprocess()
model = tf.keras.models.load_model("transformer_model.keras", compile=False)

sample = X_test[:1]
_ = model(sample)

scores = model.attention_scores.numpy()[0].mean(axis=0)

plt.imshow(scores)
plt.colorbar()
plt.title("Attention Heatmap")
plt.xlabel("Time Step")
plt.ylabel("Time Step")
plt.show()
