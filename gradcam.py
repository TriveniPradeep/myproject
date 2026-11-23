import numpy as np, tensorflow as tf
from PIL import Image
from matplotlib import cm
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def prep(img: Image.Image):
    arr = np.array(img.convert("RGB").resize((224, 224)), dtype="float32")
    return arr, preprocess_input(arr.copy())[None]

def gradcam(model, x, layer="out_relu"):
    gm = tf.keras.Model(model.input, [model.get_layer(layer).output, model.output])
    with tf.GradientTape() as tape:
        conv, pred = gm(x)
        score = pred[:, 0]
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8)

def overlay(arr, cam, alpha=0.4):
    heat = Image.fromarray((cm.jet(np.array(Image.fromarray((cam * 255).astype("uint8")).resize((224, 224)))
                                   / 255.0)[..., :3] * 255).astype("uint8"))
    return Image.blend(Image.fromarray(arr.astype("uint8")), heat, alpha)
