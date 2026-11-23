"""Train chest X-ray classifier (MobileNetV2 transfer learning - light enough for low-end hardware).
Dataset: Kaggle "Chest X-Ray Images (Pneumonia)" laid out as
data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG = (224, 224); BS = 32; ROOT = "data/chest_xray"

def load(split, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        f"{ROOT}/{split}", image_size=IMG, batch_size=BS, label_mode="binary", shuffle=shuffle)
    return ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)

train, val, test = load("train", True), load("val", False), load("test", False)

aug = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.05),
                           layers.RandomZoom(0.1)])
base = MobileNetV2(input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False
x = layers.GlobalAveragePooling2D()(base.output)   # not nested -> Grad-CAM can reach 'out_relu'
x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation="sigmoid")(x)
model = Model(base.input, out)

# class weights (dataset is imbalanced)
n = {c: len(tf.io.gfile.listdir(f"{ROOT}/train/{c}")) for c in ["NORMAL", "PNEUMONIA"]}
tot = sum(n.values())
cw = {0: tot / (2 * n["NORMAL"]), 1: tot / (2 * n["PNEUMONIA"])}

train_aug = train.map(lambda x, y: (aug(x, training=True), y))
model.compile("adam", "binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")])
model.fit(train_aug, validation_data=val, epochs=5, class_weight=cw)

base.trainable = True                                # fine-tune top layers
for l in base.layers[:-30]: l.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-5), "binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")])
model.fit(train_aug, validation_data=val, epochs=5, class_weight=cw)

print("TEST:", dict(zip(model.metrics_names, model.evaluate(test))))
model.save("models/xray_model.keras")
