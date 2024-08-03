import tensorflow as tf
from keras.models import load_model

model = load_model("SIH_Translator_pickle.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

lite_model = converter.convert()

with open("SIH_Translator_tflite.tflite","wb") as f:
    f.write(lite_model)