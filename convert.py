import tensorflow as tf

# convert the model
converter = tf.lite.TFLiteConverter.from_keras_model("mask_detector.model")  # path to the SavedModel directory
tflite_model = converter.convert()

# save the model.
with open('mask_detector.tflite', 'wb') as f:
    f.write(tflite_model)
