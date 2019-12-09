import pathlib
import numpy as np

import tensorflow as tf

cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)

model = tf.keras.applications.MobileNet(input_shape=(32,32,3), weights=None, classes=10)
print(model.summary())

model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1, batch_size=512, validation_data=(x_test, y_test))

#model.evaluate(x_test,  y_test, verbose=2)
import time
times = list()
for i in range (1000):
    start = time.monotonic()
    model.predict(x_train[i:i+1])
    inference_time = time.monotonic() - start
    times.append(inference_time * 1000)
runtimes = np.array(times)
print("GPU | MobileNet | Mean Runtime:", np.mean(runtimes))

start = time.monotonic()
model.predict(x_train[0:1000])
inference_time = time.monotonic() - start
print("GPU | MobileNet | Single Pass Runtime (BS=32):", inference_time)

start = time.monotonic()
model.predict(x_train[0:1000], batch_size=1)
inference_time = time.monotonic() - start
print("GPU | MobileNet | Single Pass Runtime (BS=1):", inference_time)


# Save the model into SaveModel format

saved_model_dir = pathlib.Path("./saved_model/")
tf.saved_model.save(model, str(saved_model_dir))


# Convert the model from saved model
images = tf.cast(x_train, tf.float32)
cifar10_ds = tf.data.Dataset.from_tensor_slices(images).batch(1)


def representative_dataset_gen():
    for input_value in cifar10_ds.take(100):
        yield [input_value]


converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen

tflite_quant_model = converter.convert()

tflite_quant_model_file = saved_model_dir/"cifar10_post_quant_model_io.tflite"
tflite_quant_model_file.write_bytes(tflite_quant_model)