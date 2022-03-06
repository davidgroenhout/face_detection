# -*- coding: utf-8 -*-
import pathlib
import tensorflow
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

data_set = tensorflow.keras.utils.image_dataset_from_directory(
    pathlib.Path('./images'), labels='inferred')
class_names = data_set.class_names
normalization_layer = tensorflow.keras.layers.Rescaling(1./255)
model = tensorflow.keras.Sequential([
  tensorflow.keras.layers.Rescaling(1./255),
  tensorflow.keras.layers.Conv2D(32, 3, activation='relu'),
  tensorflow.keras.layers.MaxPooling2D(),
  tensorflow.keras.layers.Conv2D(32, 3, activation='relu'),
  tensorflow.keras.layers.MaxPooling2D(),
  tensorflow.keras.layers.Conv2D(32, 3, activation='relu'),
  tensorflow.keras.layers.MaxPooling2D(),
  tensorflow.keras.layers.Flatten(),
  tensorflow.keras.layers.Dense(128, activation='relu'),
  tensorflow.keras.layers.Dense(2)
])
model.compile(
  optimizer='adam',
  loss=tensorflow.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
model.fit(
  data_set,
  epochs=3
)
model.save('model')
full_model = tensorflow.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tensorflow.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
)
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

print(f"Model inputs: {frozen_func.inputs}")
print(f"Model outputs: {frozen_func.outputs}")

tensorflow.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir='./model',
                  name='liveness.pb',
                  as_text=False)