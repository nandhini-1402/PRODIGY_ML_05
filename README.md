import tensorflow_datasets as tfds
dataset_list=tfds.list_builders()

print("food101" in dataset_list)
(train_data,test_data),ds_info=tfds.load(name='food101',
                                         split=["train","validation"],
                                         shuffle_files=True,
                                         as_supervised=True,  # give in tuple (data,label)
                                         with_info=True)  
# meta data
ds_info.features
# class names of data
class_names=ds_info.features["label"].names
class_names[:10]
# minimum and maximum pixel value

import tensorflow as tf

tf.reduce_min(x[0]),tf.reduce_max(x[0])

# preprocessing function

def preprocess_img(image,label,image_shape=224):
  """ convert image dtype from unit8 to float32,reshapes 
      image to [image_shape,image_shape,color_channel]
  """
  image=tf.image.resize(image,[image_shape,image_shape]) # target image
  return tf.cast(image,tf.float32),label
sample_image=x[0]
preprocessed_image=preprocess_img(x[0],x[1])[0]

print("Before preprocessing...")
print(f"""
sample image shape : {sample_image.shape}
sample image dtype : {sample_image.dtype}
""")

print("After preprocessing...")
print(f"""
preprocessed image shape : {preprocessed_image.shape}
preprocessed image dtype : {preprocessed_image.dtype}
""")
# map preprocess function to training data (by parallel processing)
train_data=train_data.map(map_func=preprocess_img,num_parallel_calls=tf.data.AUTOTUNE)

# shuffle our training data then make batches and prefetch it for faster to load
train_data=train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(tf.data.AUTOTUNE)

# now same process for testing data
test_data=test_data.map(map_func=preprocess_img,num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=32).prefetch(buffer_size=1000)
train_data,test_data

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

from tensorflow.keras import layers 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental import preprocessing
len(class_names)

Data_augumentaion=tf.keras.Sequential([
  preprocessing.RandomFlip('horizontal'),
  preprocessing.RandomRotation(0.2),
  preprocessing.RandomZoom(0.2),
  preprocessing.RandomHeight(0.2),
  preprocessing.RandomWidth(0.2)
],name="data_augumentation")

base_model=tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable=False

input=layers.Input(shape=(224,224,3),name='input_layer')
x=Data_augumentaion(input)
x=base_model(x,training=False)

x=layers.GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
x=layers.Dense(len(class_names))(x)
output=layers.Activation(activation='softmax',dtype=tf.float32,name='softmax_mixed_output_layer')(x)

model=tf.keras.Model(input,output)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()
for layer in model.layers:
  print(layer.name,layer.trainable,layer.dtype,layer.dtype_policy)

history_101=model.fit(train_data,
                      epochs=3,  # prefer 3 epochs ,over it overfitting occur
                      steps_per_epoch=len(train_data),
                      validation_data=test_data,
                      validation_steps=int(0.15*len(test_data)),
                      )
model.evaluate(test_data)

# unfreeze some layers
base_model.trainable=True

for x in base_model.layers[:-25]:
  x.trainable=False


initial_epoch=history_101.epoch

fine_tune_epoch=len(initial_epoch)+2

history_101_fine_tune=model.fit(train_data,
                    epochs=fine_tune_epoch,
                    validation_data=test_data,
                    validation_steps=int(0.15*len(test_data)),
                    initial_epoch=len(initial_epoch)-1)

initial_epoch=history_101_fine_tune.epoch

fine_tune_epoch=len(initial_epoch)+5

history=model.fit(train_data,
                    epochs=fine_tune_epoch,
                    validation_data=test_data,
                    validation_steps=int(0.15*len(test_data)),
                    initial_epoch=len(initial_epoch)-1)
model.evaluate(test_data)
tf.keras.models.save_model(model,"fine_tune_model_1.h5")

tf.keras.models.save_model(model,"/content/drive/MyDrive/Colab Notebooks/models/food_vision_fine_tune_model_1.h5")
Save Model
loaded_model=tf.keras.models.load_model('/content/07_efficientnetb0_fine_tuned_101_classes_mixed_precision')
loaded_model.evaluate(test_data)# PRODIGY_ML_05
