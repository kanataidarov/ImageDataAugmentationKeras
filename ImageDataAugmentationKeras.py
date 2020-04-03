# Image Data Augmentation with Keras
import os 
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt 

# Augmentation 
generator1 = tf.keras.preprocessing.image.ImageDataGenerator(
    # rotation_range=40,
    # width_shift_range=[-99, -66, -33, 0, 44, 66, 99], 
    # height_shift_range=[-66, -33, 0, 44, 66],
    # brightness_range=(0.5, 2),
    # shear_range=45,
    # zoom_range=[1.1, 2.5],
    # channel_shift_range=100,
    # horizontal_flip=True,
    vertical_flip=True
)

image_root_dir = 'oxford-iiit-pet-dataset'
image_cats_dir = image_root_dir+'/train/cats/'
image_path = image_cats_dir+'01-Egyptian_Mau_149.jpg'

# plt.imshow(plt.imread(image_path))
# plt.show()

x1, y1 = next( generator1.flow_from_directory(image_root_dir, shuffle=False, batch_size=1) )
plt.imshow(x1[0].astype('uint8'))
plt.show()

print( x1.mean() )
print( np.array(Image.open(image_path)).mean() )


# Normalization
(x2_train, y2_train), (x2_test, y2_test) = tf.keras.datasets.cifar10.load_data()
'''
Featurewise normalization
x2_mean = x2_train.mean()
x2_std = x2_train.std()
x2_train_norm = (x2_train-x2_mean)/x2_std 
'''
generator2 = tf.keras.preprocessing.image.ImageDataGenerator(
    # featurewise_center=True,                                    # mean will be extracted from dataset
    # featurewise_std_normalization=True,                         # everything will be divided by std value
    # samplewise_center=True,                                   
    # samplewise_std_normalization=True,
    rescale=1/255.,                                                # divide all elems of dataset to 255.
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)
generator2.fit(x2_train)

x2, y2 = next(generator2.flow(x2_train, y2_train, batch_size=1))
print(x2.mean(), x2.std(), y2)
print(x2_train.mean())

generator3 = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    horizontal_flip=True,
    rotation_range=20
)

model = tf.keras.models.Sequential([
    tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=False,
        input_shape=(32, 32, 3),
        pooling='Avg'
    ),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(
    generator3.flow(x2_train, y2_train, batch_size=32),
    epochs=1,
    steps_per_epoch=10
)
