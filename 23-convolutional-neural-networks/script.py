
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# print(tf.__version__)

# Import the Training Set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    'dataset/training_set', # Folder Path
    target_size=(64, 64), # Smaller target size to predict patterns faster
    batch_size=32, # Number of images to be imported
    class_mode='binary')

# Import the Test Set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = train_datagen.flow_from_directory(
    'dataset/test_set', # Folder Path
    target_size=(64, 64), # Smaller target size to predict patterns faster
    batch_size=32, # Number of images to be imported
    class_mode='binary')

# Init the Convolutional Neural Network
cnn = tf.keras.models.Sequential()

# Add Convolutional Layer = 3x3
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Add Pooling Step = 2x2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Add Second Convolutional Layer = 3x3
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3))

# Add Second Pooling Step = 2x2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full Connection (Artificial Neural Network)
# Larger number for hidden neurons = 128 units
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output Layer = HARD CODED "activation" for Binary classification
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the CNN = HARD CODED "loss" for Binary classification
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on imported Training Set
# High Amount of Epochs due to Computer Vision Requirements
cnn.fit(x=training_set, validation_data=test_set, epochs=25)



# Make a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
# print(training_set.class_indices)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)