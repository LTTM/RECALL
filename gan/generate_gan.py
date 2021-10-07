import tensorflow as tf
import tensorflow_hub as hub
import matplotlib
from tqdm import tqdm
import os

# classes to generate
FROM_CLASS = 0
TO_CLASS = 999
NUMBER_OF_CLASSES = TO_CLASS - FROM_CLASS
OUTPUT_FOLDER = "image_all/"
NUMBER_OF_BATCHES = 100
BATCH_SIZE = 5

# convert image data type
def im2int(image):
    return tf.image.convert_image_dtype((image + 1)/2, dtype=tf.uint8)

# Load BigGAN-deep 512 module.
module = hub.Module("module/")

# Define the model
inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
          for k, v in module.get_input_info_dict().items()}

input_z = inputs['z']
input_y = inputs['y']
input_trunc = inputs['truncation']

output = im2int(module(inputs))

truncation = 1.0  # scalar truncation value in [0.0, 1.0]
z = truncation * tf.random.truncated_normal([BATCH_SIZE, 128])  # noise sample


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for c in range(NUMBER_OF_CLASSES+1):

        # get current class number
        class_number = FROM_CLASS + c

        # create output folder for current class
        output_folder = OUTPUT_FOLDER + str(class_number).zfill(4) + "/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # label of current class
        y_index = [class_number]*BATCH_SIZE
        y = tf.one_hot(y_index, 1000)  # one-hot ImageNet label

        print("Generating images of class ", class_number)

        for j in tqdm(range(NUMBER_OF_BATCHES)):

            z_ = sess.run(z)
            y_ = sess.run(y)
            samples_ = sess.run(output, feed_dict={input_z: z_, input_y: y_, input_trunc: truncation})

            for i in range(BATCH_SIZE):
                matplotlib.image.imsave(output_folder + str(j * BATCH_SIZE + i).zfill(3) + ' .jpg', samples_[i, :, :, :])




