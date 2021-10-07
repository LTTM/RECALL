import tensorflow as tf

import numpy as np
from tqdm import tqdm
import preprocessing as pp
import os


# global variables
image_size = 321
dataset_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


# Functions to create tfrecords.

def tfrecord_from_paths(info_file, output, include_labels=False):
    """
    Generate and save a tfrecord file starting from an info txt file containing all the paths of samples to add.
    The info txt file has an entry in each line and it is in the form: IMAGE_PATH LABEL_PATH.

    :param info_file: the path of a txt info file containing, in each line, the path of a sample to add in the
    tfrecord.
    :param output: the output name of the tfrecord. It can contain the extension ".tfrecord" but it is not mandatory.
    :param include_labels: if False just the images are added to the tfrecord. If True both the images and the
    labels are added to the tfrecord.
    """

    # if the extension is not .tfrecord, add the extension.
    filename, file_extension = os.path.splitext(output)
    if file_extension != ".tfrecord":
        output = output + ".tfrecord"

    # open the TFRecord writer
    writer = tf.io.TFRecordWriter(output)

    # if the labels must be included
    if include_labels:
        images, labels = read_info_file(info_file, include_labels=include_labels)

        # number of images in current file
        number_of_images = len(images)

        # for each image
        for i in tqdm(range(number_of_images)):
            image_path = images[i]
            label_path = labels[i]

            with tf.io.gfile.GFile(image_path, 'rb') as f:
                image = f.read()

            with tf.io.gfile.GFile(label_path, 'rb') as f:
                label = f.read()

            serialized_sample = serialize_sample(image, label)
            writer.write(serialized_sample)

    # if the labels must not be included
    else:

        images = read_info_file(info_file, include_labels=include_labels)

        # number of images in current file
        number_of_images = len(images)

        # for each image
        for i in tqdm(range(number_of_images)):
            image_path = images[i]

            with tf.io.gfile.GFile(image_path, 'rb') as f:
                image = f.read()

            serialized_image = serialize_image(image)
            writer.write(serialized_image)


def tfrecord_from_folder(folders, output, include_labels=False):
    """
    Generate and save a tfrecord file starting from the paths of the folders containing images and labels.

    :param folders: a list of 1 or 2 strings: the path to the folder containing images (mandatory) and the path
    to the folder containing the labels (optional).
    :param output: the output name of the tfrecord. It can contain the extension ".tfrecord" but it is not mandatory.
    :param include_labels: if False just the images are added to the tfrecord. If True the both the images and the
    labels are added to the tfrecord.
    """

    # if the extension is not .tfrecord, add the extension.
    filename, file_extension = os.path.splitext(output)
    if file_extension != ".tfrecord":
        output = output + ".tfrecord"

    # open the TFRecord writer
    writer = tf.io.TFRecordWriter(output)

    # if the labels must be included
    if include_labels:
        image_folder = folders[0]
        label_folder = folders[1]

        # get the list of file-paths in the folders
        images = get_all_files(image_folder)
        labels = get_all_files(label_folder)

        # number of images in current file
        number_of_images = len(images)

        # for each image
        for i in tqdm(range(number_of_images)):
            image_path = images[i]
            label_path = labels[i]

            with tf.io.gfile.GFile(image_path, 'rb') as f:
                image = f.read()

            with tf.io.gfile.GFile(label_path, 'rb') as f:
                label = f.read()

            serialized_sample = serialize_sample(image, label)
            writer.write(serialized_sample)

    # if the label must not be included
    else:
        image_folder = folders[0]

        # get the list of file-paths in the folder
        images = get_all_files(image_folder)

        # number of images in current file
        number_of_images = len(images)

        # for each image
        for i in tqdm(range(number_of_images)):
            image_path = images[i]

            with tf.io.gfile.GFile(image_path, 'rb') as f:
                image = f.read()

            serialized_image = serialize_image(image)
            writer.write(serialized_image)


# Function to load tfrecord files.

def load_tfrecord(path, mode, train_image_size=image_size, mean=None):
    """
    Load one or more tfrecord files to build a tf.data.Dataset.

    :param path: a list of tfrecord paths or a single dir path containing the tfrecord files.
    :param mode: a string selecting the loading mode. The available modes are:
    -"training" = the images and labels are loaded and preprocessed for the training.
    -"validation" = the images and labels are loaded and preprocessed for the validation.
    -"validation+" = the images and labels are loaded and preprocessed for the validation. The original rgb image is
    also kept.
    -"images" = just the images are decoded.
    -"images_and_labels" = the images and labels are decoded.
    -"resize_images" = just the images are decoded and resized.
    -"resize_images_and_subtract_mean" = just the images are decoded, resized and the mean is subtracted.
    :param train_image_size: the size of training images. Used to resize images in the modes that implement this
    operation.
    :param mean: the mean of the dataset to subtract. Used in some modes.

    :return: a tf.data.Dataset.
    """

    # update global variables if needed.
    global image_size
    global dataset_mean

    if train_image_size is not None:
        image_size = train_image_size

    if mean is not None:
        dataset_mean = np.array(mean, dtype=np.float32)

    # The path argument is a list of paths or a folder?
    if type(path) == list:
        files = path
    elif os.path.isdir(path):
        files = get_all_files(path)
    else:
        print("Error while loading tfrecord!")
        return None

    # select the processing function based on the loading mode.
    if mode == "training":
        function = _process_training
    elif mode == "validation":
        function = _process_validation
    elif mode == "validation+":
        function = _process_validation_plus
    elif mode == "images":
        function = _process_images
    elif mode == "images_and_labels":
        function = _process_images_and_labels
    elif mode == "resize_images":
        function = _resize_images
    elif mode == "resize_images_and_subtract_mean":
        function = _resize_images_and_subtract_mean

    return tf.data.TFRecordDataset(files).map(function)


# function to interleave two datasets

def interleave_datasets(dataset1, dataset2, r1, r2, batch_size, buffer_size=100, repeat=True, shuffle=True):
    """
    Interleave and merge two tf.data.Dataset.

    :param dataset1: the first dataset.
    :param dataset2: the second dataset.
    :param r1: interleave ratio for dataset1.
    :param r2: interleave ratio for dataset2.
    :param batch_size: the batch size of the output dataset.
    :param buffer_size: the buffer size used for shuffling.
    :param repeat: repeat the output dataset.
    :param shuffle: shuffle the output dataset.

    :return: the interleaved dataset.

    """

    # unbatch the input dataset
    dataset1 = dataset1.apply(tf.data.Dataset.unbatch)
    dataset2 = dataset2.apply(tf.data.Dataset.unbatch)

    # extreme case r1 = 0
    if r1 == 0:
        d = dataset2
        if shuffle:
            d = d.shuffle(buffer_size)
        if repeat:
            d = d.repeat()
        return d.batch(r2)
    # extreme case r2 = 0
    elif r2 == 0:
        d = dataset1
        if shuffle:
            d = d.shuffle(buffer_size)
        if repeat:
            d = d.repeat()
        return d.batch(r1)

    # shuffle the two dataset
    if shuffle:
        dataset1 = dataset1.shuffle(buffer_size)
        dataset2 = dataset2.shuffle(buffer_size)

    # repeat the two dataset
    if repeat:
        dataset1 = dataset1.repeat()
        dataset2 = dataset2.repeat()

    # batch the two dataset using the two ratios
    dataset1 = dataset1.batch(r1)
    dataset2 = dataset2.batch(r2)

    # zip the two batched dataset and return the result
    zipped = tf.data.Dataset.zip((dataset1, dataset2)).map(_concat)

    return zipped.apply(tf.data.Dataset.unbatch).batch(batch_size)

# concat function
def _concat(x, y):

    input = tf.concat((x["input"], y["input"]), axis=0)
    label = tf.concat((x["label"], y["label"]), axis=0)

    return {'input': input, 'label': label}


# Serialization functions

def serialize_sample(image, label):
    """
    Creates a tf.Example message ready to be written to a file from an image and its label.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'input': _bytes_feature(image),
        'label': _bytes_feature(label)
    }

    # Create a Features message using tf.train.Example.
    example_msg = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_msg.SerializeToString()


def serialize_image(image):
    """
    Creates a tf.Example message ready to be written to a file from an image.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'input': _bytes_feature(image)
    }

    # Create a Features message using tf.train.Example.
    example_msg = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_msg.SerializeToString()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, str):
        value = value.encode()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Decode functions

def decode_jpeg(bytes, channels=3):
    return tf.image.decode_jpeg(bytes, channels=channels, fancy_upscaling=False)


def decode_png(bytes, channels=1):
    return tf.image.decode_png(bytes, channels=channels)


# Description dictionaries and process functions

description_input = {
    'input': tf.io.FixedLenFeature([], tf.string, default_value='')
}

description_input_label = {
    'input': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label': tf.io.FixedLenFeature([], tf.string, default_value='')
}


# Decode and preprocess images and labels for training
def _process_training(example_proto):
    example = tf.io.parse_single_example(example_proto, description_input_label)

    # decode the image and the label
    input = decode_jpeg(example['input'])
    label = decode_png(example['label'])

    # preprocess the input and the label for the training
    input, label = pp.preprocess_training_op(input, label, image_size, subtract_mean=True, mean=dataset_mean)

    return {'input': input, 'label': label}


# Decode and preprocess images and labels for validation
def _process_validation(example_proto):
    example = tf.io.parse_single_example(example_proto, description_input_label)

    # decode the image and the label
    input = decode_jpeg(example['input'])
    label = decode_png(example['label'])

    input, label = pp.preprocess_validation_op(input, label, subtract_mean=True, mean=dataset_mean)

    return {'input': input, 'label': label}


# Decode and preprocess images and labels for validation. RGB original images are also kept.
def _process_validation_plus(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, description_input_label)

    # decode the image and the label
    input = decode_jpeg(example['input'])
    label = decode_png(example['label'])

    input_pp, label_pp = pp.preprocess_validation_op(input, label, subtract_mean=True, mean=dataset_mean)

    return {'input': input_pp, 'rgb_input': input, 'label': label_pp}


# Decode images
def _process_images(example_proto):
    example = tf.io.parse_single_example(example_proto, description_input)

    input = decode_jpeg(example['input'], channels=3)

    return {'input': input}


# Decode images and labels
def _process_images_and_labels(example_proto):
    example = tf.io.parse_single_example(example_proto, description_input_label)

    # decode the image and the label
    input = decode_jpeg(example['input'])
    label = decode_png(example['label'])

    return {'input': input, 'label': label}


# Decode images and resize them
def _resize_images(example_proto):
    example = tf.io.parse_single_example(example_proto, description_input)

    # decode the image
    input = decode_jpeg(example['input'])

    input = pp.resize_image_op(input, image_size, subtract_mean=False)

    return {'input': input}


# Decode images, resize them and subtract the dataset mean
def _resize_images_and_subtract_mean(example_proto):
    example = tf.io.parse_single_example(example_proto, description_input)

    # decode the image
    input = decode_jpeg(example['input'])

    # resize the image and subtract the dataset mean
    input = pp.resize_image_op(input, image_size, subtract_mean=True, mean=dataset_mean)

    return {'input': input}


# Utility functions

def get_all_files(dir_path):
    """
    Get all file paths in the selected folder.
    :param dir_path: the folder containing some files
    :return: a list of strings. Every string is the path to a file in dir_path folder.
    """
    files = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        files.extend(filenames)
    return [dir_path + f for f in files]


def read_info_file(info_file, include_labels=False):
    """
    It reads an info txt file and returns the list of image paths and, eventually, the list of label paths.
    The info file has an entry in each line in the form: IMAGE_PATH LABEL_PATH.
    :param info_file: the path to the info txt file .
    :param include_labels: if True the list of labels is returned together with the list of images. If False just the
    list of images is returned.
    :return: the list of image paths and the list of label paths (if include_labels is True).
    """

    # open the info file and convert to a list of lines.
    with open(info_file) as f:
        lines = [line.rstrip('\n') for line in f]

    images = []
    labels = []

    # Add each file path to the respective list
    for l in lines:
        images.append(l.split(" ")[0])
        labels.append(l.split(" ")[1])

    if include_labels:
        return images, labels
    else:
        return images