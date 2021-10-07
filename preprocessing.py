import tensorflow as tf


def resize_image_op(image, size, subtract_mean=False, mean=None):
    """
    :param image: image to process. 3D tensor: [image_width, image_height, 3]
    :param size: an int. The size of output image.
    :param subtract_mean: if True the mean is subtracted from the image.
    :param mean: the mean to subtract.

    :return: modified image
    """

    image = tf.cast(image, tf.float32)

    # convert to float
    image = convert2float(image)

    # subtract mean if needed
    if subtract_mean and mean is not None:
        image = image - mean

    # resize the image
    image = tf.image.resize(image, [size, size])

    return image

def preprocess_training_op(image, label, size, subtract_mean=False, mean=None):
    """
    Pre process training data to fit the net requirements

    :param image: image to process. 3D tensor: [image_width, image_height, 3]
    :param label: labels corresponding to image. 3D tensor: [image_width, image_height, 1]
    :param size: an int. The size of output image.
    :param subtract_mean: if True the mean is subtracted from the image.
    :param mean: the mean to subtract.
    :return: modified image and label
    """

    image = tf.cast(image, tf.float32)

    # Convert to float.
    image = convert2float(image)

    # subtract mean if needed
    if subtract_mean and mean is not None:
        image = image - mean

    # Randomly scale the images and labels.
    image, label = random_image_scaling(image, label)

    # Randomly mirror the images and labels.
    image, label = image_mirroring(image, label)

    # Randomly crop image to the right size
    image, label = random_crop_and_pad_image_and_labels(image, label, size, size)

    return image, label


def preprocess_validation_op(image, label, subtract_mean=False, mean=None):
    """
    Pre process validation data to fit the net requirements

    :param image: image to process. 3D tensor: [image_width, image_height, 3]
    :param label: labels corresponding to image. 3D tensor: [image_width, image_height, 1]
    :param subtract_mean: if True the mean is subtracted from the image.
    :param mean: the mean to subtract.
    :return: modified image and label
    """
    image = tf.cast(image, tf.float32)
    image = convert2float(image)

    # subtract mean if needed
    if subtract_mean and mean is not None:
        image = image - mean

    return image, label


def convert2float(image):
    """
    Transform the input image: convert to float, subtract the mean computed on the training data

    :param image: image to convert. 3D tensor: [image_width, image_height, 3]
    :return: modified image
    """

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return image


def random_image_scaling(img, label):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    :param img: image to scale. 3D tensor: [image_width, image_height, 3]
    :param label: label corresponding to image. 3D tensor: [image_width, image_height, 1]
    :return: modified image and label
    """

    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32)

    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))

    new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=1)

    img = tf.image.resize_images(img, new_shape, method=tf.compat.v2.image.ResizeMethod.BILINEAR)

    label = tf.image.resize_images(label, new_shape, method=tf.compat.v2.image.ResizeMethod.NEAREST_NEIGHBOR)

    return img, label


def image_mirroring(img, label):
    """
    Randomly mirror the image

    :param img: image to mirror. 3D tensor: [image_width, image_height, 3]
    :param label: label corresponding to image. 3D tensor: [image_width, image_height, 1]
    :return: modified image and label
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label


def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly scale and crop and pads the input images.

    :param image: training image to crop/ pad. 3D tensor: [image_width, image_height, 3]
    :param label: segmentation mask to crop/ pad. 3D tensor: [image_width, image_height, 1]
    :param crop_h: height of cropped segment
    :param crop_w: width of cropped segment
    :param ignore_label: label to ignore during the training
    :return: modified image and label
    """

    # Pad if needed
    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
                                                tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))
    return img_crop, label_crop