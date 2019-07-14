import tensorflow as tf


def load(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image)

    width = tf.shape(image)[1] // 2
    frnt_img = image[:, :width, :]
    side_img = image[:, width:, :]

    frnt_img = tf.dtypes.cast(frnt_img, tf.float32)
    side_img = tf.dtypes.cast(side_img, tf.float32)

    return frnt_img, side_img


def resize_img(frnt_img, side_img, h, w):
    frnt_img = tf.image.resize(frnt_img, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    side_img = tf.image.resize(side_img, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return frnt_img, side_img


def random_crop(frnt_img, side_img):
    # stack both image on top to get the same cropped area
    stck_img = tf.stack([frnt_img, side_img], axis=0)
    crop_img = tf.image.random_crop(stck_img, size=[2, 256, 256, 3])

    return crop_img[0], crop_img[1]


def norm_img(frnt_img, side_img):
    frnt_img = tf.image.per_image_standardization(frnt_img)
    side_img = tf.image.per_image_standardization(side_img)

    return frnt_img, side_img


@tf.function()
def random_jitter(frnt_image, side_image):
    # resizing to 286 x 286 x 3
    frnt_image, side_image = resize_img(frnt_image, side_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    frnt_image, side_image = random_crop(frnt_image, side_image)

    # randomly mirroring only front image
    frnt_image = tf.image.random_flip_left_right(frnt_image)

    return frnt_image, side_image


def load_train(filename):
    frnt, side = load(filename)
    frnt, side = random_jitter(frnt, side)
    frnt, side = norm_img(frnt, side)

    return frnt, side


def load_test(filename):
    frnt, side = load(filename)
    frnt, side = resize_img(frnt, side, 256, 256)
    frnt, side = norm_img(frnt, side)

    return frnt, side
