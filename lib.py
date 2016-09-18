import random

import numpy as np

from PIL import Image


def gray_pixels(img_path, resize=None):
    image = Image.open(img_path).convert('L')
    if resize is not None:
        image = image.resize(resize, Image.LANCZOS)

    pixels = list(image.getdata())

    return pixels


def rgb_pixels(img_path, resize=None):
    image = Image.open(img_path)

    if resize is not None:
        image = image.resize(resize, Image.LANCZOS)

    pixels = list(image.getdata())

    return [np.array(pixel) for pixel in pixels]


def r_pixels(img_path, resize=None):
    image = Image.open(img_path)
    if resize is not None:
        image = image.resize(resize, Image.LANCZOS)

    pixels = list(image.getdata())

    return [int(pixel[0]) for pixel in pixels]


def g_pixels(img_path, resize=None):
    image = Image.open(img_path)
    if resize is not None:
        image = image.resize(resize, Image.LANCZOS)

    pixels = list(image.getdata())

    return [int(pixel[1]) for pixel in pixels]


def b_pixels(img_path, resize=None):
    image = Image.open(img_path)
    if resize is not None:
        image = image.resize(resize)

    pixels = list(image.getdata())

    return [int(pixel[2]) for pixel in pixels]


def bench_k_means(estimator, data):
    estimator.fit(data)
    return estimator


def n_random(n, list_):
    """ Return n randomly selected elements from a list """
    rand_list = []
    l = list(list_)

    while len(rand_list) != n:
        rand_elem = random.choice(l)
        rand_list.append(rand_elem)
        l.pop(l.index(rand_elem))

    return rand_list
