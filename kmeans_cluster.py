import os
import numpy as np
import time

from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

import lib

NORM = None
N_CLUSTERS = 30

img_name = "obj{number}__{degree}.png"
coil_dir = os.path.join(os.path.dirname(__file__), "data", "coil-100")
training_images = {}
for number in range(1, 101):
    for degree in [0, 75, 150, 225, 300]:
        image_name = img_name.format(number=number, degree=degree)
        image_path = os.path.join(coil_dir, image_name)
        training_images[image_name] = image_path


def rgb_kmeans(pixels):
    return lib.bench_k_means(KMeans(n_clusters=N_CLUSTERS), pixels)


def norm_histogram(image, bins, norm=None):
    hist, bin_edges = np.histogram(image, bins=bins)

    if norm is not None:
        hist = np.linalg.norm(hist, ord=norm)

    return hist


def query_image(img_name, code_words, centroids):
    choices = []

    img_path = os.path.join(coil_dir, img_name)
    image = lib.rgb_pixels(img_path, resize=(32, 32))

    hist = norm_histogram(image, centroids, norm=NORM)

    for img_name, img_hist in code_words:
        choices.append((img_name, euclidean(hist, img_hist)))

    choices.sort(key=lambda tup: tup[1])

    return choices[:10]


def sample_pixels(images):
    sampled_pixels = []

    for image in images:
        img_pixels = lib.rgb_pixels(image, resize=(32, 32))
        sampled_pixels += img_pixels

    return sampled_pixels


if __name__ == "__main__":
    start_time = time.time()

    print("Sampling pixels from training images...")
    rgb_pixels = sample_pixels(training_images.values())

    print("Running k-means for sampled data...")
    rgb_kmeans = rgb_kmeans(rgb_pixels)

    print("Creating code words dictionary...")
    code_words = {
        image_name: norm_histogram(lib.rgb_pixels(image_path, resize=(32, 32)),
                                   rgb_kmeans.cluster_centers_,
                                   norm=NORM)
        for image_name, image_path in training_images.items()
    }

    print("Querying 1st image...")
    query_image_1 = "obj2__60.png"
    best_choices_1 = query_image(query_image_1, code_words, rgb_kmeans.cluster_centers_)

    print("The 10 best choices for image {} are...".format(query_image_1))
    for choice in best_choices_1:
        print("Image {} with euclidean distance {}".format(*choice))

    print("Querying 2nd image...")
    query_image_2 = "obj3__60.png"
    best_choices_2 = query_image(query_image_1, code_words, rgb_kmeans.cluster_centers_)

    print("The 10 best choices for image {} are...".format(query_image_2))
    for choice in best_choices_2:
        print("Image {} with euclidean distance {}".format(*choice))

    end_time = time.time()

    print("Code took {} seconds to run".format(end_time - start_time))

