import os
import time
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

import lib

NORM = None
N_CLUSTERS = 30
RESIZE = (16, 16)

img_name = "obj{number}__{degree}.png"
coil_dir = os.path.join(os.path.dirname(__file__), "data", "coil-100")
training_images = {}
for number in range(1, 101):
    for degree in range(0, 360, 5):
        image_name = img_name.format(number=number, degree=degree)
        image_path = os.path.join(coil_dir, image_name)
        training_images[image_name] = image_path


def rgb_kmeans(pixels):
    return lib.bench_k_means(KMeans(n_clusters=N_CLUSTERS), pixels)


def norm_histogram(image, bins, norm=None):
    r_bins = np.array(sorted([bin[0] for bin in bins]))
    g_bins = np.array(sorted([bin[1] for bin in bins]))
    b_bins = np.array(sorted([bin[2] for bin in bins]))

    r_pixels = np.array(lib.r_pixels(image, resize=RESIZE))
    g_pixels = np.array(lib.g_pixels(image, resize=RESIZE))
    b_pixels = np.array(lib.b_pixels(image, resize=RESIZE))

    r_hist, r_bin_edges = np.histogram(r_pixels, bins=r_bins)
    g_hist, g_bin_edges = np.histogram(g_pixels, bins=g_bins)
    b_hist, b_bin_edges = np.histogram(b_pixels, bins=b_bins)

    if norm is not None:
        r_hist = np.linalg.norm(r_hist, ord=norm)
        g_hist = np.linalg.norm(g_hist, ord=norm)
        b_hist = np.linalg.norm(b_hist, ord=norm)

    return r_hist, g_hist, b_hist


def query_image(img_name, code_words, centroids):
    choices = []

    img_path = os.path.join(coil_dir, img_name)
    r_hist, g_hist, b_hist = norm_histogram(img_path, centroids, norm=NORM)

    for img_name, img_hist in code_words.items():
        r_img_hist, g_img_hist, b_img_hist = img_hist

        r_dist = euclidean(r_hist, r_img_hist)
        g_dist = euclidean(g_hist, g_img_hist)
        b_dist = euclidean(b_hist, b_img_hist)
        dist = r_dist + g_dist + b_dist

        choices.append((img_name, dist))

    choices.sort(key=lambda tup: tup[1])

    return choices[:10]


def sample_pixels(images):
    sampled_pixels = []

    for image in images:
        img_pixels = lib.rgb_pixels(image, resize=RESIZE)
        sampled_pixels += img_pixels

    return np.array(sampled_pixels)


if __name__ == "__main__":
    start_time = time.time()

    print("Sampling pixels from training images...")
    rgb_pixels = sample_pixels(training_images.values())

    print("Running k-means for sampled data...")
    rgb_kmeans = rgb_kmeans(rgb_pixels)

    print("Creating code words dictionary...")
    code_words = {
        image_name: norm_histogram(image_path,
                                   rgb_kmeans.cluster_centers_,
                                   norm=NORM)
        for image_name, image_path in training_images.items()
    }

    images_to_query = ["obj2__60.png",
                       "obj3__60.png",
                       "obj17__200.png",
                       "obj17__355.png",
                       "obj21__170.png",
                       "obj25__170.png",
                       "obj57__205.png"]

    for image in images_to_query:
        print("Querying image {}".format(image))
        best_choices = query_image(image, code_words,
                                   rgb_kmeans.cluster_centers_)

        print("The 10 best choices for image {} are...".format(image))
        for choice in best_choices:
            print("Image {} with euclidean distance {}".format(*choice))

    end_time = time.time()

    print("Code took {} seconds to run".format(end_time - start_time))
