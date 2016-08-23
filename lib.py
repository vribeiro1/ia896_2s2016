from PIL import Image


def gray_pixels(img_path):
    image = Image.open(img_path).convert('L')
    pixels = list(image.getdata())

    return pixels


def r_pixels(img_path):
    image = Image.open(img_path)
    pixels = list(image.getdata())

    return [int(pixel[0]) for pixel in pixels]


def g_pixels(img_path):
    image = Image.open(img_path)
    pixels = list(image.getdata())

    return [int(pixel[1]) for pixel in pixels]


def b_pixels(img_path):
    image = Image.open(img_path)
    pixels = list(image.getdata())

    return [int(pixel[2]) for pixel in pixels]


def bench_k_means(estimator, data):
    estimator.fit(data)
    return estimator

if __name__ == "__main__":
    import os

    coil_dir = os.path.join(os.path.dirname(__file__), "data", "coil-100")
    image_dir = os.path.join(coil_dir, "obj1__0.png")

    print(gray_pixels(image_dir))
    print(r_pixels(image_dir))
    print(g_pixels(image_dir))
    print(b_pixels(image_dir))
