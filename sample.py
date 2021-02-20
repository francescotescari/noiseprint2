import matplotlib.pyplot as plt
import numpy as np
import argparse

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('input', metavar='IMAGE_PATH')
parser.add_argument('-q', '--quality', dest='quality', required=False, default=None, metavar='QUALITY', type=int)
parser.add_argument('--show', dest='show', action='store_const', const=True, required=False, default=True)
parser.add_argument('-o', '--save', dest='save', required=False, default=None, metavar='OUTPUT_PATH')


def show_images(images):
    n = len(images)
    f = plt.figure()
    for i in range(n):
        f.add_subplot(1, n, i + 1)
        plt.axis("off")
        plt.imshow(images[i], cmap='gray')
    plt.show(block=True)


if __name__ == '__main__':
    args = parser.parse_args()
    from noiseprint2 import gen_noiseprint, normalize_noiseprint

    path = args.input
    input_image = np.asarray(Image.open(path))
    noiseprint = gen_noiseprint(path, quality=args.quality)
    if args.save is not None:
        np.save(args.save, noiseprint)
    if args.show:
        show_images((input_image, normalize_noiseprint(noiseprint)))
