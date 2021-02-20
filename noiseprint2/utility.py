import numpy as np
from PIL import Image

zigzag_index = (
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
)


def convert_dict_qtables(qtables):
    qtables = [qtables[key] for key in range(len(qtables)) if key in qtables]
    for idx, table in enumerate(qtables):
        qtables[idx] = [table[i] for i in zigzag_index]
    return qtables


def jpeg_quality_of_img(image, tnum=0, force_baseline=None):
    assert tnum == 0 or tnum == 1, 'Table number must be 0 or 1'

    if force_baseline is None:
        th_high = 32767
    elif force_baseline == 0:
        th_high = 32767
    else:
        th_high = 255

    h = np.asarray(convert_dict_qtables(image.quantization)[tnum]).reshape((8, 8))

    if tnum == 0:
        # This is table 0 (the luminance table):
        t = np.array(
            [[16, 11, 10, 16, 24, 40, 51, 61],
             [12, 12, 14, 19, 26, 58, 60, 55],
             [14, 13, 16, 24, 40, 57, 69, 56],
             [14, 17, 22, 29, 51, 87, 80, 62],
             [18, 22, 37, 56, 68, 109, 103, 77],
             [24, 35, 55, 64, 81, 104, 113, 92],
             [49, 64, 78, 87, 103, 121, 120, 101],
             [72, 92, 95, 98, 112, 100, 103, 99]])

    elif tnum == 1:
        # This is table 1 (the chrominance table):
        t = np.array(
            [[17, 18, 24, 47, 99, 99, 99, 99],
             [18, 21, 26, 66, 99, 99, 99, 99],
             [24, 26, 56, 99, 99, 99, 99, 99],
             [47, 66, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99]])

    else:
        raise ValueError(tnum, 'Table number must be 0 or 1')

    h_down = np.divide((2 * h - 1), (2 * t))
    h_up = np.divide((2 * h + 1), (2 * t))
    if np.all(h == 1): return 100
    x_down = (h_down[h > 1]).max()
    x_up = (h_up[h < th_high]).min() if (h < th_high).any() else None
    if x_up is None:
        s = 1
    elif x_down > 1 and x_up > 1:
        s = np.ceil(50 / x_up)
    elif x_up < 1:
        s = np.ceil(50 * (2 - x_up))
    else:
        s = 50
    return s


def jpeg_quality_of_file(stream, tnum=0, force_baseline=None):
    return jpeg_quality_of_img(Image.open(stream), tnum=tnum, force_baseline=force_baseline)
