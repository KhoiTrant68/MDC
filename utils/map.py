import cv2
import numpy as np


# Check if it needs to continue divide
def Division_Judge(img, h0, w0, h, w):
    area = img[h0: h0 + h, w0: w0 + w]
    mean = np.mean(area)
    std = np.std(area, ddof=1)

    total_points = 0
    operated_points = 0

    for row in range(area.shape[0]):
        for col in range(area.shape[1]):
            if (area[row][col] - mean) < 2 * std:
                operated_points += 1
            total_points += 1

    if operated_points / total_points >= 0.95:
        return True
    else:
        return False


# Merging
def Merge(img, h0, w0, h, w):
    area = img[h0:h0 + h, w0:w0 + w]
    mask = (60 < area) & (area < 150)
    img[h0:h0 + h, w0:w0 + w][mask] = 0
    img[h0:h0 + h, w0:w0 + w][~mask] = 255


# Recursion
def Recursion(img, h0, w0, h, w):
    if not Division_Judge(img, h0, w0, h, w) and min(h, w) > 5:
        Recursion(img, h0, w0, int(h / 2), int(w / 2))
        Recursion(img, h0, w0 + int(w / 2), int(h / 2), int(w / 2))
        Recursion(img, h0 + int(h / 2), w0, int(h / 2), int(w / 2))
        Recursion(img, h0 + int(h / 2), w0 + int(w / 2), int(h / 2), int(w / 2))
    else:
        Merge(img, h0, w0, h, w)


# Image segmentation
def Division_Merge_Segmented(img, new_shape):
    # Recursion
    Recursion(img, 0, 0, img.shape[0], img.shape[1])

    # Remove white space
    img = img[1:-1, 1:-1]

    return cv2.resize(img, new_shape)


def laplacian(img, new_shape):
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    laplac = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    laplac = cv2.convertScaleAbs(laplac)
    return cv2.resize(laplac, new_shape)
