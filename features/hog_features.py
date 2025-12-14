import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(
    images,
    resize=(64, 64),
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
):
    hog_features = []

    for img in images:
        img = cv2.resize(img, resize)

        hog_feat = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            channel_axis=None
        )

        hog_features.append(hog_feat)

    return np.asarray(hog_features, dtype=np.float32)