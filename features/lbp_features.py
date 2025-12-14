import numpy as np
import cv2
from skimage.feature import local_binary_pattern

def extract_lbp_features_blocks(images, grid_x=8, grid_y=8, radius=2, n_points=None):
    if n_points is None:
        n_points = 8 * radius
    n_points = int(n_points)

    n_bins = n_points + 2

    feats = []
    for img in images:
        img = cv2.resize(img, (64, 64))

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        lbp = local_binary_pattern(img, n_points, radius, method="uniform")

        h, w = lbp.shape
        step_x = w // grid_x
        step_y = h // grid_y

        hist_blocks = []
        for gy in range(grid_y):
            for gx in range(grid_x):
                x0 = gx * step_x
                y0 = gy * step_y
                x1 = (gx + 1) * step_x if gx < grid_x - 1 else w
                y1 = (gy + 1) * step_y if gy < grid_y - 1 else h

                block = lbp[y0:y1, x0:x1]

                hist, _ = np.histogram(
                    block.ravel(),
                    bins=np.arange(0, n_bins + 1),
                    range=(0, n_bins)
                )
                hist = hist.astype(np.float32)
                hist /= (hist.sum() + 1e-6)
                hist_blocks.append(hist)

        feats.append(np.concatenate(hist_blocks))

    return np.vstack(feats)
