import numpy as np
import cv2

def _gabor_kernels(ksize=21, sigmas=(3, 5), lambdas=(6, 10), thetas=8, gamma=0.5, psi=0):
    kernels = []
    for sigma in sigmas:
        for lam in lambdas:
            for t in range(thetas):
                theta = (np.pi * t) / thetas
                k = cv2.getGaborKernel(
                    (ksize, ksize),
                    sigma=sigma,
                    theta=theta,
                    lambd=lam,
                    gamma=gamma,
                    psi=psi,
                    ktype=cv2.CV_32F
                )
                # normalisieren hilft, damit Filter vergleichbar skalieren
                k /= (np.sum(np.abs(k)) + 1e-8)
                kernels.append(k)
    return kernels

# einmal global erstellen (schneller)
_KERNELS = _gabor_kernels()

def extract_gabor_features(img, pool="mean_var"):
    """
    img: grayscale 2D (H,W) oder BGR (H,W,3) -> wird zu gray
    pool:
      - "mean_var": pro Filter (mean, var) => 2 * n_kernels Features (sehr gängig)
      - "mean": nur mean
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    feats = []

    for k in _KERNELS:
        resp = cv2.filter2D(img, cv2.CV_32F, k)
        if pool == "mean_var":
            feats.append(resp.mean())
            feats.append(resp.var())
        else:
            feats.append(resp.mean())

    return np.array(feats, dtype=np.float32)

def extract_gabor_features_batch(images, pool="mean_var", resize=(64, 64)):
    feats = []
    for img in images:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, resize)
        feats.append(extract_gabor_features(img, pool=pool))

    return np.vstack(feats).astype(np.float32)
