import cv2
import numpy as np
import math


def custom_gaussian_blur(image, kernel_size=(15, 15), sigma=0):
    # Créer un noyau gaussien
    kernel_x = cv2.getGaussianKernel(kernel_size[0], sigma)
    kernel_y = cv2.getGaussianKernel(kernel_size[1], sigma)

    # Calculer le produit tensoriel des noyaux pour obtenir un noyau 2D
    kernel = np.outer(kernel_x, kernel_y)

    # Appliquer la convolution avec le noyau gaussien
    blurred_image = cv2.filter2D(image, -1, kernel)

    return blurred_image

def custom_blur(image, kernel_size=(5, 5)):
    # Créer un noyau de moyenne
    kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])

    # Appliquer la convolution avec le noyau de moyenne
    blurred_image = cv2.filter2D(image, -1, kernel)

    return blurred_image

def custom_median_blur(image, ksize=5):
    # Copier l'image pour ne pas modifier l'originale
    result_image = np.copy(image)

    # Obtenez les dimensions de l'image
    rows, cols, channels = image.shape

    # Demi-taille du noyau
    ksize_half = ksize // 2

    # Parcourir chaque pixel de l'image
    for i in range(ksize_half, rows - ksize_half):
        for j in range(ksize_half, cols - ksize_half):
            # Récupérer la fenêtre du noyau
            window = image[i - ksize_half:i + ksize_half + 1, j - ksize_half:j + ksize_half + 1]

            # Appliquer le filtre médian sur chaque canal
            for c in range(channels):
                result_image[i, j, c] = np.median(window[:, :, c])

    return result_image

def gaussian(x, sigma):
    return np.exp(-x**2 / (2 * sigma**2))

def color_similarity(window, center, sigma_color):
    return np.exp(-np.sum((window - center) ** 2, axis=2) / (2 * sigma_color ** 2))

def spatial_similarity(i, j, center, d, sigma_space,rows,cols):
    row_indices = np.arange(max(0, i - d), min(rows, i + d + 1))
    col_indices = np.arange(max(0, j - d), min(cols, j + d + 1))

    spatial_kernel = np.exp(-((row_indices - i) ** 2 + (col_indices - j) ** 2) / (2 * sigma_space ** 2))
    return spatial_kernel

def custom_aloys_bilateral(image, d=9, color=20, space=20):
    h, w, _ = image.shape
    out = np.zeros_like(image, dtype=np.uint8)

    for row in range(h):
        for col in range(w):
            norm = 0.0
            filtered = np.zeros(3, dtype=np.float32)

            for dy in range(-d//2, d//2):
                for dx in range(-d//2, d//2):
                    x, y = col + dx, row + dy
                    if x < 0 or x >= w or y < 0 or y >= h:
                        continue

                    px1 = image[row, col]
                    px2 = image[y, x]

                    # Utiliser les canaux individuels pour calculer la différence d'intensité
                    intensityDistance = np.sum(np.abs(px1 - px2))

                    intensityWeight = gaussian(intensityDistance, color)
                    spatialWeight = gaussian(math.sqrt(dy**2 + dx**2), space)

                    weight = spatialWeight * intensityWeight
                    norm += weight
                    filtered += weight * image[y, x]

            out[row, col] = (filtered / norm).astype(np.uint8)

    return out