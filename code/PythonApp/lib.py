import cv2
import numpy as np
import random
import math

def bilateral(image, d, color, space):
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
                    intensityDistance = np.abs(np.sum(px1) - np.sum(px2))

                    intensityWeight = gaussian(intensityDistance, color)
                    spatialWeight = gaussian(math.sqrt(dy**2 + dx**2), space)

                    weight = spatialWeight * intensityWeight
                    norm += weight
                    filtered += weight * image[y, x]

            out[row, col] = (filtered / norm).astype(np.uint8)

    return out

def gaussian(x, sigma):
    return np.exp(-(x ** 2) / (2.0 * sigma ** 2))

def add_poisson_noise(imageRef, intensiteMoyenne):
    image = imageRef.copy()
    generator = np.random.default_rng()
    distributionPoisson = generator.poisson(intensiteMoyenne, size=image.shape)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            echantillon = distributionPoisson[y, x]

            if image.ndim == 2:
                image[y, x] = np.uint8(np.clip(image[y, x] + echantillon, 0, 255))
            elif image.ndim == 3:
                image[y, x] = np.uint8(np.clip(image[y, x] + echantillon, 0, 255))

    return image

def add_gaussian_noise(imageRef, intensiteBruit):
    image = imageRef.copy()
    generator = np.random.default_rng()
    distributionGaussienne = generator.normal(0.0, intensiteBruit, size=image.shape)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            echantillon = distributionGaussienne[y, x]

            if image.ndim == 2:
                image[y, x] = np.uint8(np.clip(image[y, x] + echantillon, 0, 255))
            elif image.ndim == 3:
                image[y, x] = np.uint8(np.clip(image[y, x] + echantillon, 0, 255))

    return image

def add_salt_and_pepper_noise(imageRef, pourcentageBruit):
    image = imageRef.copy()
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r = random.randint(0, 100)
            if r < pourcentageBruit / 2:
                if image.ndim == 2:  # Image en niveaux de gris
                    image[y, x] = 0 if random.randint(0, 1) == 0 else 255  # Sel ou poivre
                elif image.ndim == 3:  # Image couleur (R, G, B)
                    image[y, x, :] = np.array([0, 0, 0], dtype=image.dtype) if random.randint(0, 1) == 0 else np.array([255, 255, 255], dtype=image.dtype)  # Sel ou poivre
            elif r < pourcentageBruit:
                if image.ndim == 2:
                    image[y, x] = 255 if random.randint(0, 1) == 0 else 0  # Sel ou poivre
                elif image.ndim == 3:
                    image[y, x, :] = np.array([255, 255, 255], dtype=image.dtype) if random.randint(0, 1) == 0 else np.array([0, 0, 0], dtype=image.dtype)  # Sel ou poivre

    return image



# Partie PSNR

def calculate_psnr(img1, img2):
    if img1.shape != img2.shape or img1.dtype != img2.dtype:
        print("Les dimensions ou les types d'images ne correspondent pas.")
        return -1.0

    diff = cv2.absdiff(img1, img2)
    diff = diff.astype(np.float32)

    diff = diff ** 2
    mse = np.mean(diff)

    if mse > 1e-10:
        psnr = 10.0 * np.log10(255 ** 2 / mse)
    else:
        print("Les images sont identiques, la PSNR est infinie.")
        return float('inf')

    return psnr

# Partie SSIM

def calculate_mean(image, region):
    return np.mean(image[region])

def calculate_variance(image, region, mean):
    squared_diff = (image[region] - mean) ** 2
    variance = np.mean(squared_diff)
    return variance

def calculate_covariance(image1, region1, image2, region2, mean1, mean2):
    product = (image1[region1] - mean1) * (image2[region2] - mean2)
    covariance = np.mean(product)
    return covariance

def calculate_ssim(img1, img2, window_size, C1, C2):
    stride = window_size // 2
    ssim_sum = 0.0
    num_windows = 0

    for y in range(0, img1.shape[0] - window_size + 1, stride):
        for x in range(0, img1.shape[1] - window_size + 1, stride):
            window_rect = (slice(y, y + window_size), slice(x, x + window_size))

            mean1 = calculate_mean(img1, window_rect)
            mean2 = calculate_mean(img2, window_rect)

            variance1 = calculate_variance(img1, window_rect, mean1)
            variance2 = calculate_variance(img2, window_rect, mean2)

            covariance = calculate_covariance(img1, window_rect, img2, window_rect, mean1, mean2)

            l = (2 * mean1 * mean2 + C1) / (mean1 ** 2 + mean2 ** 2 + C1)
            c = (2 * np.sqrt(variance1) * np.sqrt(variance2) + C2) / (variance1 + variance2 + C2)
            s = (covariance + C2 / 2) / (np.sqrt(variance1) * np.sqrt(variance2) + C2 / 2)

            ssim = l * c * s
            ssim_sum += ssim
            num_windows += 1

    ssim = ssim_sum / num_windows
    return ssim
