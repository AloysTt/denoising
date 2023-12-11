import cv2
import numpy as np


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
def calculate_ssim(img1, img2, window_size, C1=None, C2=None):
    if C1 is None:
        C1 = (0.01 * 255) ** 2
    if C2 is None:
        C2 = (0.03 * 255) ** 2
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