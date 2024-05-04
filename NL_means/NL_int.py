import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

def compute_integral_image(img):
    # 确保输入图像为int64类型
    img = img.astype(np.int64)
    integral_img = np.cumsum(np.cumsum(img, axis=0), axis=1, dtype=np.int64)
    return integral_img

def squared_difference_integral(integral_img, x1, y1, x2, y2):
    y1 = max(0, min(y1, integral_img.shape[0] - 1))
    x1 = max(0, min(x1, integral_img.shape[1] - 1))
    y2 = max(0, min(y2, integral_img.shape[0] - 1))
    x2 = max(0, min(x2, integral_img.shape[1] - 1))
    A = integral_img[y1-1, x1-1] if x1 > 0 and y1 > 0 else 0
    B = integral_img[y1-1, x2] if y1 > 0 else 0
    C = integral_img[y2, x1-1] if x1 > 0 else 0
    D = integral_img[y2, x2]

    return D - B - C + A



def nl_means_partial(img, integral_img, patch_size, patch_distance, h, y_start, y_end):

    pad_width = patch_distance + patch_size // 2
    padded_img = np.pad(img, pad_width, mode='reflect')
    denoised_img_partial = np.zeros((y_end - y_start, img.shape[1]), dtype=np.int64)

    for y in range(y_start, y_end):
        for x in range(img.shape[1]):
            i, j = y + pad_width, x + pad_width
            weights = 0
            average = 0

            for dy in range(-patch_distance, patch_distance + 1):
                for dx in range(-patch_distance, patch_distance + 1):
                    ii, jj = i + dy, j + dx
                    dist2 = squared_difference_integral(
                        integral_img,
                        max(jj - patch_size // 2, 0),
                        max(ii - patch_size // 2, 0),
                        min(jj + patch_size // 2, padded_img.shape[1] - 1),
                        min(ii + patch_size // 2, padded_img.shape[0] - 1)
                    )
                    weight = np.exp(-max(dist2 - 2 * patch_size**2 * h**2, 0) / (h**2 * patch_size**2))
                    weights += weight
                    average += weight * padded_img[ii, jj]

            denoised_img_partial[y - y_start, x] = average / weights

    return denoised_img_partial

def multi_threaded_nl_means(img, patch_size=5, patch_distance=15, h=0.2, num_threads=16):
    integral_img = compute_integral_image(img)
    img_height = img.shape[0]
    part_height = img_height // num_threads
    futures = []
    denoised_img = np.zeros_like(img)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            y_start = i * part_height
            y_end = (i + 1) * part_height if i != num_threads - 1 else img_height
            futures.append(executor.submit(nl_means_partial, img, integral_img, patch_size, patch_distance, h, y_start, y_end))

    for i, future in enumerate(futures):
        y_start = i * part_height
        denoised_img[y_start:y_start + future.result().shape[0]] = future.result()

    return denoised_img

if __name__ == '__main__':
    img_path = 'Canon80D_compr_Real.JPG'  
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image cannot be loaded. Please check the path.")
        exit()

    img = img.astype(np.int64)  # Use int64 to avoid overflow

    # Apply NL-means denoising
    denoised_img = multi_threaded_nl_means(img)

    # Show the results
    cv2.imshow("Original", img.astype(np.uint8))
    cv2.imshow("Denoised", denoised_img.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
