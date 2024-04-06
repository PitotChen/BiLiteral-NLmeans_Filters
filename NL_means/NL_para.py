import numpy as np
import cv2
from multiprocessing import Pool, freeze_support
import sys

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image

def calculate_patch_distance(patch1, patch2, patch_size):
    if patch1.shape != (patch_size, patch_size, 3) or patch2.shape != (patch_size, patch_size, 3):
        return float('inf')  # 如果补丁大小不符合预期，返回一个高的距离值
    diff = patch1 - patch2
    distance = np.sum(diff ** 2)
    return distance

def nl_means_denoising_chunk(args):
    padded_image, y_start, y_end, patch_size, search_window, h, process_id = args
    pad_width = patch_size // 2
    denoised_chunk = np.zeros((y_end - y_start, padded_image.shape[1] - 2 * pad_width, 3))
    processed_pixels = 0
    total_pixels = (padded_image.shape[1] - 2 * pad_width) * (padded_image.shape[1] - 2 * pad_width)
    for y in range(y_start, y_end):
        for x in range(pad_width, padded_image.shape[1] - pad_width):
            i_padded, j_padded = y + pad_width, x
            patch_center = padded_image[i_padded - pad_width:i_padded + pad_width + 1,
                                         j_padded - pad_width:j_padded + pad_width + 1, :]

            if patch_center.shape != (patch_size, patch_size, 3):
                continue  # 如果补丁大小不正确，跳过这个像素

            weights_sum = np.zeros_like(patch_center)
            normalization = 0

            for i in range(-search_window // 2, search_window // 2 + 1):
                for j in range(-search_window // 2, search_window // 2 + 1):
                    window_i, window_j = i_padded + i, j_padded + j
                    if 0 <= window_i < padded_image.shape[0] - patch_size + 1 and \
                       0 <= window_j < padded_image.shape[1] - patch_size + 1:
                        patch_window = padded_image[window_i - pad_width:window_i + pad_width + 1,
                                                    window_j - pad_width:window_j + pad_width + 1, :]
                        if patch_window.shape != (patch_size, patch_size, 3):
                            continue  # 如果窗口大小不正确，跳过这个补丁

                        distance = calculate_patch_distance(patch_center, patch_window, patch_size)
                        weight = np.exp(-distance / (h ** 2))
                        weights_sum += patch_window * weight
                        normalization += weight

            denoised_pixel = weights_sum / normalization if normalization > 0 else patch_center
            denoised_chunk[y - y_start, x - pad_width, :] = denoised_pixel[pad_width, pad_width, :]
            
            processed_pixels += 1
            
            # 每处理100个像素打印一次进度
            sys.stdout.write(f"\rProcess {process_id}: Processed {processed_pixels} of {total_pixels} pixels.")
            sys.stdout.flush()

    return denoised_chunk

def parallel_nl_means_denoising(image, patch_size, search_window, h, num_processes):
    pad_width = patch_size // 2
    padded_image = np.pad(image, [(pad_width, pad_width), (pad_width, pad_width), (0, 0)], mode='reflect')

    chunk_size = image.shape[0] // num_processes
    chunk_ranges = [(i * chunk_size, min((i + 1) * chunk_size, image.shape[0])) for i in range(num_processes)]

    with Pool(processes=num_processes) as pool:
        args = [(padded_image, y_start, y_end, patch_size, search_window, h, i) for i, (y_start, y_end) in enumerate(chunk_ranges)]
        chunks = pool.map(nl_means_denoising_chunk, args)

    denoised_image = np.vstack(chunks)
    return denoised_image

def main():
    image_path = './PolyU-Real-World-Noisy-Images-Dataset-master/CroppedImages/Canon5D2_5_160_6400_bicycle_6_real.JPG'
    image = preprocess_image(image_path)
    '''
    h (denoising strength): 
    This parameter controls the intensity of denoising. If the h value is set too high, the denoising process will over-smooth the image, 
    resulting in a loss of detail. Decreasing the h value can reduce the denoising intensity, thereby retaining more details.

    patch_size: 
    Patches are small image areas compared in the algorithm. If the patch is too large, 
    it may cause the image to be over-smoothed. Typically, small patch sizes preserve image details better, 
    but the denoising effect may be reduced. You can try reducing the patch size, but also pay attention to the denoising effect.

    search_window (search window size): 
    This parameter determines the range in which the algorithm searches for similar patches. 
    Larger search windows may cause the algorithm to smooth over large areas, which may also result in loss of detail. 
    Reducing the search window size can reduce this smoothing effect
    '''
    
    denoised_image = parallel_nl_means_denoising(image, patch_size=7, search_window=7, h=2, num_processes=15)#minimum 772

    # 转换回uint8并保存去噪后的图像
    output_path = 'denoised_image.jpg'
    denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(denoised_image_uint8, cv2.COLOR_RGB2BGR))
    print("Denoised image saved to:", output_path)

if __name__ == '__main__':
    freeze_support()  # For Windows support
    main()
