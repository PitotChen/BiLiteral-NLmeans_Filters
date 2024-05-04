import numpy as np
import cv2
from multiprocessing import Pool

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image

def calculate_patch_distance(patch1, patch2, patch_size):
    if patch1.shape != (patch_size, patch_size, 3) or patch2.shape != (patch_size, patch_size, 3):
        return float('inf')
    diff = patch1 - patch2
    distance = np.sum(diff ** 2)
    return distance

def nl_means_denoising_chunk(padded_image, y_start, y_end, x_start, x_end, patch_size, search_window, h):
    pad_width = patch_size // 2
    denoised_chunk = np.zeros((y_end - y_start, x_end - x_start, 3), dtype=np.float32)
    
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            i_padded, j_padded = y + pad_width, x + pad_width
            patch_center = padded_image[i_padded - pad_width:i_padded + pad_width + 1,
                                        j_padded - pad_width:j_padded + pad_width + 1, :]
            weights_sum = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
            normalization = 0.0

            for i in range(-search_window // 2, search_window // 2 + 1):
                for j in range(-search_window // 2, search_window // 2 + 1):
                    window_i, window_j = i_padded + i, j_padded + j
                    if window_i >= pad_width and window_i < padded_image.shape[0] - pad_width and \
                       window_j >= pad_width and window_j < padded_image.shape[1] - pad_width:
                        patch_window = padded_image[window_i - pad_width:window_i + pad_width + 1,
                                                    window_j - pad_width:window_j + pad_width + 1, :]
                        
                        if patch_window.shape == (patch_size, patch_size, 3):
                            distance = calculate_patch_distance(patch_center, patch_window, patch_size)
                            weight = np.exp(-distance / (h ** 2))
                            weights_sum += patch_window * weight
                            normalization += weight

            denoised_pixel = weights_sum / normalization if normalization > 0 else patch_center
            denoised_chunk[y - y_start, x - x_start, :] = denoised_pixel[pad_width, pad_width, :]

    return denoised_chunk


def parallel_nl_means_denoising(image, patch_size, search_window, h, num_processes):
    pad_width = patch_size // 2
    padded_image = np.pad(image, [(pad_width, pad_width), (pad_width, pad_width), (0, 0)], mode='reflect')
    
    num_procs_sqrt = int(np.sqrt(num_processes))
    chunk_size_y = (image.shape[0] + num_procs_sqrt - 1) // num_procs_sqrt
    chunk_size_x = (image.shape[1] + num_procs_sqrt - 1) // num_procs_sqrt

    args = []
    for i in range(num_procs_sqrt):
        for j in range(num_procs_sqrt):
            y_start = i * chunk_size_y
            y_end = min((i + 1) * chunk_size_y, image.shape[0])
            x_start = j * chunk_size_x
            x_end = min((j + 1) * chunk_size_x, image.shape[1])
            args.append((padded_image, y_start, y_end, x_start, x_end, patch_size, search_window, h))

    with Pool(processes=num_processes) as pool:
        denoised_chunks = pool.starmap(nl_means_denoising_chunk, args)

    denoised_image = np.zeros_like(image)
    chunk_idx = 0
    for i in range(num_procs_sqrt):
        for j in range(num_procs_sqrt):
            y_start = i * chunk_size_y
            y_end = min((i + 1) * chunk_size_y, image.shape[0])
            x_start = j * chunk_size_x
            x_end = min((j + 1) * chunk_size_x, image.shape[1])
            denoised_image[y_start:y_end, x_start:x_end, :] = denoised_chunks[chunk_idx]
            chunk_idx += 1

    return denoised_image


# def main():
#     image_path = 'Canon80D_compr_Real.JPG'  
#     image = preprocess_image(image_path)
#     num_processes = 2048    # for 5M pixel image (2976x1680), > 2048 process would not work properly
#     denoised_image = parallel_nl_means_denoising(image, patch_size=5, search_window=15, h=0.2, num_processes=num_processes)

#     output_path = './blockout/Canon80D_2048_5_15_0.2.jpg.jpg'
#     denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)
#     cv2.imwrite(output_path, cv2.cvtColor(denoised_image_uint8, cv2.COLOR_RGB2BGR))
#     print("Image Saved:", output_path)

# if __name__ == '__main__':
#     main()


# def main():
#     # image_path = 'Canon80D_compr_Real.JPG'
#     image_path = 'Canon80D_GO_Real.JPG'  
#     image = preprocess_image(image_path)
#     i = 0
#     for i in range(13): 
#         num_processes = 2**i
#         print(num_processes)
#         denoised_image = parallel_nl_means_denoising(image, patch_size=5, search_window=15, h=0.2, num_processes=num_processes)

#         # Create the output filename including the num_processes
#         output_path = f'./blockout/Canon80D_GO_{num_processes}_5_15_0.2.jpg'
#         denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)
#         cv2.imwrite(output_path, cv2.cvtColor(denoised_image_uint8, cv2.COLOR_RGB2BGR))
#         print(f"Image Saved with num_processes={num_processes}: {output_path}")

# if __name__ == '__main__':
#     main()
import glob
import os

def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
    image_files = glob.glob(os.path.join(input_folder, '*.JPG')) 
    for image_path in image_files:
        image = preprocess_image(image_path)
        num_processes = 32
        denoised_image = parallel_nl_means_denoising(image, patch_size=5, search_window=15, h=0.2, num_processes=num_processes)

        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, base_name.replace('.JPG', 'dn.jpg'))
        denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)
        cv2.imwrite(output_path, cv2.cvtColor(denoised_image_uint8, cv2.COLOR_RGB2BGR))
        print("Image Saved:", output_path)

if __name__ == "__main__":
    input_folder = './image_800/noisy'
    output_folder = './blockout/800'
    main(input_folder, output_folder)