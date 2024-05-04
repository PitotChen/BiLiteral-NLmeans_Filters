from skimage import io
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
import os
import glob

def load_image(image_path):
    return io.imread(image_path)

def crop_to_match(image1, image2):
    if image1.shape[0] > image2.shape[0] or image1.shape[1] > image2.shape[1]:
        return image1[:image2.shape[0], :image2.shape[1]], image2
    elif image2.shape[0] > image1.shape[0] or image2.shape[1] > image1.shape[1]:
        return image1, image2[:image1.shape[0], :image1.shape[1]]
    else:
        return image1, image2

def calculate_metrics(image1, image2):
    mse = mean_squared_error(image1, image2)
    psnr = peak_signal_noise_ratio(image1, image2)
    ssim = structural_similarity(image1, image2, multichannel=True)
    return mse, psnr, ssim

reference_path = 'Canon80D_compr_mean.JPG'
reference_image = load_image(reference_path)
folder_path = './blockout'
evaluated_paths = glob.glob(os.path.join(folder_path, 'Canon80D_*.jpg'))

results_path = './evaluation_results_compr.txt'
with open(results_path, 'w') as file:
    file.write('Image, MSE, PSNR, SSIM\n')
    
    for path in evaluated_paths:
        mse=0
        psnr=0
        ssim_value=0
        evaluated_image = load_image(path)
        ref_cropped, eval_cropped = crop_to_match(reference_image, evaluated_image)
        mse, psnr, ssim_value = calculate_metrics(ref_cropped, eval_cropped)
        
        #file.write(f'{os.path.basename(path)}, {mse:.2f}, {psnr:.2f}, {ssim_value:.4f}\n')
        file_name = os.path.basename(path)
        print(f'Processing {file_name}: MSE={mse:.8f}, PSNR={psnr:.8f}, SSIM={ssim_value:.8f}')
        file.write(f'{file_name}, {mse:.8f}, {psnr:.8f}, {ssim_value:.8f}\n')

print("Metrics calculated and saved to", results_path)