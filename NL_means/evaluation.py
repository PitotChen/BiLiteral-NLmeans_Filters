from skimage import io
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

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

reference_path_ori = './PolyU-Real-World-Noisy-Images-Dataset-master/OriginalImages/Canon80D_compr_mean.JPG'
evaluated_path_ori = './PolyU-Real-World-Noisy-Images-Dataset-master/OriginalImages/Canon80D_compr_Real.JPG'

reference_path = './PolyU-Real-World-Noisy-Images-Dataset-master/OriginalImages/Canon80D_compr_mean.JPG'
evaluated_path = './eval_pic/7_21_0.2denoised_image.jpg'

reference_path_opencv = './PolyU-Real-World-Noisy-Images-Dataset-master/OriginalImages/Canon80D_compr_mean.JPG'
evaluated_path_opencv = 'image_opencv.jpg'

reference_image_ori = load_image(reference_path_ori)
evaluated_image_ori = load_image(evaluated_path_ori)
reference_image = load_image(reference_path)
evaluated_image = load_image(evaluated_path)
reference_image_opencv = load_image(reference_path_opencv)
evaluated_image_opencv = load_image(evaluated_path_opencv)

reference_image_ori, evaluated_image_ori = crop_to_match(reference_image_ori, evaluated_image_ori)
reference_image, evaluated_image = crop_to_match(reference_image, evaluated_image)
reference_image_opencv, evaluated_image_opencv = crop_to_match(reference_image_opencv, evaluated_image_opencv)

mse_ori, psnr_ori, ssim_ori = calculate_metrics(reference_image_ori, evaluated_image_ori)
mse, psnr, ssim = calculate_metrics(reference_image, evaluated_image)
mse_opencv, psnr_opencv, ssim_opencv = calculate_metrics(reference_image_opencv, evaluated_image_opencv)

mse_diff = mse_ori - mse
psnr_diff = psnr - psnr_ori
ssim_diff = ssim - ssim_ori

mse_diff_to_opencv = mse_opencv - mse
psnr_diff_to_opencv = psnr - psnr_opencv
ssim_diff_to_opencv = ssim - ssim_opencv

print(f"MSE_ori: {mse_ori}, PSNR_ori: {psnr_ori}, SSIM_ori: {ssim_ori}") # noisy - mean
print(f"MSE: {mse}, PSNR: {psnr}, SSIM: {ssim}")# our alg - mean

print(f"MSE_diff:{mse_diff}, PSNR_diff:{psnr_diff}, SSIM_diff:{ssim_diff}") # our alg verses ori_noisy


print(f"MSE_opencv: {mse_opencv}, PSNR_opencv: {psnr_opencv}, SSIM_opencv: {ssim_opencv}")# opencv - mean
print(f"MSE_diff_to_opencv:{mse_diff_to_opencv}, PSNR_diff_to_opencv:{psnr_diff_to_opencv}, SSIM_diff_to_opencv:{ssim_diff_to_opencv}") # our alg verses opencv