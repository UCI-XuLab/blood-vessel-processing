"""Intensity enhancement and bias-field correction applied before segmentation."""

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def auto_contrast(data: np.ndarray, alpha: float = None, beta: float = None) -> np.ndarray:
    """
    Preprocess tiff files to automatically adjust brightness and contrast.
    https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
    """
    if not alpha:
        alpha = np.iinfo(data.dtype).max / (np.max(data) - np.min(data))
    if not beta:
        beta = -np.min(data) * alpha
    img = cv2.convertScaleAbs(data.copy(), alpha=alpha, beta=beta)
    return img


def gamma_correction(image: np.ndarray, gamma: float = 2.0, min_value=None, max_value=None) -> np.ndarray:
    """
    Apply gamma correction to the image.

    Args:
        image: np.ndarray, input image
        gamma: float, gamma value
        min_value: float, intensities below this are clamped to 0 (no clamping if None)
        max_value: float, intensities above this are clamped to it; the normalisation
            and rescaling reference. Defaults to image.max().

    Returns:
        image_enhanced: np.ndarray, gamma corrected image
    """
    if min_value is not None:
        image = image.copy()
        image[image < min_value] = 0
    if max_value is None:
        max_value = image.max()
    else:
        image = image.copy()
        image[image > max_value] = max_value
    # Normalize the image to the range [0, 1]
    image_normalized = image / max_value
    # Apply the exponential transformation
    image_enhanced = np.power(image_normalized, gamma)
    # Rescale the image back to the original intensity range
    image_enhanced = image_enhanced * max_value
    return image_enhanced


def histogram_equalization(image):
    """Equalize an 8-bit image by remapping through its normalised CDF."""
    # Compute the histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Normalize the CDF
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Mask all zero values (if any)
    cdf_m = np.ma.masked_equal(cdf, 0)

    # Normalize the CDF to the range [0, 255]
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    # Fill the masked values with 0
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Map the original intensity levels to the equalized levels
    equalized_image = cdf[image]

    return equalized_image


def compute_average_image(images):
    """
    Compute the average image from a list of images.

    Args:
        images: list, list of images

    Returns:
        average_image: np.ndarray, average image
    """
    average_image = np.mean(images, axis=0)
    return average_image


def n4_bias_correction(img, bg_mask, shrink_factor: float = 15, show: bool = False) -> np.ndarray:
    """
    N4 bias correction for the input image.

    The correction is estimated on a shrunken copy for speed, then the resulting
    log bias field is evaluated at full resolution and divided out.

    Parameters:
    - img: The input image to correct.
    - bg_mask: Brain tissue mask restricting where the bias field is estimated.
    - shrink_factor: The shrink factor for downsampling the image for bias correction.
    - show: Whether to show the intermediate results.

    Returns:
    - corrected_image_full_resolution: The bias corrected image.
    """
    # Create the brain tissue mask
    bg_mask = bg_mask.astype(np.uint8)
    mask_img = sitk.GetImageFromArray(bg_mask)
    mask_img = sitk.LiThreshold(mask_img, 0, 1)

    # Use the raw image and convert it to float32
    raw_img = sitk.GetImageFromArray(img.copy())
    raw_img = sitk.Cast(raw_img, sitk.sitkFloat32)

    # Downsample it for bias correction.
    # Both fall back to full resolution when shrink_factor <= 1; the notebook
    # originals left maskImage unbound on that branch and raised NameError below.
    inputImage = raw_img
    maskImage = mask_img
    if shrink_factor > 1:
        inputImage = sitk.Shrink(raw_img, [shrink_factor] * raw_img.GetDimension())
        maskImage = sitk.Shrink(mask_img, [shrink_factor] * inputImage.GetDimension())

    # Run bias correction
    start_time = time.time()
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias_corrector.Execute(inputImage, maskImage)

    # Apply bias correction to full resolution image
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img)
    corrected_image_full_resolution = raw_img / sitk.Exp(log_bias_field)
    end_time = time.time()
    corrected_image_full_resolution = sitk.GetArrayFromImage(corrected_image_full_resolution)

    # Show the process if True
    if show:
        print(f"Time taken for bias correction: {end_time - start_time:.2f} seconds")

        # Show the brain tissue mask
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(sitk.GetArrayFromImage(mask_img), cmap='gray')
        plt.title(f"Full resolution brain mask")
        plt.subplot(1, 2, 2)
        plt.imshow(sitk.GetArrayFromImage(maskImage), cmap='gray')
        plt.title(f"Downsampled brain mask (shrink factor={shrink_factor})")
        plt.show()

        # Show the log bias field
        plt.figure(figsize=(10, 5))
        plt.imshow(sitk.GetArrayFromImage(log_bias_field))
        plt.colorbar()
        plt.title(f"Log bias field")
        plt.show()

        # Show the corrected bias field image
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Original raw image")
        plt.subplot(1, 2, 2)
        plt.imshow(corrected_image_full_resolution, cmap='gray')
        plt.title(f"Corrected bias raw image")
        plt.show()

        # Increase the contrast of the corrected image and show side-by-side
        preview_alpha = 0.25
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        contrast_comparison = auto_contrast(img, alpha=preview_alpha)
        plt.imshow(contrast_comparison, cmap='gray')
        plt.title(f"Original contrast image (alpha={preview_alpha})")
        plt.subplot(1, 2, 2)
        corrected_bias_contrast = auto_contrast(corrected_image_full_resolution, alpha=preview_alpha)
        plt.imshow(corrected_bias_contrast, cmap='gray')
        plt.title(f"Corrected bias contrast image (alpha={preview_alpha})")
        plt.show()

    return corrected_image_full_resolution
