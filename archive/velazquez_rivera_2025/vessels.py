"""Hessian-based vessel detection and mask post-processing.

Polarity warning: `detect_vessels` runs the objectness filter with
`SetBrightObject(False)`, so its output is *dark* where vessels are. That is why
`process_vessels` inverts its threshold. Changing either without the other
silently inverts every mask.
"""

import cv2
import itk
import numpy as np
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects


def detect_vessels(input_image: np.ndarray, min_sigma: float = 1.0, max_sigma: float = 10.0,
                   num_steps: int = 10, alpha=0.5, beta=0.5, gamma=5.0):
    """
    Use the Hessian-based vesselness filter to detect vessels in the image.

    https://examples.itk.org/src/nonunit/review/segmentbloodvesselswithmultiscalehessianbasedmeasure/documentation

    Args:
        input_image: np.ndarray, input image
        min_sigma: float, minimum sigma value
        max_sigma: float, maximum sigma value
        num_steps: int, number of steps
        alpha: float, sensitivity to blob-like structures
        beta: float, sensitivity to plate-like structures
        gamma: float, sensitivity to noise

    Returns:
        segmented_vessels_array: np.ndarray, segmented vessels
    """
    # Run ITK
    input_image = itk.image_from_array(input_image)

    ImageType = type(input_image)
    Dimension = input_image.GetImageDimension()
    HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]

    objectness_filter = itk.HessianToObjectnessMeasureImageFilter[
        HessianImageType, ImageType
    ].New()
    objectness_filter.SetBrightObject(False)  # Set to True if the structures are bright on a dark background
    objectness_filter.SetScaleObjectnessMeasure(False)  # Set to True to scale the objectness measure by the scale
    objectness_filter.SetAlpha(alpha)  # Sensitivity to blob-like structures
                                     # Set/Get Alpha, the weight corresponding to R_A
                                     # (the ratio of the smallest eigenvalue that has to be large to the larger ones).
                                     # Smaller values lead to increased sensitivity to the object dimensionality.
    objectness_filter.SetBeta(beta)   # Sensitivity to plate-like structures - 1.0 default
                                     # Set/Get Beta, the weight corresponding to R_B
                                     # (the ratio of the largest eigenvalue that has to be small to the larger ones).
                                     # Smaller values lead to increased sensitivity to the object dimensionality.
    objectness_filter.SetGamma(gamma)  # Sensitivity to noise - 5.0 default
                                     # Set/Get Gamma, the weight corresponding to S
                                     # (the Frobenius norm of the Hessian matrix, or second-order structureness)

    multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[
        ImageType, HessianImageType, ImageType
    ].New()
    multi_scale_filter.SetInput(input_image)
    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()
    multi_scale_filter.SetSigmaMinimum(min_sigma)
    multi_scale_filter.SetSigmaMaximum(max_sigma)
    multi_scale_filter.SetNumberOfSigmaSteps(num_steps)

    OutputPixelType = itk.UC
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
    rescale_filter.SetInput(multi_scale_filter)
    rescale_filter.Update()

    # Get numpy array
    segmented_vessels = rescale_filter.GetOutput()
    segmented_vessels_array = itk.array_view_from_image(segmented_vessels)
    segmented_vessels_array = np.asarray(segmented_vessels_array, dtype=np.float32)
    return segmented_vessels_array


def process_vessels(vessel_image: np.ndarray, thresh: int, min_size: int = 10,
                    area_threshold: float = 2000, smoothing: int = 3):
    """
    Process the thresholded vessels.

    Args:
        vessel_image: np.ndarray, input image
        thresh: int, threshold value
        min_size: int, minimum size
        area_threshold: float, area threshold
        smoothing: int, smoothing factor

    Returns:
        thresholded_vessels: np.ndarray, thresholded vessels
    """
    # Process the thresholded vessels.
    # Inverted because detect_vessels runs with SetBrightObject(False).
    thresholded_vessels = vessel_image > thresh
    thresholded_vessels = np.invert(thresholded_vessels)

    # Get rid of small objects
    thresholded_vessels = remove_small_objects(thresholded_vessels, min_size=min_size)
    thresholded_vessels = remove_small_holes(thresholded_vessels, area_threshold=area_threshold)

    # Smoothen edges
    thresholded_vessels = binary_closing(thresholded_vessels, footprint=disk(smoothing))

    return thresholded_vessels


def get_brain_mask(brain_image, area_threshold=300000, min_size=10000):
    """
    Get the mask of the brain from the image (run before contrast enhancement).

    Args:
        brain_image: np.ndarray, input image
        area_threshold: int, area threshold
        min_size: int, minimum size of objects to keep

    Returns:
        mask: np.ndarray, mask of the brain
    """
    _, mask = cv2.threshold(brain_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    mask = remove_small_holes(mask.astype(bool), area_threshold=area_threshold)
    mask = remove_small_objects(mask, min_size=min_size)
    return mask
