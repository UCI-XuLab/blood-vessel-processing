"""Inline preview and figure-export helpers used by the tuning cells."""

import matplotlib.pyplot as plt


def show(image, contour=None,
         image2=None, contour2=None, contour_alpha: float = 0.75,
         title: str = "", title2: str = "",
         xlim=None, ylim=None,
         xlim2=None, ylim2=None,
         axis: bool = True,
         figsize=(10, 10)):
    """
    Display the image.

    Args:
        image: np.ndarray, input image
        title: str, title of the image
    """
    f = plt.figure(figsize=figsize)
    # If there are two images, display them side by side
    if image2 is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        if contour is not None:
            plt.contour(contour, colors='red', linewidths=0.5, alpha=contour_alpha)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.axis(axis)
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap='gray')
        plt.title(title2)
        if contour2 is not None:
            plt.contour(contour2, colors='red', linewidths=0.5, alpha=contour_alpha)
        if xlim2 is not None:
            plt.xlim(xlim2)
        if ylim2 is not None:
            plt.ylim(ylim2)
        plt.axis(axis)
    # If there is only one image, display it
    else:
        plt.imshow(image, cmap='gray')
        plt.title(title)
        if contour is not None:
            plt.contour(contour, colors='red', linewidths=0.5, alpha=contour_alpha)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.axis(axis)
    plt.show()
    f.clear()
    plt.close(f)


def show3(image, contour=None,
          image2=None, contour2=None,
          image3=None, contour3=None,
          contour_alpha: float = 0.75,
          title: str = "", title2: str = "", title3: str = "",
          xlim=None, ylim=None,
          xlim2=None, ylim2=None,
          xlim3=None, ylim3=None,
          axis: bool = True,
          figsize=(20, 10)):
    """
    Display the image.

    Args:
        image: np.ndarray, input image
        title: str, title of the image
    """
    f = plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    if contour is not None:
        plt.contour(contour, colors='red', linewidths=0.5, alpha=contour_alpha)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.axis(axis)

    plt.subplot(1, 3, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(title2)
    if contour2 is not None:
        plt.contour(contour2, colors='red', linewidths=0.5, alpha=contour_alpha)
    if xlim2 is not None:
        plt.xlim(xlim2)
    if ylim2 is not None:
        plt.ylim(ylim2)
    plt.axis(axis)

    plt.subplot(1, 3, 3)
    plt.imshow(image3, cmap='gray')
    plt.title(title3)
    if contour3 is not None:
        plt.contour(contour3, colors='red', linewidths=0.5, alpha=contour_alpha)
    if xlim3 is not None:
        plt.xlim(xlim3)
    if ylim3 is not None:
        plt.ylim(ylim3)
    plt.axis(axis)

    plt.show()
    f.clear()
    plt.close(f)


def show_4(original_image, contrast_enhanced, thresholded, xlim=None, ylim=None):
    """Original / contrast-enhanced / mask / mask-over-contrast, in a 2x2 grid."""
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original zoomed")
    if xlim is not None:
        plt.xlim(xlim)
        plt.ylim(ylim[::-1])

    plt.subplot(2, 2, 2)
    plt.imshow(contrast_enhanced, cmap='gray')
    plt.title("Contrast enhanced")
    if xlim is not None:
        plt.xlim(xlim)
        plt.ylim(ylim[::-1])

    plt.subplot(2, 2, 3)
    plt.imshow(thresholded, cmap='gray')
    plt.title("Thresholded")
    if xlim is not None:
        plt.xlim(xlim)
        plt.ylim(ylim[::-1])

    plt.subplot(2, 2, 4)
    plt.imshow(contrast_enhanced, cmap='gray')
    plt.contour(thresholded, colors='red', linewidths=0.5, alpha=0.35)
    plt.title("Contrast enhanced + contour")
    if xlim is not None:
        plt.xlim(xlim)
        plt.ylim(ylim[::-1])
    plt.show()


def save_figure(image, filename, contours=None):
    """
    Save figure to disk.

    Args:
        image: np.ndarray, input image
        filename: str, path to save the image
        contours: np.ndarray, contours to overlay on the image
    """
    plt.figure(figsize=(20, 20))
    plt.imshow(image, cmap='gray')
    if contours is not None:
        plt.contour(contours, colors='red', linewidths=0.15, alpha=0.35)
    plt.axis('off')
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    print(f"Saved figure to {filename}")
