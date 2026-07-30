"""Reading TIFF slices and unpacking their channels.

`load_channels` / `load_3_channels` assume the acquisition wrote one file per
channel per slice, interleaved in sorted order, so slice `idx` occupies files
`idx * n_channels` through `idx * n_channels + n_channels - 1`. A stray file in
the glob shifts every pairing.
"""

import glob
from pprint import pprint

import SimpleITK as sitk


def read_tif(filepath):
    """
    Read tiff files using SimpleITK

    Args:
        filepath: str, path to tiff file

    Returns:
        image: np.ndarray, tiff image
    """
    image = sitk.ReadImage(filepath)
    image = sitk.GetArrayFromImage(image)
    return image


def load_channel(filepath: str, idx: int, debug=False):
    """
    Load the image channels from the given filepath.

    Args:
        filepath: str, path to the image
        idx: int, index of the image to load

    Returns:
        channels: np.ndarray, image channels
    """
    filepaths = sorted(glob.glob(filepath))


    # Load the image
    curr_img = read_tif(filepaths[idx])

    if debug:
        print(f"Found {int(len(filepaths))} slices")
        pprint(filepaths)
        print("\nImage:", filepaths[idx])

        # Check image stats
        print(f"\nChannel shape: {curr_img.shape}")
        print(f"Channel dtype: {curr_img.dtype}")
        print(f"Channel min: {curr_img.min()}")
        print(f"Channel max: {curr_img.max()}")
        print(f"Channel mean: {curr_img.mean()}")

    return curr_img, filepaths[idx]


def load_channels(filepath: str, idx: int):
    """
    Load the image channels from the given filepath.

    Args:
        filepath: str, path to the image
        idx: int, index of the image to load

    Returns:
        channels: np.ndarray, image channels
    """
    filepaths = sorted(glob.glob(filepath))
    print(f"Found {int(len(filepaths)/2)} slices")
    pprint(filepaths)

    # Load the image
    file_idx = idx * 2  # Multiply by 2 because we have 2 channels and they're stored in pairs
    curr_img = (read_tif(filepaths[file_idx]), read_tif(filepaths[file_idx + 1]))
    print("\nCh1:", filepaths[file_idx])
    print("Ch2:", filepaths[file_idx + 1])
    curr_ch1 = curr_img[0]
    curr_ch2 = curr_img[1]

    # Check image stats
    print(f"\nChannel shape: {curr_ch1.shape}")
    print(f"Channel dtype: {curr_ch1.dtype}")
    print(f"Channel 1 min: {curr_ch1.min()}")
    print(f"Channel 1 max: {curr_ch1.max()}")
    print(f"Channel 1 mean: {curr_ch1.mean()}")

    return curr_ch1, curr_ch2


def load_3_channels(filepath: str, idx: int):
    """
    Load the image channels from the given filepath.

    Args:
        filepath: str, path to the image
        idx: int, index of the image to load

    Returns:
        channels: np.ndarray, image channels
    """
    filepaths = sorted(glob.glob(filepath))
    print(f"Found {int(len(filepaths)/3)} slices")
    pprint(filepaths)

    # Load the image
    file_idx = idx * 3  # Multiply by 3 because we have 3 channels and they're stored in triples
    curr_img = (read_tif(filepaths[file_idx]), read_tif(filepaths[file_idx + 1]), read_tif(filepaths[file_idx + 2]))
    print("\nCh1:", filepaths[file_idx])
    print("Ch2:", filepaths[file_idx + 1])
    print("Ch3:", filepaths[file_idx + 2])
    curr_ch1 = curr_img[0]
    curr_ch2 = curr_img[1]
    curr_ch3 = curr_img[2]

    # Check image stats
    print(f"\nChannel shape: {curr_ch1.shape}")
    print(f"Channel dtype: {curr_ch1.dtype}")
    print(f"Channel 1 min: {curr_ch1.min()}")
    print(f"Channel 1 max: {curr_ch1.max()}")
    print(f"Channel 1 mean: {curr_ch1.mean()}")

    return curr_ch1, curr_ch2, curr_ch3
