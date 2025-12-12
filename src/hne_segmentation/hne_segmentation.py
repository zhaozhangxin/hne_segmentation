# %%
import re
import warnings
from pathlib import Path
from typing import Generator, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _validate_unique_item(candidates: list, tag: str, raise_error: bool = True):
    """
    Validate that exactly one item exists in the candidates list.

    Parameters
    ----------
    candidates : list
        List of candidate items to validate.
    tag : str
        Description of the item type for error/warning messages.
    raise_error : bool, optional
        If True, raise ValueError on validation failure.
        If False, issue warning and return None. Default is True.

    Returns
    -------
    item or None
        The single item if exactly one candidate is found.
        None if validation fails and raise_error is False.

    Raises
    ------
    ValueError
        If raise_error is True and validation fails (no items or multiple items found).

    """
    if len(candidates) == 0:
        msg = f"No {tag} found."
        if raise_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
            return None

    elif len(candidates) > 1:
        msg = f"Multiple {tag} found: {candidates}"
        if raise_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
            return None

    else:
        return candidates[0]


def parse_atomx_dir(
    atomx_dir,
    flatfiles=None,
    rawfiles=None,
    regex_morphology2d=r"_F(\d+)\.tif",
    regex_segmentation_dir=r"^FOV(\d+)$",
    regex_segmentation_mask=r"CellLabels_F(\d+)\.tif",
):
    atomx_dir = Path(atomx_dir)
    atomx_subdirs = [d for d in atomx_dir.iterdir() if d.is_dir()]

    if flatfiles is None:
        flatfiles = [d.name for d in atomx_subdirs if d.name in ["flatFiles"]]
        flatfiles = _validate_unique_item(flatfiles, "flatFiles directory")

    if rawfiles is None:
        rawfiles = [
            d.name for d in atomx_subdirs if d.name in ["RawFiles", "DecodedFiles"]
        ]
        rawfiles = _validate_unique_item(rawfiles, "rawFiles directory")

    flatfiles_dir = atomx_dir / flatfiles
    rawfiles_dir = atomx_dir / rawfiles

    # Find flat files
    polygons = _validate_unique_item(
        list(flatfiles_dir.glob("./*/*polygons.csv*")), "polygons.csv"
    )
    exprmat_file = _validate_unique_item(
        list(flatfiles_dir.glob("./*/*exprMat_file.csv*")), "exprMat_file.csv"
    )
    fov_positions_file = _validate_unique_item(
        list(flatfiles_dir.glob("./*/*fov_positions_file.csv*")),
        "fov_positions_file.csv",
    )
    metadata_file = _validate_unique_item(
        list(flatfiles_dir.glob("./*/*metadata_file.csv*")), "metadata_file.csv"
    )
    tx_file = _validate_unique_item(
        list(flatfiles_dir.glob("./*/*tx_file.csv*")), "tx_file.csv"
    )

    # Find morphology2d files
    morphology2d_dir = _validate_unique_item(
        list(rawfiles_dir.glob("./*/*/CellStatsDir/Morphology2D")),
        "Morphology2D directory",
    )
    morphology2d_dict = {
        int(re.search(regex_morphology2d, f.name, re.IGNORECASE).group(1)): str(f)
        for f in morphology2d_dir.iterdir()
        if re.search(regex_morphology2d, f.name, re.IGNORECASE)
    }

    # Find segmentation files
    segmentation_dir_dict = {
        int(re.search(regex_segmentation_dir, d.name, re.IGNORECASE).group(1)): d
        for d in rawfiles_dir.glob("./*/*/CellStatsDir/*")
        if d.is_dir() and re.search(regex_segmentation_dir, d.name, re.IGNORECASE)
    }
    segmentation_mask_dict = {
        int(re.search(regex_segmentation_mask, f.name, re.IGNORECASE).group(1)): str(f)
        for d in segmentation_dir_dict.values()
        for f in d.iterdir()
        if re.search(regex_segmentation_mask, f.name, re.IGNORECASE)
    }

    # Final output dictionaries
    flat_files_dict = {
        "polygons": polygons,
        "exprmat_file": exprmat_file,
        "fov_positions_file": fov_positions_file,
        "metadata_file": metadata_file,
        "tx_file": tx_file,
        "morphology2d_dir": morphology2d_dir,
    }

    df_morphology2d = pd.DataFrame(
        list(morphology2d_dict.items()), columns=["fov_id", "morphology2d"]
    )
    df_segmentation_mask = pd.DataFrame(
        list(segmentation_mask_dict.items()),
        columns=["fov_id", "segmentation_mask"],
    )
    df_summary = pd.merge(
        df_morphology2d, df_segmentation_mask, on="fov_id", how="outer"
    ).sort_values("fov_id")
    df_summary["fov_id"] = df_summary["fov_id"].astype(str)

    fov_files_dict = {}
    for _, row in df_summary.iterrows():
        fov_id = row["fov_id"]
        fov_files_dict[fov_id] = {
            "morphology2d": row["morphology2d"],
            "segmentation": row["segmentation_mask"],
        }

    return {"flat_files": flat_files_dict, "fov_files": fov_files_dict}


def crop_regions_from_image(
    image: NDArray,
    segmentation: NDArray,
    background: Union[int, float, list, tuple] = 0,
    channel_axis: int = -1,
) -> Generator[Tuple[int, Union[NDArray, Tuple[NDArray, ...]]], None, None]:
    """
    Crop individual regions from an image based on segmentation mask.

    Parameters
    ----------
    image : NDArray
        Input image, can be 2D (YX) or 3D (YXC or CYX).
    segmentation : NDArray
        Segmentation mask with same spatial dimensions as image.
        Background should be 0, regions labeled with positive integers.
    background : int, float, list, or tuple, optional
        Background value(s) to fill outside the region mask.
        - If scalar: single cropped image with that background value
        - If list/tuple: multiple cropped images, one per background value (each value is a scalar)
        Default is 0.
    channel_axis : int, optional
        Position of channel axis in image array.
        - Use -1 or 2 for YXC format (default)
        - Use 0 for CYX format
        Ignored if image is 2D.

    Yields
    ------
    region_id : int
        ID of the current region from segmentation mask.
    cropped_image : NDArray or tuple of NDArray
        - If background is scalar: single cropped image
        - If background is list/tuple: tuple of cropped images, one per background value

    Raises
    ------
    ValueError
        If image spatial dimensions don't match segmentation shape.
        If channel_axis is not -1, 2, or 0.

    Examples
    --------
    >>> # Single background
    >>> for region_id, img_crop in crop_regions_from_image(hne, seg, background=0):
    ...     print(f"Region {region_id}: {img_crop.shape}")

    >>> # Multiple backgrounds
    >>> for region_id, (img_black, img_white) in crop_regions_from_image(hne, seg, background=[0, 255]):
    ...     print(f"Region {region_id}: black={img_black.shape}, white={img_white.shape}")
    """
    # Validate channel_axis
    if channel_axis not in [-1, 2, 0]:
        raise ValueError(
            f"channel_axis must be -1, 2 (for YXC format) or 0 (for CYX format), got {channel_axis}"
        )

    # Validate spatial dimensions match
    if (
        image.shape[:2] != segmentation.shape  # YX or YXC
        and image.shape[-2:] != segmentation.shape  # CYX
    ):
        raise ValueError(
            f"Image spatial dimensions {image.shape} don't match "
            f"segmentation shape {segmentation.shape}"
        )

    # Determine if image has channels and normalize to YXC format
    is_3d = image.ndim == 3
    restore_channel_axis = False
    if is_3d and channel_axis == 0:
        # Convert CYX to YXC for processing
        image = np.moveaxis(image, 0, -1)
        restore_channel_axis = True

    # Determine if multiple backgrounds requested
    multiple_backgrounds = isinstance(background, (list, tuple))

    # Get unique region IDs (excluding background 0)
    region_ids = np.unique(segmentation)
    region_ids = region_ids[region_ids != 0]

    # Process each region
    for region_id in region_ids:
        # Create binary mask for current region
        region_mask = segmentation == region_id

        # Find bounding box
        ys, xs = np.where(region_mask)
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        # Crop mask to bounding box
        region_mask_crop = region_mask[ymin : ymax + 1, xmin : xmax + 1]

        # Process based on number of backgrounds
        if multiple_backgrounds:
            # Generate multiple crops with different backgrounds
            img_crops = []
            for bg_value in background:
                img_crop = image[ymin : ymax + 1, xmin : xmax + 1].copy()
                img_crop[~region_mask_crop] = bg_value

                if restore_channel_axis:
                    img_crop = np.moveaxis(img_crop, -1, 0)
                img_crops.append(img_crop)

            yield int(region_id), tuple(img_crops)

        else:
            img_crop = image[ymin : ymax + 1, xmin : xmax + 1].copy()
            img_crop[~region_mask_crop] = background

            if restore_channel_axis:
                img_crop = np.moveaxis(img_crop, -1, 0)

            yield int(region_id), img_crop
