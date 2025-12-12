# %%
import re
import warnings
from pathlib import Path

import pandas as pd


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


# %%
