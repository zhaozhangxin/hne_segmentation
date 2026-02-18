# %%
import json
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

from hne_segmentation import crop_regions_with_context, parse_atomx_dir

TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


cosmx_root = Path("/mnt/nfs/storage/cosmx/cosmx_backup")
hne_root = Path("/mnt/nfs/storage/shuli_temp/20251009_inframe_cosmx_he_rcc")
output_root = Path("/mnt/nfs/storage/zhangxin_temp/251208_zhangxin_hne_segmentation")


# hne_tag to cosmx_tag
tag_dict = {
    "RCC_TMA001_section07_v132": "RCC_TMA001_section07_v132",
    "RCC_TMA002_section07_v132": "RCC_TMA002_section07_v132",
    "RCC_TMA003_section07_v132": "RCC_TMA003_section07_v132",
    "RCC_TMA004_section07_v132": "RCC_TMA004_section07_v132",
    "RCC_TMA541_section07_v132": "RCC_TMA541_section07_2ug_v132",
    "RCC_TMA542_section05_240802_v132": "RCC_TMA542_section05_v132",
    "RCC_TMA542_section05_v132": "RCC_TMA542_section05_v132",
    "RCC_TMA543_section05_240802_v132": "RCC_TMA543_section05_v132",
    "RCC_TMA543_section05_v132": "RCC_TMA543_section05_v132",
    "RCC_TMA544_section05_240802_v132": "RCC_TMA544_section05_v132",
    "RCC_TMA544_section05_v132": "RCC_TMA544_section05_v132",
    "RCC_TMA609_section05_240802_v132": "RCC_TMA609_section05_v132",
    "RCC_TMA609_section05_v132": "RCC_TMA609_section05_v132",
}
n_tags = len(tag_dict)

missing_fovs = {}
for i_tag, (hne_tag, cosmx_tag) in enumerate(tag_dict.items()):
    hne_dir = hne_root / hne_tag / "99_final_output" / "warped_he_non_rigid"
    atomx_dir = cosmx_root / cosmx_tag / "AtoMx"

    file_dict = parse_atomx_dir(atomx_dir)

    fovs_segmentation = list(file_dict["fov_files"].keys())
    fovs_hne = [p.stem for p in hne_dir.glob("*.tiff")]
    fovs = list(set(fovs_segmentation) & set(fovs_hne))
    n_fovs = len(fovs)

    if len(set(fovs_segmentation) - set(fovs)) > 0:
        _fovs = set(fovs_segmentation) - set(fovs)

        print("Warning: FOVs in segmentation not found in H&E directory:")
        print(f"    {_fovs}")

        if hne_tag not in missing_fovs:
            missing_fovs[hne_tag] = {}
        missing_fovs[hne_tag]["in_segmentation_not_in_hne"] = _fovs

    if len(set(fovs_hne) - set(fovs)) > 0:
        _fovs = set(fovs_hne) - set(fovs)
        print("Warning: FOVs in H&E directory not found in segmentation:")
        print(f"    {_fovs}")

        if hne_tag not in missing_fovs:
            missing_fovs[hne_tag] = {}
        missing_fovs[hne_tag]["in_hne_not_in_segmentation"] = _fovs

    # %%
    output_dir = output_root / "cosmx_segmentation" / hne_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # %%
    for i_fov, fov in enumerate(fovs):
        print(
            f"\nProcessing: tag: {hne_tag} ({i_tag + 1}/{n_tags}), fov: {fov} ({i_fov + 1}/{n_fovs})"
        )

        segmentation_f = file_dict["fov_files"][fov]["segmentation"]
        hne_f = hne_dir / f"{fov}.tiff"

        segmentation = tifffile.imread(segmentation_f)
        hne = tifffile.imread(hne_f)

        regions = np.unique(segmentation)
        regions = regions[regions != 0]

        # # back and white background versions
        # iterator = tqdm(
        #     crop_regions_from_image(hne, segmentation, background=[0, 255]),
        #     desc=f"Processing regions in fov_{fov} (masked)",
        #     bar_format=TQDM_FORMAT,
        #     total=len(regions),
        # )

        # for region_id, (hne_black, hne_white) in iterator:
        #     output_f_black = (
        #         output_dir / fov / "01_black_background" / f"{region_id}.tiff"
        #     )
        #     output_f_black.parent.mkdir(parents=True, exist_ok=True)
        #     tifffile.imwrite(output_f_black, hne_black)

        #     output_f_white = (
        #         output_dir / fov / "02_white_background" / f"{region_id}.tiff"
        #     )
        #     output_f_white.parent.mkdir(parents=True, exist_ok=True)
        #     tifffile.imwrite(output_f_white, hne_white)

        # New: 448x448 version with surrounding tissue context
        iterator_context = tqdm(
            crop_regions_with_context(hne, segmentation, crop_size=448, pad_value=255),
            desc=f"Processing regions in fov_{fov} (with context)",
            bar_format=TQDM_FORMAT,
            total=len(regions),
        )

        for region_id, hne_surround in iterator_context:
            output_f_surround = (
                output_dir / fov / "03_surround_context" / f"{region_id}.tiff"
            )
            output_f_surround.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(output_f_surround, hne_surround)

with open(output_root / "missing_fovs.json", "w") as f:
    json.dump(missing_fovs, f, indent=2)