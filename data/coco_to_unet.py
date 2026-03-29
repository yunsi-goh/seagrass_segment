"""
Convert a COCO segmentation dataset to U-Net training data (images + binary masks).

Usage
-----
  python data/coco_to_unet.py --input data/CESS.coco-segmentation.zip
  python data/coco_to_unet.py --input data/CESS.coco-segmentation.zip --modality rgb_nir

Output
------
  data/<modality>/images/   *.jpg
  data/<modality>/masks/    *.png  (0=background, 255=seagrass)

Filenames are simplified from 'prefix_jpg.rf.ABC123.jpg' to 'ABC123.jpg'.
The extracted folder is deleted after processing.
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ANNOTATION_FILE = "_annotations.coco.json"


def stage_images_dir(modality: str = "rgb") -> Path:
    return Path("data") / modality / "images"


def stage_masks_dir(modality: str = "rgb") -> Path:
    return Path("data") / modality / "masks"


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess source imagery for the seagrass U-Net pipeline."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/CESS.coco-segmentation.zip"),
        help=(
            "Input COCO-segmentation zip file or extracted folder. "
            "Default: data/CESS.coco-segmentation.zip"
        ),
    )
    parser.add_argument(
        "--modality", "-m",
        type=str,
        default="rgb",
        help="Output subdirectory name under data/. Default: rgb",
    )
    return parser.parse_args()


# ── COCO flow helpers ──────────────────────────────────────────────────────────

def resolve_extract_dir(zip_path: Path) -> Path:
    """Derive a sibling extraction directory by stripping all suffixes from the zip name."""
    stem = zip_path.name
    for suffix in zip_path.suffixes:
        stem = stem.removesuffix(suffix)
    return zip_path.parent / stem


def _extract_zip_unicode(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract zip, re-encoding CP437 filenames to UTF-8 where needed."""
    for info in zf.infolist():
        if not (info.flag_bits & 0x800):   # UTF-8 flag not set
            try:
                info.filename = info.filename.encode("cp437").decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
        zf.extract(info, dest)


def extract_coco_zip(zip_path: Path, extract_dir: Path) -> None:
    if any(extract_dir.rglob(ANNOTATION_FILE)):
        print(f"Already extracted: {extract_dir}")
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Input zip not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} → {extract_dir} …")
    if sys.version_info >= (3, 11):
        with zipfile.ZipFile(zip_path, metadata_encoding="utf-8") as zf:
            zf.extractall(extract_dir)
    else:
        with zipfile.ZipFile(zip_path) as zf:
            _extract_zip_unicode(zf, extract_dir)
    print("Done.")


def find_splits(extract_dir: Path) -> list[Path]:
    """Return all subdirectories that contain an annotation JSON, sorted by name."""
    splits = sorted({p.parent for p in extract_dir.rglob(ANNOTATION_FILE)})
    if not splits:
        raise FileNotFoundError(f"No {ANNOTATION_FILE} found under {extract_dir}")
    print(f"Splits found: {[s.name for s in splits]}")
    return splits


_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def simplify_stem(file_name: str) -> str:
    """'prefix_jpg.rf.ABC123.jpg' → 'ABC123'."""
    stem = Path(file_name).stem
    if ".rf." in stem:
        return stem.rsplit(".rf.", 1)[-1]
    return stem


def find_source_file(split_dir: Path, file_name: str) -> "Path | None":
    """Find image on disk by exact name, or by unique ID after '.rf.' as fallback."""
    exact = split_dir / file_name
    if exact.exists():
        return exact
    stem = Path(file_name).stem
    if ".rf." in stem:
        uid = stem.rsplit(".rf.", 1)[-1]
        for candidate in split_dir.iterdir():
            if candidate.suffix.lower() in _IMG_EXTS:
                cstem = candidate.stem
                if cstem == uid or cstem.endswith(f".rf.{uid}"):
                    return candidate
    return None


def seagrass_category_ids(categories: list[dict]) -> set[int]:
    """Return ids of categories whose name contains 'seagrass' (case-insensitive)."""
    ids     = {c["id"] for c in categories if "seagrass" in c["name"].lower()}
    matched = [c["name"] for c in categories if c["id"] in ids]
    other   = [c["name"] for c in categories if c["id"] not in ids]
    print(f"  Seagrass  ({len(matched)}): {matched}")
    print(f"  Background ({len(other)})")
    return ids


def render_mask(annotations: list[dict], height: int, width: int) -> np.ndarray:
    """Rasterise COCO polygon segments → binary uint8 mask (0 / 255)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        for seg in ann.get("segmentation", []):
            if not isinstance(seg, list):
                # RLE-encoded crowd annotations — skip
                continue
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts.astype(np.int32).reshape(-1, 1, 2)], color=255)
    return mask


def process_coco_split(
    split_dir: Path,
    seen_stems: set[str],
    images_dir: Path,
    masks_dir: Path,
) -> tuple[int, int]:
    """Process one COCO split folder. Returns (n_images, n_with_seagrass)."""
    with open(split_dir / ANNOTATION_FILE) as f:
        data = json.load(f)

    sg_ids = seagrass_category_ids(data["categories"])

    img_meta: dict[int, dict] = {img["id"]: img for img in data["images"]}
    img_anns: dict[int, list] = {img_id: [] for img_id in img_meta}
    for ann in data["annotations"]:
        if ann["category_id"] in sg_ids:
            img_anns[ann["image_id"]].append(ann)

    n_images = n_seagrass = 0

    for img_id, meta in tqdm(img_meta.items(), desc=split_dir.name, unit="img"):
        src_path = find_source_file(split_dir, meta["file_name"])
        if src_path is None:
            print(f"  WARNING: image not found: {meta['file_name']}")
            continue

        stem = simplify_stem(meta["file_name"])
        if stem in seen_stems:
            print(f"  WARNING: duplicate stem '{stem}' — skipped")
            continue
        seen_stems.add(stem)

        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  WARNING: cannot read image: {src_path}")
            continue
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), img)

        mask = render_mask(img_anns[img_id], meta["height"], meta["width"])
        cv2.imwrite(str(masks_dir / f"{stem}.png"), mask)

        n_images += 1
        if mask.any():
            n_seagrass += 1

    return n_images, n_seagrass


def run_coco_flow(input_path: Path, images_dir: Path, masks_dir: Path) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        extract_dir = input_path
        cleanup = False
    else:
        extract_dir = resolve_extract_dir(input_path)
        extract_coco_zip(input_path, extract_dir)
        cleanup = True

    splits = find_splits(extract_dir)
    seen_stems: set[str] = set()
    total_imgs = total_sg = 0

    for split in splits:
        print(f"\n── {split.name} ──")
        n, sg = process_coco_split(split, seen_stems, images_dir, masks_dir)
        total_imgs += n
        total_sg   += sg
        print(f"  {n} images, {sg} with seagrass annotations")

    print(f"\n  Total : {total_imgs} images  ({total_sg} with seagrass)")

    if cleanup:
        print(f"\nCleaning up extracted folder: {extract_dir} …")
        shutil.rmtree(extract_dir)

# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    images_dir = stage_images_dir(args.modality)
    masks_dir  = stage_masks_dir(args.modality)

    print(f"Modality  : {args.modality}")
    print(f"Input     : {args.input}")
    print(f"Images →  : {images_dir}")
    print(f"Masks  →  : {masks_dir}")
    print()

    images_dir.mkdir(parents=True, exist_ok=True)
    run_coco_flow(args.input, images_dir, masks_dir)

    print(f"\nDone  [{args.modality}]")
    print(f"  Images dir : {images_dir}")
    print(f"  Masks dir  : {masks_dir}")


if __name__ == "__main__":
    main()
