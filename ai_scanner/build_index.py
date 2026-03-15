#!/usr/bin/env python3
"""
VendorBoss AI Scanner — Phase 1: Build Visual Index
====================================================
Scans a folder of reference card images and builds a perceptual hash index.
The index maps card_number -> hash so we can do fast visual matching at scan time.

Usage:
    python3 build_index.py --game fftcg
    python3 build_index.py --game fftcg --images /path/to/images
    python3 build_index.py --game all
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from PIL import Image
    import imagehash
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run: pip3 install -r requirements.txt")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

SCAN_DIRS = {
    "fftcg": [
        "/Users/travisdewitt/Repos/Scans/Scans/eg",
        "/Users/travisdewitt/Repos/Scans/Scans",
    ],
    "pokemon": [
        "/Users/travisdewitt/Repos/Scans/training_data/pokemon_original",
        "/Users/travisdewitt/Repos/Scans/training_data/pokemon",
    ],
    "magic": [
        "/Users/travisdewitt/Repos/Scans/training_data/magic_original",
        "/Users/travisdewitt/Repos/Scans/training_data/magic",
    ],
}

INDEX_DIR = Path(__file__).parent / "indexes"

# ── Card number extraction ────────────────────────────────────────────────────

def extract_card_number_fftcg(filename: str) -> str | None:
    """
    FFTCG filenames are like: 1-001H.jpg, 27-002H_eg.jpg, Re-003H.jpg
    Card number is the stem without _eg or _FL suffixes.
    """
    stem = Path(filename).stem
    stem = re.sub(r'_(eg|FL|eg_FL)$', '', stem, flags=re.IGNORECASE)
    # Validate it looks like a FFTCG card number: {set}-{num}{rarity}
    if re.match(r'^(\d+|[A-Z][a-z]?)-\d{3}[CRHLS]$', stem, re.IGNORECASE):
        return stem.upper()
    return None

def extract_card_number_pokemon(filename: str, folder_path: str) -> str | None:
    """
    Pokemon images are organized in folders by set.
    Try to use folder name + filename as the card number.
    """
    stem = Path(filename).stem
    set_folder = Path(folder_path).name
    return f"{set_folder}/{stem}"

def extract_card_number_magic(filename: str, folder_path: str) -> str | None:
    """
    Magic images from Scryfall are organized by set code folders.
    """
    stem = Path(filename).stem
    set_folder = Path(folder_path).name
    return f"{set_folder}/{stem}"

# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(game: str, image_dirs: list[str] = None) -> dict:
    """
    Walk image directories and compute perceptual hashes.
    Returns dict: { card_number: { "hash": str, "path": str } }
    """
    dirs = image_dirs or SCAN_DIRS.get(game, [])
    index = {}
    total_found = 0
    total_failed = 0

    for scan_dir in dirs:
        if not os.path.exists(scan_dir):
            print(f"  ⚠ Directory not found, skipping: {scan_dir}")
            continue

        # Collect all image files (including subdirectories)
        image_files = []
        for root, _, files in os.walk(scan_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_files.append((root, f))

        if not image_files:
            print(f"  ⚠ No images found in: {scan_dir}")
            continue

        print(f"\n  📂 {scan_dir} — {len(image_files)} images")

        for folder, filename in tqdm(image_files, desc=f"  Hashing", unit="img"):
            filepath = os.path.join(folder, filename)

            # Extract card number based on game
            if game == "fftcg":
                card_number = extract_card_number_fftcg(filename)
            elif game == "pokemon":
                card_number = extract_card_number_pokemon(filename, folder)
            elif game == "magic":
                card_number = extract_card_number_magic(filename, folder)
            else:
                card_number = Path(filename).stem

            if not card_number:
                total_failed += 1
                continue

            try:
                img = Image.open(filepath).convert("RGB")

                # Compute multiple hash types for robustness
                phash = str(imagehash.phash(img, hash_size=16))      # perceptual
                dhash = str(imagehash.dhash(img, hash_size=16))      # difference
                ahash = str(imagehash.average_hash(img, hash_size=16))  # average

                index[card_number] = {
                    "phash": phash,
                    "dhash": dhash,
                    "ahash": ahash,
                    "path": filepath,
                    "filename": filename,
                }
                total_found += 1

            except Exception as e:
                print(f"\n    ✗ Failed {filename}: {e}")
                total_failed += 1

    print(f"\n  ✅ Indexed {total_found} cards ({total_failed} failed)")
    return index


def save_index(game: str, index: dict):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    path = INDEX_DIR / f"{game}_index.json"
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  💾 Saved to {path} ({size_mb:.1f} MB, {len(index)} entries)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build VendorBoss visual card index")
    parser.add_argument("--game", choices=["fftcg", "pokemon", "magic", "all"],
                        default="fftcg", help="Which game to index")
    parser.add_argument("--images", help="Override image directory")
    args = parser.parse_args()

    games = ["fftcg", "pokemon", "magic"] if args.game == "all" else [args.game]

    for game in games:
        print(f"\n🃏 Building index for: {game.upper()}")
        dirs = [args.images] if args.images else None
        index = build_index(game, dirs)
        if index:
            save_index(game, index)
        else:
            print(f"  ⚠ No cards indexed for {game}")

    print("\n✅ Done! Run match_card.py to test matching.")


if __name__ == "__main__":
    main()
