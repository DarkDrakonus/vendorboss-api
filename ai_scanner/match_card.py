#!/usr/bin/env python3
"""
VendorBoss AI Scanner — Phase 1: Match a Card Image
====================================================
Takes a card image and finds the best matches in the visual index.
Optionally queries the database to return full card details.

Usage:
    python3 match_card.py my_card.jpg
    python3 match_card.py my_card.jpg --game fftcg
    python3 match_card.py my_card.jpg --top 5
    python3 match_card.py my_card.jpg --db          # also query postgres
    python3 match_card.py /path/to/folder/          # batch test a folder
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from PIL import Image
    import imagehash
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run: pip3 install -r requirements.txt")
    sys.exit(1)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # HEIC support optional — falls back to sips conversion on macOS

INDEX_DIR = Path(__file__).parent / "indexes"

# ── Load index ────────────────────────────────────────────────────────────────

def load_index(game: str) -> dict:
    path = INDEX_DIR / f"{game}_index.json"
    if not path.exists():
        print(f"❌ Index not found for {game}. Run: python3 build_index.py --game {game}")
        sys.exit(1)
    print(f"📂 Loading {game} index...")
    with open(path) as f:
        index = json.load(f)
    print(f"   {len(index)} cards loaded")
    return index


def load_all_indexes() -> dict:
    """Load all available indexes and merge them."""
    combined = {}
    for path in INDEX_DIR.glob("*_index.json"):
        game = path.stem.replace("_index", "")
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            v["game"] = game
            combined[k] = v
        print(f"   Loaded {game}: {len(data)} cards")
    return combined


# ── Hashing ───────────────────────────────────────────────────────────────────

def convert_heic(img_path: str) -> str:
    """
    Convert HEIC to JPEG using macOS sips command.
    Returns path to converted file.
    """
    import subprocess
    import tempfile
    out_path = tempfile.mktemp(suffix='.jpg')
    result = subprocess.run(
        ['sips', '-s', 'format', 'jpeg', img_path, '--out', out_path],
        capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"sips conversion failed: {result.stderr.decode()}")
    return out_path


def hash_image(img_path: str, auto_detect: bool = True) -> dict:
    """Compute all hash types for a query image. Handles HEIC and auto card detection."""
    img = None

    # Step 1: Try automatic card detection (crop card from background)
    if auto_detect:
        try:
            from card_detector import detect_and_crop_card
            detected = detect_and_crop_card(img_path)
            if detected:
                print("  ✅ Card detected and cropped from background")
                img = detected
            else:
                print("  ⚠️  Card detection failed — using full image")
        except ImportError:
            print("  ⚠️  card_detector not available (install opencv-python-headless)")

    # Step 2: Load full image if detection didn't run or failed
    if img is None:
        if img_path.lower().endswith('.heic'):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                print("  Converting HEIC via sips...")
                jpg_path = convert_heic(img_path)
                img = Image.open(jpg_path).convert("RGB")
        else:
            img = Image.open(img_path).convert("RGB")

    return {
        "phash": imagehash.phash(img, hash_size=16),
        "dhash": imagehash.dhash(img, hash_size=16),
        "ahash": imagehash.average_hash(img, hash_size=16),
    }


# ── Matching ──────────────────────────────────────────────────────────────────

def compute_distance(query_hashes: dict, candidate: dict) -> float:
    """
    Weighted combination of hash distances.
    Lower = more similar. 0 = identical.
    phash is weighted most heavily as it's most robust to minor differences.
    """
    phash_dist = query_hashes["phash"] - imagehash.hex_to_hash(candidate["phash"])
    dhash_dist = query_hashes["dhash"] - imagehash.hex_to_hash(candidate["dhash"])
    ahash_dist = query_hashes["ahash"] - imagehash.hex_to_hash(candidate["ahash"])

    # Weighted average (phash gets 50%, others 25% each)
    return (phash_dist * 0.5 + dhash_dist * 0.25 + ahash_dist * 0.25)


HASH_CONFIDENCE_THRESHOLD = 0.70  # Below this, fall back to CLIP


def find_matches(query_path: str, index: dict, top_n: int = 5,
                 use_clip_fallback: bool = True) -> list[dict]:
    """
    Hybrid matching: fast perceptual hash first, CLIP fallback if confidence is low.

    - Scanner/flat images:  hash confidence usually >90% → instant result
    - Phone photos:         hash confidence usually <30% → falls back to CLIP
    """
    print(f"\n🔍 Hashing query image...")
    query_hashes = hash_image(query_path)

    print(f"🔎 Searching {len(index)} cards (perceptual hash)...")
    t0 = time.time()

    results = []
    for card_number, candidate in index.items():
        dist = compute_distance(query_hashes, candidate)
        results.append({
            "card_number": card_number,
            "distance": dist,
            "path": candidate.get("path", ""),
            "game": candidate.get("game", "unknown"),
            "method": "phash",
        })

    results.sort(key=lambda x: x["distance"])
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.2f}s")

    # Check top result confidence
    top_confidence = distance_to_confidence(results[0]["distance"]) if results else 0

    if top_confidence >= HASH_CONFIDENCE_THRESHOLD:
        print(f"   ✅ Hash confidence {top_confidence:.1%} — result accepted")
        return results[:top_n]

    # Low confidence — try CLIP fallback
    if use_clip_fallback:
        print(f"   ⚠️  Hash confidence {top_confidence:.1%} is below {HASH_CONFIDENCE_THRESHOLD:.0%} threshold")
        print(f"   🤖 Falling back to CLIP embeddings...")
        clip_results = _clip_fallback(query_path, top_n)
        if clip_results:
            return clip_results
        print(f"   ⚠️  CLIP unavailable — returning hash results")

    return results[:top_n]


def _clip_fallback(query_path: str, top_n: int) -> list[dict] | None:
    """Attempt CLIP-based matching as fallback."""
    try:
        from clip_engine import ClipEngine
        from card_detector import detect_and_crop_card
        engine = ClipEngine()

        # Load all available CLIP indexes
        n = engine.load_all_indexes()
        if n == 0:
            print(f"   ⚠️  No CLIP indexes found. Build them with:")
            print(f"      python3 clip_engine.py build --game fftcg")
            return None

        # Try card detection first — pass cropped image directly to CLIP
        # This is more reliable than passing the full photo
        detected = detect_and_crop_card(query_path)
        if detected:
            print("   ✅ Using detected card crop for CLIP")
            matches = engine.find_matches_pil(detected, top_n=top_n)
        else:
            matches = engine.find_matches(query_path, top_n=top_n)

        return matches if matches else None

    except ImportError:
        print("   ⚠️  CLIP engine not available (install torch + clip)")
        return None
    except Exception as e:
        print(f"   ⚠️  CLIP failed: {e}")
        return None


def distance_to_confidence(distance: float) -> float:
    """Convert hash distance to a 0-1 confidence score."""
    # Max possible distance for 16-bit hash is 256
    # Distance of 0 = 100% confidence, distance of 256 = 0% confidence
    # We use an exponential decay: conf = e^(-distance/20)
    import math
    return round(math.exp(-distance / 20), 3)


# ── Database lookup ───────────────────────────────────────────────────────────

def lookup_db(card_numbers: list[str]) -> dict:
    """
    Look up card details from Postgres for matched card numbers.
    NOTE: Only Magic (tcg_details) and Sports (card_details) are currently
    in the database. FFTCG cards are NOT in the DB yet — visual matching
    still works, you just won't get a product_id or name back for FFTCG.
    """
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / "vendorboss-api" / ".env")

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return {}

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        results = {}

        # Try FFTCG table first
        placeholders = ",".join(["%s"] * len(card_numbers))
        cur.execute(f"""
            SELECT t.card_number, t.card_name, t.set_id, t.rarity, t.image_url,
                   p.product_id
            FROM tcg_details t
            JOIN products p ON p.product_id = t.product_id
            WHERE t.card_number = ANY(%s)
        """, (card_numbers,))

        for row in cur.fetchall():
            results[row[0]] = {
                "card_number": row[0],
                "card_name":   row[1],
                "set_id":      row[2],
                "rarity":      row[3],
                "image_url":   row[4],
                "product_id":  row[5],
            }

        cur.close()
        conn.close()
        return results

    except Exception as e:
        print(f"  ⚠ DB lookup failed: {e}")
        return {}


# ── Display ───────────────────────────────────────────────────────────────────

def display_results(results: list[dict], db_data: dict = None):
    print(f"\n{'='*60}")
    print(f"  TOP {len(results)} MATCHES")
    print(f"{'='*60}")

    for i, match in enumerate(results, 1):
        # Handle both hash results (distance) and CLIP results (confidence)
        if "confidence" in match:
            confidence = match["confidence"]
        else:
            confidence = distance_to_confidence(match["distance"])
        card_number = match["card_number"]

        # Confidence indicator
        if confidence >= 0.90:
            indicator = "🟢"
        elif confidence >= 0.70:
            indicator = "🟡"
        else:
            indicator = "🔴"

        print(f"\n  #{i} {indicator} {card_number}")
        if 'distance' in match:
            print(f"      Confidence: {confidence:.1%}  (distance: {match['distance']:.1f})")
        else:
            print(f"      Confidence: {confidence:.1%}")
        print(f"      Game:       {match['game'].upper()}")

        method = match.get('method', 'phash')
        method_label = "🔷 CLIP" if method == "clip" else "#️⃣  Hash"
        print(f"      Method:     {method_label}")

        if db_data and card_number in db_data:
            card = db_data[card_number]
            print(f"      Name:       {card['card_name']}")
            print(f"      Set:        {card['set_id']}")
            print(f"      Rarity:     {card['rarity']}")
            if card.get("product_id"):
                print(f"      Product ID: {card['product_id']}")

        print(f"      Ref image:  {Path(match['path']).name}")

    print(f"\n{'='*60}\n")


# ── Batch test ────────────────────────────────────────────────────────────────

def batch_test(folder: str, index: dict, top_n: int = 1) -> dict:
    """
    Test a folder of images. If filenames contain card numbers,
    check if the top match is correct (for accuracy measurement).
    """
    image_files = [f for f in Path(folder).iterdir()
                   if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    correct = 0
    total = 0
    results_log = []

    for img_path in tqdm(image_files, desc="Batch testing"):
        matches = find_matches(str(img_path), index, top_n=top_n)
        if not matches:
            continue

        top_match = matches[0]
        confidence = distance_to_confidence(top_match["distance"])

        # Try to determine ground truth from filename
        ground_truth = None
        stem = img_path.stem.upper()
        if stem in index:
            ground_truth = stem

        is_correct = ground_truth and top_match["card_number"] == ground_truth
        if ground_truth:
            total += 1
            if is_correct:
                correct += 1

        results_log.append({
            "query": img_path.name,
            "top_match": top_match["card_number"],
            "confidence": confidence,
            "ground_truth": ground_truth,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else None
    summary = {
        "total_tested": len(image_files),
        "with_ground_truth": total,
        "correct": correct,
        "accuracy": f"{accuracy:.1%}" if accuracy else "N/A",
        "results": results_log,
    }

    print(f"\n{'='*60}")
    print(f"  BATCH RESULTS")
    print(f"{'='*60}")
    print(f"  Images tested:    {len(image_files)}")
    print(f"  With ground truth: {total}")
    print(f"  Correct:          {correct}")
    print(f"  Accuracy:         {summary['accuracy']}")
    print(f"{'='*60}\n")

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Match a card image against the VendorBoss index")
    parser.add_argument("image", help="Path to image file or folder for batch test")
    parser.add_argument("--game", choices=["fftcg", "pokemon", "magic", "all"],
                        default="all", help="Which index to search (default: all)")
    parser.add_argument("--top", type=int, default=5, help="Number of matches to return")
    parser.add_argument("--db", action="store_true", help="Look up results in Postgres")
    parser.add_argument("--batch", action="store_true", help="Test a folder of images")
    parser.add_argument("--save", help="Save results to JSON file")
    args = parser.parse_args()

    # Load index
    if args.game == "all":
        print("📚 Loading all available indexes...")
        index = load_all_indexes()
    else:
        index = load_index(args.game)

    if not index:
        print("❌ No index data loaded. Run build_index.py first.")
        sys.exit(1)

    # Batch mode
    if args.batch or os.path.isdir(args.image):
        results = batch_test(args.image, index, top_n=args.top)
        if args.save:
            with open(args.save, "w") as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {args.save}")
        return

    # Single image mode
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    matches = find_matches(args.image, index, top_n=args.top)

    # Optional DB lookup
    db_data = {}
    if args.db:
        card_numbers = [m["card_number"] for m in matches]
        print(f"🗄  Looking up {len(card_numbers)} cards in database...")
        db_data = lookup_db(card_numbers)

    display_results(matches, db_data)

    if args.save:
        output = [{"card_number": m["card_number"],
                   "confidence": distance_to_confidence(m["distance"]),
                   "distance": m["distance"],
                   "game": m["game"],
                   "db": db_data.get(m["card_number"])}
                  for m in matches]
        with open(args.save, "w") as f:
            json.dump(output, f, indent=2)
        print(f"💾 Results saved to {args.save}")


if __name__ == "__main__":
    main()
