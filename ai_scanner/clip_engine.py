#!/usr/bin/env python3
"""
VendorBoss CLIP Embedding Engine — Phase 2
==========================================
Uses OpenAI's CLIP model to generate semantic image embeddings.
Unlike perceptual hashing which compares pixel patterns, CLIP understands
image *content* — it recognizes "black mage hat, glowing eyes, FFTCG frame"
regardless of lighting, angle, or camera differences.

This is the fallback for when perceptual hashing confidence is too low
(phone photos, angled shots, cards in sleeves, etc.)

CLIP runs entirely locally — no API calls, no internet required after setup.

Requirements:
    pip3 install torch torchvision clip-by-openai

Usage:
    from clip_engine import ClipEngine
    engine = ClipEngine()
    engine.build_index("fftcg", image_dir="/path/to/images")
    matches = engine.find_matches("/path/to/photo.jpg", top_n=5)
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

INDEX_DIR = Path(__file__).parent / "indexes"
CLIP_INDEX_DIR = INDEX_DIR / "clip"
CONFIDENCE_THRESHOLD = 0.70  # Below this, perceptual hash falls back to CLIP

# Card image directories (same as build_index.py)
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


# ── CLIP Engine ───────────────────────────────────────────────────────────────

class ClipEngine:
    """
    Manages CLIP embeddings for card matching.
    Lazy-loads the model on first use to avoid slow startup.
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        self.model_name = model_name
        self._model = None
        self._preprocess = None
        self._device = None

    def _load_model(self):
        """Lazy-load CLIP model on first use."""
        if self._model is not None:
            return

        try:
            import torch
            import clip
        except ImportError:
            print("\n❌ CLIP not installed. Run:")
            print("   pip3 install torch torchvision")
            print("   pip3 install git+https://github.com/openai/CLIP.git")
            sys.exit(1)

        import torch
        import clip

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  🤖 Loading CLIP model ({self.model_name}) on {self._device}...")
        t0 = time.time()
        self._model, self._preprocess = clip.load(self.model_name, device=self._device)
        print(f"  ✅ CLIP loaded in {time.time()-t0:.1f}s")

    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Convert a PIL image to a CLIP embedding vector."""
        import torch

        self._load_model()
        img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # normalize

        return embedding.cpu().numpy().flatten().astype(np.float32)

    def embed_image_path(self, img_path: str) -> np.ndarray | None:
        """Load image from path (including HEIC) and embed it."""
        try:
            # Handle HEIC
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass

            img = Image.open(img_path).convert("RGB")
            return self.embed_image(img)
        except Exception as e:
            print(f"  ✗ Failed to embed {img_path}: {e}")
            return None

    # ── Index building ─────────────────────────────────────────────────────────

    def build_index(self, game: str, image_dirs: list[str] = None,
                    batch_size: int = 32) -> int:
        """
        Build CLIP embedding index for all cards in a game.
        Saves embeddings as numpy arrays + metadata JSON.
        Returns number of cards indexed.
        """
        import torch
        self._load_model()

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kwargs: x

        dirs = image_dirs or SCAN_DIRS.get(game, [])
        CLIP_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        # Collect all image files
        image_files = []
        for scan_dir in dirs:
            if not os.path.exists(scan_dir):
                print(f"  ⚠ Skipping missing dir: {scan_dir}")
                continue
            for root, _, files in os.walk(scan_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        image_files.append((root, f))

        if not image_files:
            print(f"  ⚠ No images found for {game}")
            return 0

        print(f"\n  📂 Found {len(image_files)} images for {game}")

        # Extract card numbers (reuse logic from build_index.py)
        import re

        def get_card_number(folder, filename):
            if game == "fftcg":
                stem = Path(filename).stem
                stem = re.sub(r'_(eg|FL|eg_FL)$', '', stem, flags=re.IGNORECASE)
                if re.match(r'^(\d+|[A-Z][a-z]?)-\d{3}[CRHLS]$', stem, re.IGNORECASE):
                    return stem.upper()
                return None
            else:
                return f"{Path(folder).name}/{Path(filename).stem}"

        # Process in batches
        embeddings = {}
        metadata = {}

        batch_imgs = []
        batch_keys = []
        batch_paths = []

        def flush_batch():
            if not batch_imgs:
                return

            import torch
            tensors = torch.stack([self._preprocess(img) for img in batch_imgs])
            tensors = tensors.to(self._device)

            with torch.no_grad():
                embs = self._model.encode_image(tensors)
                embs = embs / embs.norm(dim=-1, keepdim=True)
                embs = embs.cpu().numpy().astype(np.float32)

            for key, path, emb in zip(batch_keys, batch_paths, embs):
                embeddings[key] = emb
                metadata[key] = {"path": path, "filename": Path(path).name, "game": game}

            batch_imgs.clear()
            batch_keys.clear()
            batch_paths.clear()

        failed = 0
        for folder, filename in tqdm(image_files, desc=f"  Embedding {game}", unit="img"):
            card_number = get_card_number(folder, filename)
            if not card_number:
                failed += 1
                continue

            filepath = os.path.join(folder, filename)
            try:
                img = Image.open(filepath).convert("RGB")
                batch_imgs.append(img)
                batch_keys.append(card_number)
                batch_paths.append(filepath)

                if len(batch_imgs) >= batch_size:
                    flush_batch()

            except Exception as e:
                failed += 1

        flush_batch()  # Process remaining

        # Save embeddings
        n = len(embeddings)
        if n == 0:
            print(f"  ⚠ No embeddings generated")
            return 0

        # Save as numpy array + card number lookup
        card_numbers = list(embeddings.keys())
        emb_matrix = np.stack([embeddings[k] for k in card_numbers])

        np.save(CLIP_INDEX_DIR / f"{game}_embeddings.npy", emb_matrix)

        with open(CLIP_INDEX_DIR / f"{game}_metadata.json", "w") as f:
            json.dump({"card_numbers": card_numbers, "metadata": metadata}, f)

        size_mb = (CLIP_INDEX_DIR / f"{game}_embeddings.npy").stat().st_size / 1024 / 1024
        print(f"\n  💾 Saved {n} embeddings ({size_mb:.1f} MB) — {failed} failed")
        return n

    # ── Matching ───────────────────────────────────────────────────────────────

    def load_index(self, game: str) -> bool:
        """Load a pre-built embedding index into memory."""
        emb_path = CLIP_INDEX_DIR / f"{game}_embeddings.npy"
        meta_path = CLIP_INDEX_DIR / f"{game}_metadata.json"

        if not emb_path.exists() or not meta_path.exists():
            return False

        self._embeddings = np.load(emb_path)
        with open(meta_path) as f:
            data = json.load(f)
        self._card_numbers = data["card_numbers"]
        self._metadata = data["metadata"]
        self._loaded_game = game
        return True

    def load_all_indexes(self) -> int:
        """Load all available CLIP indexes."""
        all_embeddings = []
        all_card_numbers = []
        all_metadata = {}

        for path in CLIP_INDEX_DIR.glob("*_embeddings.npy"):
            game = path.stem.replace("_embeddings", "")
            meta_path = CLIP_INDEX_DIR / f"{game}_metadata.json"
            if not meta_path.exists():
                continue

            embs = np.load(path)
            with open(meta_path) as f:
                data = json.load(f)

            all_embeddings.append(embs)
            all_card_numbers.extend(data["card_numbers"])
            all_metadata.update(data["metadata"])
            print(f"   CLIP {game}: {len(data['card_numbers'])} cards")

        if not all_embeddings:
            return 0

        self._embeddings = np.vstack(all_embeddings)
        self._card_numbers = all_card_numbers
        self._metadata = all_metadata
        return len(all_card_numbers)

    def find_matches_pil(self, img: Image.Image, top_n: int = 5) -> list[dict]:
        """Find top N matches for a PIL image (e.g. from card detector)."""
        if not hasattr(self, '_embeddings'):
            return []
        query_emb = self.embed_image(img)
        return self._search(query_emb, top_n)

    def find_matches(self, img_path: str, top_n: int = 5) -> list[dict]:
        """
        Find top N matching cards for a query image using CLIP.
        Returns list of dicts with card_number, confidence, path, game.
        """
        if not hasattr(self, '_embeddings'):
            print("  ⚠ No CLIP index loaded. Call load_index() first.")
            return []

        query_emb = self.embed_image_path(img_path)
        if query_emb is None:
            return []
        return self._search(query_emb, top_n)

    def _search(self, query_emb: np.ndarray, top_n: int) -> list[dict]:
        """Core similarity search against loaded embeddings."""
        # Safety check — catch NaN/zero embeddings
        if not np.isfinite(query_emb).all() or np.allclose(query_emb, 0):
            print("  ⚠ Query embedding is invalid (NaN or zero) — image may be corrupt")
            return []

        t0 = time.time()
        similarities = self._embeddings @ query_emb
        # Replace any NaN/inf with -1
        similarities = np.nan_to_num(similarities, nan=-1.0, posinf=1.0, neginf=-1.0)
        elapsed = time.time() - t0
        print(f"   CLIP search done in {elapsed:.2f}s")

        top_indices = np.argsort(similarities)[::-1][:top_n]

        results = []
        for idx in top_indices:
            card_number = self._card_numbers[idx]
            sim = float(similarities[idx])
            meta = self._metadata.get(card_number, {})
            results.append({
                "card_number": card_number,
                "confidence": round((sim + 1) / 2, 3),  # normalize -1..1 to 0..1
                "similarity": sim,
                "path": meta.get("path", ""),
                "game": meta.get("game", "unknown"),
                "method": "clip",
            })

        return results


# ── Standalone CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLIP card matching engine")
    subparsers = parser.add_subparsers(dest="command")

    # Build index command
    build_parser = subparsers.add_parser("build", help="Build CLIP index")
    build_parser.add_argument("--game", default="fftcg",
                               choices=["fftcg", "pokemon", "magic", "all"])
    build_parser.add_argument("--batch-size", type=int, default=32)

    # Match command
    match_parser = subparsers.add_parser("match", help="Match a card image")
    match_parser.add_argument("image", help="Path to card image")
    match_parser.add_argument("--game", default="all")
    match_parser.add_argument("--top", type=int, default=5)

    args = parser.parse_args()

    engine = ClipEngine()

    if args.command == "build":
        games = ["fftcg", "pokemon", "magic"] if args.game == "all" else [args.game]
        for game in games:
            print(f"\n🃏 Building CLIP index for: {game.upper()}")
            n = engine.build_index(game, batch_size=args.batch_size)
            print(f"  ✅ {n} cards indexed")

    elif args.command == "match":
        if args.game == "all":
            n = engine.load_all_indexes()
        else:
            engine.load_index(args.game)

        print(f"\n🔍 Matching: {args.image}")
        matches = engine.find_matches(args.image, top_n=args.top)

        print(f"\n{'='*60}")
        for i, m in enumerate(matches, 1):
            conf = m['confidence']
            indicator = "🟢" if conf >= 0.90 else "🟡" if conf >= 0.70 else "🔴"
            print(f"  #{i} {indicator} {m['card_number']}")
            print(f"      Confidence: {conf:.1%}  (similarity: {m['similarity']:.4f})")
            print(f"      Game: {m['game'].upper()}")
        print(f"{'='*60}\n")
    else:
        parser.print_help()
