# VendorBoss AI Scanner

Local card recognition system — runs entirely on your Mac, no API calls needed.

## Setup (one time)

```bash
cd /Users/travisdewitt/Repos/VendorBoss2.0/ai_scanner
pip3 install -r requirements.txt
```

## Step 1 — Build the index

This scans your reference image folders and creates a visual fingerprint for every card.
You only need to do this once (or when you add new card sets).

```bash
# Index FFTCG cards (fastest to start)
python3 build_index.py --game fftcg

# Index all games
python3 build_index.py --game all
```

Index files are saved to `indexes/` folder. FFTCG (~3000 cards) takes about 2-3 minutes.

## Database Coverage

| Game | Visual Matching | DB Lookup (--db flag) |
|------|----------------|----------------------|
| FFTCG | ✅ Works | ❌ Not in DB yet |
| Magic | ✅ Works | ✅ Works |
| Pokemon | ✅ Works | ⚠️ Partial |
| Sports | N/A (no images yet) | ✅ Works |

Visual matching returns card numbers from the image filename regardless of DB status.
The `--db` flag enriches results with card name, set, and product_id — only useful for games in the catalog.

To get FFTCG into the DB, we need to run the FFTCG import script against the catalog data.

## Step 2 — Match a card

```bash
# Match a single image
python3 match_card.py /path/to/your/card_photo.jpg

# Match and also look up names/details from the database
python3 match_card.py /path/to/your/card_photo.jpg --db

# Show top 10 matches instead of 5
python3 match_card.py /path/to/your/card_photo.jpg --top 10

# Search only FFTCG index
python3 match_card.py /path/to/your/card_photo.jpg --game fftcg
```

## Step 3 — Batch accuracy test

Test a whole folder of cards to measure accuracy.
If the filenames match card numbers in the index, it will report % correct.

```bash
# Test all cards in a folder
python3 match_card.py /Users/travisdewitt/Repos/Scans/Scans/eg/ --batch

# Save results to JSON for analysis
python3 match_card.py /Users/travisdewitt/Repos/Scans/Scans/eg/ --batch --save results.json
```

## How it works

**Phase 1 (current):** Perceptual hashing
- Generates 3 types of visual fingerprints per card (pHash, dHash, aHash)
- At match time, computes fingerprints of the query image and finds closest matches
- Very fast (~0.5s for 3000 cards), works well for clean/full card images
- Confidence drops significantly for partial scans or poor lighting

**Phase 2 (planned):** CLIP embeddings
- Uses a neural network to understand card content semantically
- Much better accuracy for partial scans, damaged cards, different angles
- Requires PyTorch (~2GB install)
- Same interface — just swap the hash functions for embed functions

## Understanding the output

```
#1 🟢 1-001H
    Confidence: 97.5%  (distance: 0.5)
    Game:       FFTCG
    Name:       Auron
    Set:        Opus 1
    Product ID: abc-123-...
```

- 🟢 High confidence (>90%) — reliable match
- 🟡 Medium confidence (70-90%) — likely correct, worth verifying
- 🔴 Low confidence (<70%) — partial scan or no good match found

## Adding new card sets

Just add the folder path to `SCAN_DIRS` in `build_index.py` and run `build_index.py` again.

## Files

| File | Purpose |
|------|---------|
| `build_index.py` | Scans image folders, builds visual fingerprint index |
| `match_card.py` | Takes a query image, returns top matches |
| `requirements.txt` | Python dependencies |
| `indexes/` | Generated index files (gitignored) |
