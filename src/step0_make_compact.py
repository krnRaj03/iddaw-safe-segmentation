"""Step 0: build a compact copy of IDD-AW for Google Colab.

Reads the original download (left unchanged) and writes, for every labelled pair:
  - RGB  resized to 1024x768, JPEG quality 95
  - NIR  resized to 1024x768, single-channel (grayscale) JPEG quality 95
  - mask 26-class level-3 label map (official IDD / IDD-AW mapping), drawn from
         the polygon JSON at full resolution, then resized with nearest neighbour
         so no class values are blended. 255 = ignore (unlabeled, out of roi, ...)
It also writes cleaning_log.csv (file-level issues) and pairs.csv (per-pair facts).

Usage: python step0_make_compact.py SRC_DIR DST_DIR [--workers 4]
The script is resumable: pairs already written are skipped.
"""
import argparse, csv, glob, json, os, sys, hashlib
from multiprocessing import Pool
from PIL import Image, ImageDraw

OUT_W, OUT_H = 1024, 768
IGNORE = 255

# Level-3 ids (26 classes) from the official IDD label definition
# (AutoNUE public-code, helpers/anue_labels.py, column level3Id).
LEVEL3 = {
    "road": 0, "parking": 1, "drivable fallback": 1, "sidewalk": 2,
    "rail track": 3, "non-drivable fallback": 3, "person": 4, "animal": 4,
    "rider": 5, "motorcycle": 6, "bicycle": 7, "autorickshaw": 8, "car": 9,
    "truck": 10, "bus": 11, "caravan": 12, "trailer": 12, "train": 12,
    "vehicle fallback": 12, "curb": 13, "wall": 14, "fence": 15,
    "guard rail": 16, "billboard": 17, "traffic sign": 18, "traffic light": 19,
    "pole": 20, "polegroup": 20, "obs-str-bar-fallback": 21, "building": 22,
    "bridge": 23, "tunnel": 23, "vegetation": 24, "sky": 25,
    "fallback background": 25,
    "unlabeled": IGNORE, "ego vehicle": IGNORE, "rectification border": IGNORE,
    "out of roi": IGNORE, "license plate": IGNORE,
}

def norm_label(lab):
    lab = lab.strip()
    if lab.endswith("_new"):
        lab = lab[:-4]
    if lab not in LEVEL3 and lab.endswith("group"):
        lab = lab[:-5]
    return lab

def find_pairs(src):
    pairs, log = [], []
    for rgb in sorted(glob.glob(os.path.join(src, "*", "*", "rgb", "*", "*_rgb.png"))):
        rel = os.path.relpath(rgb, src)
        split, cond, _, seq, fname = rel.split(os.sep)
        frame = fname.replace("_rgb.png", "")
        nir = os.path.join(src, split, cond, "nir", seq, frame + "_nir.png")
        js = os.path.join(src, split, cond, "gtSeg", seq, frame + "_mask.json")
        if not os.path.exists(nir):
            log.append([rel, "missing NIR partner", "excluded"]); continue
        if not os.path.exists(js):
            log.append([rel, "missing label file", "excluded"]); continue
        pairs.append((split, cond, seq, frame, rgb, nir, js))
    # NIR files with no RGB partner
    for nir in sorted(glob.glob(os.path.join(src, "*", "*", "nir", "*", "*_nir.png"))):
        rel = os.path.relpath(nir, src)
        split, cond, _, seq, fname = rel.split(os.sep)
        rgb = os.path.join(src, split, cond, "rgb", seq, fname.replace("_nir.png", "_rgb.png"))
        if not os.path.exists(rgb):
            log.append([rel, "NIR file without RGB partner", "excluded"])
    return pairs, log

def process(args):
    (split, cond, seq, frame, rgb_p, nir_p, js_p), dst = args
    base = os.path.join(dst, split, cond)
    out_rgb = os.path.join(base, "rgb", seq, frame + "_rgb.jpg")
    out_nir = os.path.join(base, "nir", seq, frame + "_nir.jpg")
    out_msk = os.path.join(base, "mask", seq, frame + "_mask.png")
    rec = {"split": split, "condition": cond, "sequence": seq, "frame": frame,
           "status": "ok", "note": ""}
    try:
        with open(rgb_p, "rb") as f:
            rec["rgb_md5"] = hashlib.md5(f.read()).hexdigest()
        rgb = Image.open(rgb_p); rgb.load(); rgb = rgb.convert("RGB")
        nir = Image.open(nir_p); nir.load(); nir = nir.convert("L")
        rec["width"], rec["height"] = rgb.size
        if nir.size != rgb.size:
            rec["note"] += "RGB/NIR size mismatch;"
        with open(js_p) as f:
            ann = json.load(f)
        W, H = int(ann["imgWidth"]), int(ann["imgHeight"])
        if (W, H) != rgb.size:
            rec["note"] += "label size differs from image;"
        mask = Image.new("L", (W, H), IGNORE)
        draw = ImageDraw.Draw(mask)
        n_obj = n_del = n_bad = n_unknown = 0
        for obj in ann.get("objects", []):
            n_obj += 1
            if obj.get("deleted", 0):
                n_del += 1; continue
            poly = obj.get("polygon", [])
            if isinstance(poly, str):
                poly = json.loads(poly)
            if len(poly) < 3:
                n_bad += 1; continue
            lab = norm_label(obj.get("label", ""))
            if lab not in LEVEL3:
                n_unknown += 1; continue
            draw.polygon([(float(x), float(y)) for x, y in poly], fill=LEVEL3[lab])
        rec.update(n_objects=n_obj, n_deleted=n_del, n_bad_polygon=n_bad, n_unknown_label=n_unknown)
        if not os.path.exists(out_msk):
            for p in (out_rgb, out_nir, out_msk):
                os.makedirs(os.path.dirname(p), exist_ok=True)
            rgb.resize((OUT_W, OUT_H), Image.BILINEAR).save(out_rgb, "JPEG", quality=95)
            nir.resize((OUT_W, OUT_H), Image.BILINEAR).save(out_nir, "JPEG", quality=95)
            mask.resize((OUT_W, OUT_H), Image.NEAREST).save(out_msk)  # written last = done marker
        hist = mask.resize((OUT_W, OUT_H), Image.NEAREST).histogram()
        rec["ignore_frac"] = round(hist[IGNORE] / (OUT_W * OUT_H), 5)
        rec["n_classes_present"] = sum(1 for c in range(26) if hist[c] > 0)
    except Exception as e:  # corrupt or unreadable file
        rec["status"] = "error"; rec["note"] += repr(e)
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    pairs, log = find_pairs(a.src)
    print(f"{len(pairs)} complete pairs; {len(log)} file-level issues", flush=True)
    recs = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process, [(p, a.dst) for p in pairs], chunksize=8), 1):
            recs.append(r)
            if i % 250 == 0:
                print(f"{i}/{len(pairs)}", flush=True)
    recs.sort(key=lambda r: (r["split"], r["condition"], r["sequence"], r["frame"]))
    keys = ["split", "condition", "sequence", "frame", "status", "width", "height",
            "n_objects", "n_deleted", "n_bad_polygon", "n_unknown_label",
            "ignore_frac", "n_classes_present", "rgb_md5", "note"]
    with open(os.path.join(a.dst, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(recs)
    with open(os.path.join(a.dst, "cleaning_log.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "issue", "action"]); w.writerows(log)
        for r in recs:
            if r["status"] != "ok" or r.get("note"):
                w.writerow([f'{r["split"]}/{r["condition"]}/{r["sequence"]}/{r["frame"]}', r.get("note", ""), "check"])
    print("done", flush=True)

if __name__ == "__main__":
    main()
