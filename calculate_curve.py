import os, json, argparse, sys
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# ---- Emotion model (7-class) ----
_EMO_MODEL = "j-hartmann/emotion-english-distilroberta-base"

def _lazy_import_emotion():
    from transformers import pipeline, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_EMO_MODEL, use_fast=True)
    clf = pipeline("text-classification", model=_EMO_MODEL, tokenizer=tok, top_k=None, device_map="auto")
    return clf

# ---- SBERT ----
def _lazy_import_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

# ---- PCA ----
def _lazy_import_pca():
    from sklearn.decomposition import PCA
    return PCA

def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd

def load_story(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_texts_and_ids(data: Dict[str, Any]) -> Tuple[List[str], List[str], str]:
    story_id = str(data.get("story_id") or data.get("id") or "story")
    chs = data.get("chapters") or []
    if not isinstance(chs, list) or len(chs) == 0:
        raise ValueError("JSON has no 'chapters' array.")
    texts, ids = [], []
    for i, ch in enumerate(chs, start=1):
        t = (ch.get("text") or "").strip()
        texts.append(t)
        ids.append(str(ch.get("id") or i))
    return texts, ids, story_id

def emotion_curve(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    clf = _lazy_import_emotion()
    labels = None
    for t in texts:
        if t:
            out0 = clf(t, truncation=True)[0]
            labels = [d["label"] for d in out0]
            break
    if labels is None:
        labels = ["anger","disgust","fear","joy","neutral","sadness","surprise"]

    rows = []
    for t in texts:
        if not t:
            rows.append([0.0]*7); continue
        res = clf(t, truncation=True)[0]
        sc = {d["label"]: float(d["score"]) for d in res}
        rows.append([sc.get(lb, 0.0) for lb in labels])
    E = np.array(rows, dtype=float)
    Ez = zscore(E, axis=0)
    PCA = _lazy_import_pca()
    pc1 = PCA(n_components=1).fit_transform(Ez).ravel()
    pc1z = zscore(pc1, axis=0).ravel()
    return pc1z.astype(float), labels

def sbert_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> np.ndarray:
    ST = _lazy_import_sbert()
    model = ST(model_name, device=device)
    clean = [(t or "").strip() for t in texts]
    embs = model.encode(clean, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embs.astype(np.float32)

def twist_curve(texts: List[str], model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> np.ndarray:
    embs = sbert_embeddings(texts, model_name=model_name, device=device)
    n = embs.shape[0]
    v = np.zeros(n, dtype=np.float32)
    if n >= 2:
        dots = np.sum(embs[1:] * embs[:-1], axis=1)
        v[1:] = 1.0 - np.clip(dots, -1.0, 1.0)
    vz = zscore(v, axis=0).ravel()
    return vz.astype(float)

def fuse_curves(emo_pc1: np.ndarray, twist: np.ndarray, mode: str = "sum", alpha: float = 0.5) -> np.ndarray:
    emo_pc1 = np.asarray(emo_pc1, float).reshape(-1, 1)
    twist = np.asarray(twist, float).reshape(-1, 1)
    if emo_pc1.shape != twist.shape:
        raise ValueError(f"Curve length mismatch: emo={len(emo_pc1)} vs twist={len(twist)}")
    if mode == "sum":
        fused = alpha * emo_pc1 + (1.0 - alpha) * twist
        return zscore(fused, axis=0).ravel()
    elif mode == "pca":
        PCA = _lazy_import_pca()
        X = np.hstack([emo_pc1, twist])
        pc1 = PCA(n_components=1).fit_transform(X).ravel()
        return zscore(pc1, axis=0).ravel()
    else:
        raise ValueError("Unsupported fuse mode. Use 'sum' or 'pca'.")

def process_file(in_path: str,
                 alpha: float = 0.5,
                 fuse: str = "sum",
                 sbert_model: str = "all-MiniLM-L6-v2",
                 sbert_device: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute curves for ONE story and return a record (no file writes).
    """
    data = load_story(in_path)
    texts, ch_ids, story_id = extract_texts_and_ids(data)
    emo_pc1, emo_labels = emotion_curve(texts)
    tw = twist_curve(texts, model_name=sbert_model, device=sbert_device)
    values = fuse_curves(emo_pc1, tw, mode=fuse, alpha=alpha)

    rec = {
        "story_id": story_id,
        "file": os.path.abspath(in_path),
        "chapters": ch_ids,
        "values": [float(x) for x in values],
        "curves": {
            "emo_pc1": [float(x) for x in emo_pc1],
            "twist": [float(x) for x in tw],
        },
        "fuse": {"mode": fuse, "alpha": float(alpha)},
        "meta": {"emotion_labels": emo_labels}
    }
    return rec

def process_folder(folder: str, pattern: str = "*.json", **kwargs) -> Dict[str, Any]:
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")])
    bundle = {"folder": os.path.abspath(folder), "count": 0, "stories": []}
    for fp in files:
        try:
            rec = process_file(fp, **kwargs)
            bundle["stories"].append(rec)
        except Exception as e:
            print(f"[error] {fp}: {e}", file=sys.stderr)
    bundle["count"] = len(bundle["stories"])
    return bundle

def write_bundle(bundle: Dict[str, Any], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(os.path.normpath(bundle.get("folder") or "bundle"))
    out_path = os.path.join(out_dir, f"{base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--folder", type=str)
    g.add_argument("--folders", type=str, nargs="+")
    ap.add_argument("--pattern", type=str, default="*.json")
    ap.add_argument("--out", type=str, default="bundles")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--fuse", type=str, choices=["sum","pca"], default="sum")
    ap.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--sbert_device", type=str, default=None)
    args = ap.parse_args()

    if args.folder:
        b = process_folder(args.folder, pattern=args.pattern,
                           alpha=args.alpha, fuse=args.fuse,
                           sbert_model=args.sbert_model, sbert_device=args.sbert_device)
        path = write_bundle(b, args.out)
        print(f"[bundle] -> {path}")
    else:
        for folder in args.folders:
            b = process_folder(folder, pattern=args.pattern,
                               alpha=args.alpha, fuse=args.fuse,
                               sbert_model=args.sbert_model, sbert_device=args.sbert_device)
            path = write_bundle(b, args.out)
            print(f"[bundle] -> {path}")

if __name__ == "__main__":
    main()
