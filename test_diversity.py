import os, sys, glob, json, math, argparse, random, itertools, warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import nltk
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from scipy.spatial.distance import euclidean, cosine
try:
    from fastdtw import fastdtw
    _HAS_FDTW = True
except Exception:
    _HAS_FDTW = False

def _lazy_import_hf():
    from transformers import pipeline, AutoTokenizer
    return pipeline, AutoTokenizer

def _lazy_import_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

# ------------------ IO utils ------------------

def load_texts_from_folder(folder: str, pattern="*.txt") -> List[str]:
    texts = []
    for p in sorted(Path(folder).glob(pattern)):
        try:
            texts.append(Path(p).read_text(encoding="utf-8"))
        except Exception:
            texts.append(Path(p).read_text(errors="ignore"))
    return texts

def _coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None

def _extract_from_chapter(ch: dict) -> Optional[float]:
    for key in ("value", "pc1", "score"):
        v = _coerce_float(ch.get(key))
        if v is not None:
            return v
    txt = (ch.get("text") or "").strip()
    if txt:
        return float(len(txt.split()))  # fallback: word count
    return None

def load_stories_from_jsons(folder: str, pattern="*.json", recursive: bool=False, limit: int|None=None) -> Tuple[List[Dict], List[str]]:
    """Return list of story dicts: {"id", "chapters":[{"id","text"}...], "values":[...], "texts":[...]}"""
    # Support: file path, folder + pattern, optional recursive, optional limit
    folder_path = Path(folder)
    if folder_path.is_file():
        paths = [folder_path]
    else:
        it = folder_path.rglob(pattern) if recursive else folder_path.glob(pattern)
        paths = sorted(it)
    if limit is not None:
        paths = paths[:int(limit)]
    stories, ids = [], []
    for p in paths:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] skip {p}: {e}")
            continue
        story_id = str(data.get("story_id") or p.stem)
        rec = {"id": story_id, "chapters": None, "values": None, "texts": None}

        if "values" in data:
            rec["values"] = [float(v) for v in data["values"]]
            # optional texts if available
            chs = data.get("chapters") or []
            if chs and isinstance(chs, list):
                rec["texts"] = [ (ch.get("text") or "").strip() for ch in chs ]
        elif "chapters" in data and isinstance(data["chapters"], list):
            chs = data["chapters"]
            rec["chapters"] = [{"id": str(ch.get("id") or i+1), "text": (ch.get("text") or "").strip()} for i, ch in enumerate(chs)]
            # build values from chapter fields or fallback word counts
            vals = []
            for ch in chs:
                v = _extract_from_chapter(ch)
                vals.append(float(v if v is not None else 0.0))
            rec["values"] = vals
            rec["texts"] = [c["text"] for c in rec["chapters"]]
        else:
            # not a recognized json; skip
            print(f"[warn] {p} has no 'values' nor 'chapters' — skipping")
            continue

        stories.append(rec)
        ids.append(story_id)
    return stories, ids


def _tokenize_words(text: str) -> List[str]:
    return [t for t in text.split() if t]

def distinct_n_corpus(texts: List[str], n=2) -> float:
    all_ngrams = []
    for t in texts:
        toks = _tokenize_words(t)
        all_ngrams += list(ngrams(toks, n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / max(1, len(all_ngrams))

def self_bleu(texts: List[str]) -> float:
    if len(texts) <= 1:
        return 0.0
    chencherry = SmoothingFunction()
    scores = []
    refs = [ _tokenize_words(t) for t in texts ]
    for i, t in enumerate(texts):
        hyp = _tokenize_words(t)
        other_refs = [r for j, r in enumerate(refs) if j != i]
        if not other_refs:
            continue
        try:
            scores.append(sentence_bleu(other_refs, hyp, smoothing_function=chencherry.method1))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else 0.0

# ------------------ arc features ------------------

def zscore(M: np.ndarray, axis=0) -> np.ndarray:
    mu = M.mean(axis=axis, keepdims=True)
    sd = M.std(axis=axis, keepdims=True)
    sd[sd == 0] = 1.0
    return (M - mu) / sd

def resample_1d(seq: List[float], m: int = 50) -> np.ndarray:
    """Resample a 1D sequence to fixed length m via linear interpolation."""
    if not seq:
        return np.zeros(m, dtype=float)
    x = np.linspace(0, 1, num=len(seq))
    xi = np.linspace(0, 1, num=m)
    return np.interp(xi, x, np.asarray(seq, dtype=float))

def resample_2d(seq2d: List[List[float]], m: int = 50) -> np.ndarray:
    """Resample a list of d-dim vectors (len n) to (m, d)."""
    if not seq2d:
        return np.zeros((m, 1), dtype=float)
    A = np.asarray(seq2d, dtype=float)
    n, d = A.shape
    xs = np.linspace(0, 1, num=n)
    xi = np.linspace(0, 1, num=m)
    out = []
    for j in range(d):
        out.append(np.interp(xi, xs, A[:, j]))
    return np.stack(out, axis=1)  # (m, d)

def dtw_distance(a: List[float], b: List[float]) -> float:
    if not _HAS_FDTW:
        # fallback: euclidean between resampled vectors
        return float(np.linalg.norm(resample_1d(a, 100) - resample_1d(b, 100)))
    dist, _ = fastdtw(a, b, dist=euclidean)
    return float(dist)

# emotion arc via HF pipeline (optional)
def compute_emotion_arc_for_texts(texts: List[str], model_name="j-hartmann/emotion-english-distilroberta-base") -> np.ndarray:
    """
    Return an array of shape (n_chapters, n_emotions). Each row is the probability distribution over emotions.
    """
    pipeline, AutoTokenizer = _lazy_import_hf()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    clf = pipeline("text-classification", model=model_name, tokenizer=tok, top_k=None, device_map="auto")
    # compute label order from first item
    first = clf(texts[0], truncation=True)
    if isinstance(first, list) and first and isinstance(first[0], list):
        first = first[0]
    labels = [d["label"] for d in first]

    out = []
    for t in texts:
        res = clf(t, truncation=True)
        if isinstance(res, list) and res and isinstance(res[0], list):
            res = res[0]
        sc = {d["label"]: float(d["score"]) for d in res}
        out.append([sc.get(lb, 0.0) for lb in labels])
    E = np.asarray(out, dtype=float)
    return zscore(E, axis=0)

# ------------------ pairwise / dispersion ------------------

def mean_pairwise_distance(distance_matrix: np.ndarray) -> float:
    n = distance_matrix.shape[0]
    if n < 2:
        return 0.0
    idx = np.triu_indices(n, k=1)
    vals = distance_matrix[idx]
    return float(np.mean(vals)) if len(vals) else 0.0

def pairwise_dtw(seqs: List[List[float]]) -> np.ndarray:
    n = len(seqs)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = dtw_distance(seqs[i], seqs[j])
            D[i, j] = D[j, i] = d
    return D

# ------------------ SBERT dispersion (optional) ------------------

def sbert_dispersion(texts: List[str], model_name="all-MiniLM-L6-v2") -> float:
    try:
        SentenceTransformer = _lazy_import_sbert()
        model = SentenceTransformer(model_name)
        X = model.encode(texts)
        # mean pairwise cosine distance
        n = len(X)
        if n < 2:
            return 0.0
        ds = []
        for i in range(n):
            for j in range(i+1, n):
                xi, xj = X[i], X[j]
                cs = 1.0 - np.dot(xi, xj) / (np.linalg.norm(xi) * np.linalg.norm(xj) + 1e-8)
                ds.append(cs)
        return float(np.mean(ds)) if ds else 0.0
    except Exception:
        warnings.warn("SBERT not available; skipping semantic dispersion")
        return None

# ------------------ SDI ------------------

def normalize(x, lo, hi):
    if x is None: return None
    if hi <= lo + 1e-8: return 0.5
    z = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, z))

def compute_sdi(metrics: Dict[str, float]) -> float:
    """
    Combine metrics into a single Story Diversity Index.
    Weights can be adjusted as needed.
    """
    # choose robust ranges for normalization (heuristics; tune for your corpus)
    m = metrics.copy()
    m["distinct2_n"] = normalize(m.get("distinct2", 0.0), 0.0, 0.8)          # typical range 0~0.8
    m["selfbleu_n"]  = 1.0 - normalize(m.get("selfbleu", 0.0), 0.0, 0.8)     # lower is better
    m["emo_mpd_n"]   = normalize(m.get("emo_mpd", 0.0), 0.0, 60.0)            # DTW scale
    m["plot_mpd_n"]  = normalize(m.get("plot_mpd", 0.0), 0.0, 60.0)
    if metrics.get("sbert_mpd") is not None:
        m["sbert_mpd_n"] = normalize(m.get("sbert_mpd", 0.0), 0.0, 1.0)
    # weights
    weights = {
        "distinct2_n": 0.2,
        "selfbleu_n":  0.2,
        "emo_mpd_n":   0.25,
        "plot_mpd_n":  0.25,
        "sbert_mpd_n": 0.10
    }
    s = 0.0
    w = 0.0
    for k, wt in weights.items():
        if m.get(k) is not None:
            s += wt * m[k]
            w += wt
    return s / (w + 1e-8)

# ------------------ bootstrap CI ------------------

def bootstrap_mean(vals: List[float], B=1000, alpha=0.05, seed=0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return float('nan'), float('nan'), float('nan')
    means = []
    for _ in range(B):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(np.mean(sample))
    means = np.sort(means)
    lo = np.percentile(means, 100*alpha/2)
    hi = np.percentile(means, 100*(1-alpha/2))
    return float(np.mean(vals)), float(lo), float(hi)

# ------------------ visualization ------------------

def plot_heatmap(D: np.ndarray, labels: List[str], title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(D, interpolation="nearest")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_bar_with_ci(groups: List[str], means: List[float], los: List[float], his: List[float], title: str, ylabel: str, out_path: str):
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x, means)
    yerr = np.vstack([np.array(means)-np.array(los), np.array(his)-np.array(means)])
    ax.errorbar(x, means, yerr=yerr, fmt='none', ecolor='k', capsize=4, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='lower'))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ------------------ main compute per group ------------------

def compute_group_metrics(folder: str, args) -> Dict[str, Any]:
    # texts: use either aggregated story texts (join chapters) or .txt
    texts = []
    emotion_seqs = []   # list of [ [emo7] per chapter ], z-scored per story
    plot_seqs = []      # list of [values per chapter]
    story_ids = []

    # prefer JSON
    stories, ids = load_stories_from_jsons(folder, pattern=getattr(args, 'pattern', "*.json"), recursive=getattr(args, 'recursive', False), limit=getattr(args, 'limit', None))
    if stories:
        for rec in stories:
            story_ids.append(rec["id"])
            # text
            if rec.get("texts"):
                texts.append("\n".join([t for t in rec["texts"] if t]))
            else:
                texts.append("")
            # plot values
            plot = rec.get("values") or []
            plot_seqs.append([float(v) for v in plot])
            # emotion arc (compute from chapter texts if enabled)
            if args.compute_emotions and rec.get("texts"):
                try:
                    Ez = compute_emotion_arc_for_texts(rec["texts"], model_name=args.emotion_model)
                except Exception as e:
                    warnings.warn(f"emotion model failed for {rec['id']}: {e}")
                    Ez = None
            else:
                Ez = None
            if Ez is not None and Ez.size > 0:
                # reduce to 1D via per-step L2 norm (or first PC, but keep simple)
                emo_1d = np.linalg.norm(Ez, axis=1).tolist()
            else:
                emo_1d = []
            emotion_seqs.append(emo_1d)

    # if no JSON, fallback to TXT
    if not stories:
        txts = load_texts_from_folder(folder, pattern="*.txt")
        for i, t in enumerate(txts, 1):
            story_ids.append(f"txt_{i}")
            texts.append(t)
            # no arcs for .txt unless provided elsewhere
            plot_seqs.append([])
            emotion_seqs.append([])

    # --- lexical ---
    distinct2 = distinct_n_corpus(texts, n=2) if texts else 0.0
    sb = self_bleu(texts) if texts else 0.0

    # --- arcs: mean pairwise DTW ---
    emo_mpd = None
    if any(len(s)>1 for s in emotion_seqs):
        D_emo = pairwise_dtw(emotion_seqs)
        emo_mpd = mean_pairwise_distance(D_emo)
        if args.save_figs and len(story_ids) >= 2:
            plot_heatmap(D_emo, story_ids, f"{Path(folder).name} – Emotion DTW", str(Path(args.out_dir) / f"{Path(folder).name}_emo_DTW.png"))

    plot_mpd = None
    if any(len(s)>1 for s in plot_seqs):
        D_plot = pairwise_dtw(plot_seqs)
        plot_mpd = mean_pairwise_distance(D_plot)
        if args.save_figs and len(story_ids) >= 2:
            plot_heatmap(D_plot, story_ids, f"{Path(folder).name} – Plot DTW", str(Path(args.out_dir) / f"{Path(folder).name}_plot_DTW.png"))

    # --- optional semantic dispersion ---
    sbert_mpd = None
    if args.sbert:
        sbert_mpd = sbert_dispersion(texts)

    return {
        "group": Path(folder).name,
        "n_stories": len(story_ids),
        "distinct2": distinct2,
        "selfbleu": sb,
        "emo_mpd": emo_mpd,
        "plot_mpd": plot_mpd,
        "sbert_mpd": sbert_mpd,
    }

# ------------------ CLI ------------------

def main():
    parser = argparse.ArgumentParser(description="Compute cross-story diversity for one or more folders (groups).")
    parser.add_argument("folders", nargs="+", help="One or more folders *or files*; each is a group (e.g., mem0_temp0.4 mem0_temp0.8)")
    parser.add_argument("--pattern", default="*.json", help="Glob pattern for JSON files inside each folder (e.g., '404b-*.json')")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders recursively with the pattern")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N matched files per group")
    parser.add_argument("--out-dir", default="diversity_out", help="Output directory")
    parser.add_argument("--compute-emotions", action="store_true", help="Compute emotion arcs from chapter texts via HF model (j-hartmann/...)")
    parser.add_argument("--emotion-model", default="j-hartmann/emotion-english-distilroberta-base", help="HuggingFace emotion model name")
    parser.add_argument("--sbert", action="store_true", help="Also compute SBERT semantic dispersion (requires sentence-transformers)")
    parser.add_argument("--save-figs", action="store_true", help="Save distance heatmaps and bar charts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for fol in args.folders:
        print(f"[info] computing metrics for group: {fol}")
        rows.append(compute_group_metrics(fol, args))

    # compute SDI per group
    for r in rows:
        r["SDI"] = compute_sdi(r)

    # save CSV
    import csv
    csv_path = str(Path(args.out_dir) / "diversity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group","n_stories","distinct2","selfbleu","emo_mpd","plot_mpd","sbert_mpd","SDI"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] saved {csv_path}")

    # bar + CI charts for key metrics across groups
    if args.save_figs and len(rows) >= 1:
        groups = [r["group"] for r in rows]

        # SDI bars
        means = [r["SDI"] for r in rows]
        los   = means[:]  # no bootstrap per-group by default (we have only one SDI per group)
        his   = means[:]
        plot_bar_with_ci(groups, means, los, his, "Story Diversity Index (SDI)", "SDI (0~1)", str(Path(args.out_dir) / "SDI_by_group.png"))

        # Self-BLEU (lower better) & Distinct-2 (higher better)
        plot_bar_with_ci(groups, [r["selfbleu"] for r in rows], [r["selfbleu"] for r in rows], [r["selfbleu"] for r in rows],
                         "Self-BLEU (lower=more diverse)", "Self-BLEU", str(Path(args.out_dir) / "SelfBLEU_by_group.png"))
        plot_bar_with_ci(groups, [r["distinct2"] for r in rows], [r["distinct2"] for r in rows], [r["distinct2"] for r in rows],
                         "Distinct-2 (higher=more diverse)", "Distinct-2", str(Path(args.out_dir) / "Distinct2_by_group.png"))

        # Emotion / Plot MPD (may be None)
        if any(r["emo_mpd"] is not None for r in rows):
            vals = [ (r["emo_mpd"] if r["emo_mpd"] is not None else 0.0) for r in rows ]
            plot_bar_with_ci(groups, vals, vals, vals, "Emotion-arc MPD (higher=more diverse)", "DTW distance", str(Path(args.out_dir) / "EmoMPD_by_group.png"))
        if any(r["plot_mpd"] is not None for r in rows):
            vals = [ (r["plot_mpd"] if r["plot_mpd"] is not None else 0.0) for r in rows ]
            plot_bar_with_ci(groups, vals, vals, vals, "Plot-arc MPD (higher=more diverse)", "DTW distance", str(Path(args.out_dir) / "PlotMPD_by_group.png"))

    print("[done]")

if __name__ == "__main__":
    main()
