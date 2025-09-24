import os, json, glob, argparse, sys
from typing import List, Tuple, Optional

import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Embeddings ---
def _lazy_import_st():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception as e:
        print("[error] sentence-transformers not installed. Please run:", file=sys.stderr)
        print("   pip install sentence-transformers", file=sys.stderr)
        raise

def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> np.ndarray:
    SentenceTransformer = _lazy_import_st()
    model = SentenceTransformer(model_name, device=device)
    clean = [(t or "").strip() for t in texts]
    embs = model.encode(clean, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embs.astype(np.float32)

def novelty_from_embeddings(embs: np.ndarray) -> np.ndarray:
    """
    v_t = 1 - cos(e_t, e_{t-1}),  v_1 = 0
    """
    n = embs.shape[0]
    v = np.zeros(n, dtype=np.float32)
    if n <= 1:
        return v
    dots = np.sum(embs[1:] * embs[:-1], axis=1)
    v[1:] = 1.0 - np.clip(dots, -1.0, 1.0)
    return v

def load_chapters(json_path: str) -> Tuple[List[str], List[str], str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    story_id = str(data.get("story_id") or os.path.splitext(os.path.basename(json_path))[0])
    chs = data.get("chapters") or []
    if not isinstance(chs, list) or len(chs) == 0:
        raise ValueError(f"{json_path} has no 'chapters' array.")
    texts, ids = [], []
    for i, ch in enumerate(chs, start=1):
        txt = (ch.get("text") or "").strip()
        if not txt:
            raise ValueError(f"{json_path} chapter {i} has empty 'text'; cannot compute embeddings without text.")
        texts.append(txt)
        ids.append(str(ch.get("id") or i))
    return texts, ids, story_id

# --- Segmentation + plotting ---
def _ensure_last_n(bkps, n):
    if not bkps or bkps[-1] != n:
        bkps = [b for b in bkps if 0 < b < n] + [n]
    return bkps

def segment_and_plot(values: np.ndarray,
                     chapter_ids: List[str],
                     *, mode: str = "pen", pen: float = 0.2,
                     min_size: int = 2, model: str = "rbf",
                     fig_path: str = "seg1d.png", title: Optional[str] = None) -> List[int]:

    v = np.asarray(values).reshape(-1, 1)
    n = v.shape[0]
    if n == 0:
        raise ValueError("empty series")

    if mode == "pen":
        bkps = rpt.Pelt(model=model, min_size=max(1, min_size)).fit(v).predict(pen=pen)
    else:
        raise ValueError("Only 'pen' mode is supported in this script.")
    bkps = _ensure_last_n(bkps, n)

    if title is None:
        title = f"PELT pen={pen} (segments={len(bkps)}) – signal=embed_novelty"

    x = np.arange(n)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(x, v[:, 0], marker='o', linewidth=1.6)

    ends = bkps
    seg_bounds = list(zip([0] + ends[:-1], ends))
    colors = ["#DDEEFF", "#FFEDD5", "#E6FFE6", "#FFF0F6", "#FFF9DD", "#E0E0FF", "#FFDDEE"]
    for i, (s, e) in enumerate(seg_bounds):
        ax.axvspan(s - 0.5, e - 0.5, alpha=0.25, facecolor=colors[i % len(colors)], linewidth=0)
    for b in ends[:-1]:
        ax.axvline(b - 0.5, linestyle='--', linewidth=1.2)

    ax.set_ylabel("Novelty", rotation=0, ha="right", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(chapter_ids)
    ax.grid(True, linewidth=0.5, alpha=0.4)

    # import matplotlib.patches as mpatches
    # patches = [mpatches.Patch(alpha=0.25, facecolor=colors[i % len(colors)],
    #                           label=f"Seg {i+1}: {chapter_ids[s]}-{chapter_ids[e-1]}")
    #            for i, (s, e) in enumerate(seg_bounds)]
    # fig.legend(handles=patches, loc='upper center',
    #            ncol=min(6, len(patches)), frameon=False, bbox_to_anchor=(0.5, 1.05))

    plt.suptitle(title, y=0.92, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return bkps

def run_file(json_path: str,
             pen: float = 0.2,
             model_name: str = "all-MiniLM-L6-v2",
             device: Optional[str] = None,
             out_dir: str = "out_twist",
             min_size: int = 2) -> str:
    texts, chapter_ids, story_id = load_chapters(json_path)
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"{story_id}_twist.png")

    embs = embed_texts(texts, model_name=model_name, device=device)
    novelty = novelty_from_embeddings(embs)

    mu, sd = float(np.mean(novelty)), float(np.std(novelty))
    novelty_z = (novelty - mu) / sd if sd > 0 else novelty

    title = f"With mem0 Plot analyze"
    segment_and_plot(
        novelty_z, chapter_ids,
        mode="pen", pen=pen, min_size=min_size, model="rbf",
        fig_path=fig_path, title=title
    )
    print("saved figure ->", os.path.abspath(fig_path))
    return fig_path

def run_folder(folder: str, pattern: str = "*.json", **kwargs) -> None:
    import glob
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    print(f"[info] {folder} files: {len(paths)}")
    for p in paths:
        try:
            print(f"=== {p} ===")
            run_file(p, **kwargs)
        except Exception as e:
            print(f"[error] {p}: {e}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str)
    g.add_argument("--folder", type=str)
    ap.add_argument("--pattern", type=str, default="*.json")
    ap.add_argument("--pen", type=float, default=0.2)
    ap.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--min_size", type=int, default=2)
    ap.add_argument("--out_dir", type=str, default="out_twist")
    args = ap.parse_args()

    if args.file:
        run_file(args.file, pen=args.pen, model_name=args.model, device=args.device, out_dir=args.out_dir, min_size=args.min_size)
    else:
        run_folder(args.folder, pattern=args.pattern, pen=args.pen, model_name=args.model, device=args.device, out_dir=args.out_dir, min_size=args.min_size)

if __name__ == "__main__":
    main()
