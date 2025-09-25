#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, glob, argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# fastdtw optional
try:
    from fastdtw import fastdtw
    _HAS_FDTW = True
except Exception:
    _HAS_FDTW = False

def resample_1d(seq, m=50):
    if not seq: return np.zeros(m)
    x = np.linspace(0,1,num=len(seq)); xi = np.linspace(0,1,num=m)
    return np.interp(xi, x, np.asarray(seq, float))

def dtw_distance(a, b):
    if not _HAS_FDTW:
        return float(np.linalg.norm(resample_1d(a,100)-resample_1d(b,100)))
    dist,_ = fastdtw(a, b, dist=lambda x,y: abs(float(x)-float(y)))
    return float(dist)

def pairwise_dtw(seqs):
    n = len(seqs); D = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i+1,n):
            d = dtw_distance(seqs[i], seqs[j]); D[i,j]=D[j,i]=d
    return D

def mean_pairwise(D):
    n = D.shape[0]
    if n < 2: return 0.0
    idx = np.triu_indices(n,1); vals = D[idx]
    return float(np.mean(vals)) if len(vals) else 0.0

def plot_heatmap(D, labels, title, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(D, interpolation="nearest")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

def _z_matrix(M: np.ndarray) -> np.ndarray:
    """對距離矩陣的上三角做 z-score，回寫到同尺度矩陣。"""
    M = np.asarray(M, float)
    if M.size == 0: 
        return M
    tri = np.triu_indices_from(M, k=1)
    v = M[tri]
    mu, sd = float(np.mean(v)), float(np.std(v))
    if sd == 0: sd = 1.0
    Z = (M - mu) / sd
    return Z

def load_bundle(path):
    with open(path, "r", encoding="utf-8") as f:
        b = json.load(f)
    if "stories" not in b or not isinstance(b["stories"], list):
        raise ValueError(f"{path} is not a valid bundle")
    return b

def plot_combo_heatmap(D_plot, D_emo, labels, title, out_path):
    # 兩個熱圖併成一張（左：Plot/values 右：Emotion/emo_pc1）
    from matplotlib import pyplot as plt
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2 = axes

    if D_plot is not None:
        im1 = ax1.imshow(D_plot, interpolation="nearest")
        ax1.set_title("Plot DTW")
        ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, rotation=90)
        ax1.set_yticks(range(len(labels))); ax1.set_yticklabels(labels)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    if D_emo is not None:
        im2 = ax2.imshow(D_emo, interpolation="nearest")
        ax2.set_title("Emotion DTW")
        ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, rotation=90)
        ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_one(bundle, args, current_path):
    D_plot = None
    D_emo = None

    group = Path(current_path).stem.replace("_bundle","")
    stories = bundle.get("stories", [])

    # values (fused)
    plot_seqs = [ s.get("values") for s in stories
                  if isinstance(s.get("values"), list) and len(s["values"]) > 1 ]
    # emotion PC1
    emo_seqs = [ (s.get("curves") or {}).get("emo_pc1") for s in stories
                 if isinstance((s.get("curves") or {}).get("emo_pc1"), list)
                 and len((s.get("curves") or {}).get("emo_pc1")) > 1 ]

    if len(stories) < 2:
        print(f"[warn] {group}: n_stories={len(stories)} < 2, metrics will be 0/None")

    # === DTW: values (Plot) ===
    plot_mpd = None
    if len(plot_seqs) >= 2:
        D_plot = pairwise_dtw(plot_seqs)
        plot_mpd = mean_pairwise(D_plot)
        if args.save_figs and not args.only_merged:
            plot_heatmap(
                D_plot,
                [str(i+1) for i in range(len(plot_seqs))],
                f"{group} – Plot DTW",
                str(Path(args.fig_dir) / f"{group}_plot_DTW.png"),
            )


    # === DTW: emo_pc1 (Emotion) ===
    emo_mpd = None
    if len(emo_seqs) >= 2:
        D_emo = pairwise_dtw(emo_seqs)
        emo_mpd = mean_pairwise(D_emo)
        if args.save_figs and not args.only_merged:
            plot_heatmap(
                D_emo,
                [str(i+1) for i in range(len(emo_seqs))],
                f"{group} – Emotion DTW",
                str(Path(args.fig_dir) / f"{group}_emo_DTW.png"),
            )

    # === Lexical JSD（讀回原始 JSON，把章節文字串起來）===
    lex_jsd = None
    texts = []
    for s in stories:
        src = s.get("file")
        if not src:
            continue
        try:
            data = json.loads(Path(src).read_text(encoding="utf-8"))
            chs = data.get("chapters") or []
            t = "\n".join([(c.get("text") or "").strip() for c in chs if (c.get("text") or "").strip()]) \
                if chs else (data.get("text") or "")
            if t.strip():
                texts.append(t.strip())
        except Exception:
            pass
    lex_jsd = mean_pairwise_jsd(texts) if len(texts) >= 2 else 0.0

    if args.save_figs and args.combo_merge and (D_plot is not None or D_emo is not None):
        try:
            w_plot, w_emo = [float(x) for x in args.merge_w.split(",")]
        except Exception:
            w_plot, w_emo = 0.5, 0.5
        if D_plot is None:
            D_merged = _z_matrix(D_emo)
        elif D_emo is None:
            D_merged = _z_matrix(D_plot)
        else:
            Zp = _z_matrix(D_plot); Ze = _z_matrix(D_emo)
            D_merged = w_plot * Zp + w_emo * Ze
        labels = [str(i+1) for i in range(max(len(plot_seqs), len(emo_seqs)))]
        plot_heatmap(
            D_merged, labels,
            f"{group} – Merged DTW (Plot+Emotion)",
            str(Path(args.fig_dir) / f"{group}_merged_DTW.png"),
        )



    return {
        "group": group,
        "n_stories": len(stories),
        # "distinct2": distinct2,
        # "selfbleu": selfbleu,
        "emo_mpd": emo_mpd,
        "plot_mpd": plot_mpd,
        # "sbert_mpd": sbert_mpd,
        "lex_jsd": lex_jsd,
    }

from collections import Counter
import numpy as np
from pathlib import Path
import json

def _build_unigram_matrix(texts):
    counters = [Counter(t.split()) for t in texts]
    vocab = {w for c in counters for w in c}
    if not vocab:
        return np.zeros((len(texts), 0), float)
    idx = {w:i for i,w in enumerate(sorted(vocab))}
    X = np.zeros((len(texts), len(idx)), float)
    for r,c in enumerate(counters):
        for w,f in c.items():
            X[r, idx[w]] = f
    return X

def _jsd(p, q, eps=1e-12):
    p = p + eps; q = q + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5*(p+q)
    def kl(a,b): return np.sum(a*np.log(a/b))
    return 0.5*kl(p,m) + 0.5*kl(q,m)

def mean_pairwise_jsd(texts):
    X = _build_unigram_matrix(texts)
    n = X.shape[0]
    if n < 2 or X.shape[1] == 0:
        return 0.0
    ds=[]
    for i in range(n):
        for j in range(i+1, n):
            ds.append(_jsd(X[i], X[j]))
    return float(np.mean(ds)) if ds else 0.0

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--folder", type=str, help="Directory containing bundle json files")
    g.add_argument("--bundles", type=str, nargs="+", help="Explicit bundle paths")
    ap.add_argument("--pattern", type=str, default="*.json", help="Glob pattern under --folder")
    ap.add_argument("--out-dir", type=str, default="diversity_out")
    ap.add_argument("--save-figs", action="store_true")
    ap.add_argument("--fig-dir", type=str, default=None, help="Heatmap output dir (default: <out-dir>/out_pic)")
    ap.add_argument("--require", type=int, default=None, help="Warn if bundle count != this number (e.g., 16)")
    ap.add_argument("--combo-fig", action="store_true",
                help="Save one figure with Plot-DTW and Emotion-DTW side by side")
    ap.add_argument("--combo-merge", action="store_true",
                help="輸出一張合併熱圖（Plot+Emotion）")
    ap.add_argument("--only-merged", action="store_true",
                    help="只輸出合併熱圖，不輸出各自熱圖")
    ap.add_argument("--merge-w", default="0.5,0.5",
                    help="合併權重 w_plot,w_emo（預設 0.5,0.5）")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.fig_dir is None:
        args.fig_dir = str(Path(args.out_dir) / "out_pic")
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    bundle_paths = (sorted(Path(args.folder).glob(args.pattern)) if args.folder
                    else [Path(p) for p in args.bundles])

    if args.require is not None and len(bundle_paths) != args.require:
        print(f"[warn] expected {args.require} bundles, got {len(bundle_paths)}")

    rows=[]
    for bp in bundle_paths:
        try:
            b = load_bundle(str(bp))
        except Exception as e:
            print(f"[error] skip {bp}: {e}")
            continue
        rows.append(compute_one(b, args, str(bp)))

    # CSV
    import csv
    out_csv = str(Path(args.out_dir)/"diversity.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group","n_stories","distinct2","selfbleu","emo_mpd","plot_mpd","sbert_mpd", "lex_jsd"])
        w.writeheader(); w.writerows(rows)
    print("[ok] saved", out_csv)
    if args.save_figs:
        print("[ok] heatmaps ->", args.fig_dir)

if __name__ == "__main__":
    main()
