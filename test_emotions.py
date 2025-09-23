import json, numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from transformers import pipeline, AutoTokenizer
from pathlib import Path

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
classifier = pipeline("text-classification", model=MODEL_NAME,
                      tokenizer=tok, top_k=None, device_map="auto")

def _unpack_all_scores(out):
    if isinstance(out, dict):
        return [out]
    if isinstance(out, list) and out and isinstance(out[0], list):
        return out[0]
    return out  

def process_json(json_path, out_dir, pen=12.0, model="rbf", min_size=2):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    chapters = data.get("chapters", [])
    texts, chapter_ids = [], []
    for i, c in enumerate(chapters, 1):
        t = (c.get("text") or "").strip()
        if t:
            texts.append(t)
            chapter_ids.append(str(c.get("id", i)))
    if len(texts) < 2:
        print(f"[skip] {json_path} (<2 chapters)")
        return

    #  7 類
    out0 = _unpack_all_scores(classifier(texts[0], truncation=True))
    labels = [d["label"] for d in out0]

    # 所有章節
    em = []
    for t in texts:
        res = _unpack_all_scores(classifier(t, truncation=True))
        sc = {d["label"]: float(d["score"]) for d in res}
        em.append([sc.get(lb, 0.0) for lb in labels])
    E = np.array(em, dtype=float)

    # z-score
    mu = E.mean(axis=0, keepdims=True)
    sd = E.std(axis=0, keepdims=True); sd[sd == 0] = 1.0
    Z = (E - mu) / sd
    n, d = Z.shape

    # PELT 
    ms = max(2, int(min_size)) if model == "rbf" else max(1, int(min_size))
    algo = rpt.Pelt(model=model, min_size=ms).fit(Z)
    bkps = algo.predict(pen=float(pen))
    if not bkps or bkps[-1] != n:
        bkps = [b for b in bkps if b <= n] + [n]
    print(f"[{Path(json_path).name}] pen={pen} -> cps={len(bkps)-1}, bkps={bkps}")

    # 繪圖
    x = np.arange(n)
    fig, axes = plt.subplots(d, 1, figsize=(12, 2*d), sharex=True)
    if d == 1: axes = [axes]
    colors = ["#DDEEFF", "#FFEDD5", "#E6FFE6", "#FFF0F6"]
    for i, ax in enumerate(axes):
        ax.plot(x, Z[:, i], marker='o', linewidth=1.5, label=labels[i])
        start = 0
        for j, end in enumerate(bkps):
            ax.axvspan(start-0.5, end-0.5, facecolor=colors[j % len(colors)], alpha=0.25, linewidth=0)
            start = end
        for b in bkps[:-1]:
            ax.axvline(b-0.5, linestyle='--', linewidth=1.2)
        ax.set_ylabel(labels[i], rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(chapter_ids)
    fig.suptitle(f"{Path(json_path).stem} – PELT pen={pen} (segments={len(bkps)})", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{Path(json_path).stem}_emo_pen{int(pen)}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] -> {out_png}")

def process_dir(in_dir, out_root="out_emo", pen=12.0, model="rbf", min_size=2, pattern="*.json"):
    from pathlib import Path
    in_dir = Path(in_dir)
    out_dir = Path(out_root)            
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(in_dir.glob(pattern)):
        process_json(p, out_dir, pen=pen, model=model, min_size=min_size)

if __name__ == "__main__":
    import sys
    in_dir  = sys.argv[1] if len(sys.argv) > 1 else "data/none_101a"
    pen     = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    out_dir = "out_emo/none_101a"         
    process_dir(in_dir, out_root=out_dir, pen=pen, model="rbf", min_size=2)
