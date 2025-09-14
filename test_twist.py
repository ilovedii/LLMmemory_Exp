import numpy as np
import os
import matplotlib.pyplot as plt
import ruptures as rpt
from matplotlib.patches import Patch

def _cap_params(n, k, min_size, model="rbf"):
    m = max(1, int(min_size))
    if model == "rbf" and m < 2:
        m = 2
    k_max = max(0, n // m - 1)   
    if k > k_max:
        print(f"[warn] k={k} > k_max={k_max} (n={n}, min_size={m})，自動下修為 {k_max}")
        k = k_max
    return k, m

def _ensure_last_n(bkps, n):
    if not bkps or bkps[-1] != n:
        bkps = [b for b in bkps if 0 < b < n] + [n]
    return bkps

def segment_plot_1d(values, chapter_ids=None, *, 
                    mode="k", k=1, pen=10.0, cuts=None,
                    min_size=2, model="rbf",
                    fig_path="seg1d.png", title=None):

    v = np.asarray(values).reshape(-1, 1)
    n = v.shape[0]
    if chapter_ids is None:
        chapter_ids = [str(i+1) for i in range(n)]
    if n == 0:
        raise ValueError("空資料")
    if n == 1:
        bkps = [1]
        seg_bounds = [(0, 1)]
        _plot_1d(v, bkps, chapter_ids, title or "Fixed k=0 (segments=1)", fig_path)
        print(f"[info] change-points: []  (segments=1)")
        return bkps

    if mode == "cuts":
        bkps = _ensure_last_n(sorted(cuts or []), n)
        used_k, used_min = len(bkps)-1, min_size
    elif mode == "pen":
        # PELT 
        bkps = rpt.Pelt(model=model, min_size=max(1, min_size)).fit(v).predict(pen=pen)
        bkps = _ensure_last_n(bkps, n)
        used_k, used_min = len(bkps)-1, min_size
    elif mode == "k":
        # 固定刀數
        k, m = _cap_params(n, int(k), min_size, model=model)
        if k == 0:
            bkps = [n]
        else:
            try:
                bkps = rpt.Binseg(model=model, min_size=m).fit(v).predict(n_bkps=k)
            except Exception:
                bkps = rpt.Dynp(model=model, min_size=m).fit(v).predict(n_bkps=k)
        bkps = _ensure_last_n(bkps, n)
        used_k, used_min = k, m
    else:
        raise ValueError("mode 必須是 'k'、'pen' 或 'cuts'")

    #  繪圖 
    if title is None:
        if mode == "k":
            title = f"Fixed k={used_k} (segments={used_k+1}), min_size={used_min}"
        elif mode == "pen":
            title = f"PELT pen={pen} (segments={len(bkps)})"
        else:
            title = f"Manual cuts {bkps[:-1]} (segments={len(bkps)})"

    _plot_1d(v, bkps, chapter_ids, title, fig_path)

    cps = bkps[:-1]
    print(f"[info] change-points: {cps}  (segments={len(bkps)})")
    for j, cp in enumerate(cps, 1):
        li, ri = cp-1, cp
        print(f"  cp{j} = {cp}  between chapter_ids[{li}]={chapter_ids[li]} and chapter_ids[{ri}]={chapter_ids[ri]}")
    print(f"saved figure -> {fig_path}")
    return bkps

def _plot_1d(v, bkps, chapter_ids, title, fig_path):
    n = v.shape[0]
    ends = bkps                      # e.g. [cp1, cp2, n]
    seg_bounds = list(zip([0] + ends[:-1], ends))
    cmap = plt.get_cmap("tab20")
    seg_colors = [cmap(i % cmap.N) for i in range(len(seg_bounds))]

    x = np.arange(n)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), sharex=True)
    ax.plot(x, v[:, 0], marker='o', linewidth=1.6)

    # 每段不同底色 
    for (s, e), color in zip(seg_bounds, seg_colors):
        ax.axvspan(s - 0.5, e - 0.5, alpha=0.25, facecolor=color, linewidth=0)
    for b in ends[:-1]:
        ax.axvline(b - 0.5, linestyle='--', linewidth=1.2)

    ax.set_ylabel("Value", rotation=0, ha="right", va="center")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(chapter_ids, rotation=0)

    patches = [Patch(facecolor=seg_colors[i], edgecolor='none',
                     label=f"Seg {i+1}: {chapter_ids[s]}-{chapter_ids[e-1]}")
               for i, (s, e) in enumerate(seg_bounds)]
    fig.legend(handles=patches, loc='upper center',
               ncol=min(6, len(patches)), frameon=False, bbox_to_anchor=(0.5, 1.05))

    plt.suptitle(title, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
import os, json, glob, math
import numpy as np
from typing import Tuple, List, Any


def _coerce_float(x: Any) -> float | None:
    try:
        if x is None: return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None

def _extract_from_chapter(ch: dict) -> float | None:
    for key in ("value", "pc1", "score"):
        v = _coerce_float(ch.get(key))
        if v is not None:
            return v
    # 退回：用字數作為一維值
    txt = (ch.get("text") or "").strip()
    if txt:
        return float(len(txt.split()))
    return None

def load_values_from_json(json_path: str) -> Tuple[List[float], List[str], str]:
  
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    story_id = str(data.get("story_id") or os.path.splitext(os.path.basename(json_path))[0])

    if "values" in data:
        values = data["values"]
        chapter_ids = data.get("chapter_ids") or [str(i + 1) for i in range(len(values))]
        values = [float(x) for x in values]  # 轉 float
        return values, chapter_ids, story_id

    chs = data.get("chapters") or []
    if not isinstance(chs, list) or len(chs) == 0:
        raise ValueError(f"{json_path} 沒有 values，也沒有 chapters 可萃取")

    values: List[float] = []
    chapter_ids: List[str] = []
    for i, ch in enumerate(chs, start=1):
        v = _extract_from_chapter(ch)
        if v is None:
            v = 0.0
        values.append(float(v))
        chapter_ids.append(str(ch.get("id") or i))

    return values, chapter_ids, story_id

def run_one(json_path: str,
            mode="k", k=1, pen=10.0, min_size=1,
            out_dir="out") -> str:
    values, chapter_ids, story_id = load_values_from_json(json_path)
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"{story_id}_twist.png")

    title = f"{story_id} – mode={mode}" + (f", k={k}" if mode=="k" else f", pen={pen}")
    segment_plot_1d(
        values,
        chapter_ids,
        mode=mode,
        k=k,
        pen=pen,
        min_size=min_size,
        fig_path=fig_path,
        title=title
    )
    print("saved figure ->", os.path.abspath(fig_path))
    return fig_path

def run_folder(folder: str,
               pattern: str = "*.json",
               **kwargs) -> None:
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        print(f"[warn] {folder} 下找不到 {pattern}")
        return
    print(f"[info] 批次處理 {len(paths)} 個 JSON")
    for p in paths:
        try:
            print(f"\n=== {p} ===")
            run_one(p, **kwargs)
        except Exception as e:
            print(f"[error] {p}: {e}")

if __name__ == "__main__":
    # 跑單一檔
    # run_one("mem0_temp0.4/404a-1.json", mode="k", k=1, min_size=1)

    # 整個資料夾
    run_folder("mem0_temp0.4", pattern="*.json", mode="k", k=1, min_size=1)
