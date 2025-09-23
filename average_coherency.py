import os
import json
from collections import defaultdict
from test_coherency import evaluate_file

INFO = {
    "ppl3_add1": ("困惑度(越低越好讀)", "0~inf"),

    "PU": ("主語連貫性", None),
    "PW": ("賓語連貫性", None),
    "PACC": ("整體連貫性", None),
    "PU_dist": ("PU 距離量度", None),
    "PW_dist": ("PW 距離量度", None),
    "PACC_dist": ("PACC 距離量度", None),

    "jac_median": ("相鄰章實體重疊中位數", "0~1"),
    "jac_low_ratio": ("相鄰章幾乎不重疊的比例(越低越好)", "0~1"),
    "main_persist": ("主角持續度", "0~1"),

    "global_coherence": ("全域相似度", "-1~1"),
    "local_coherence": ("相鄰章和非相鄰章的相似度", "-1~1"),
    
    "nli_pairs": ("跨章比對的句子對數量", None),
    "nli_contradictions": ("矛盾的對數", "0~nli_pairs"),
    "nli_contradiction_rate": ("矛盾比", "0~1"),
}

def wrap(key, val):
    entry = {"value": round(val, 4)}
    if key in INFO:
        desc, rng = INFO[key]
        if desc: entry["explain"] = desc
        if rng:  entry["range"] = rng
    return entry

def main():
    base_dir = "data/mem0_404b"
    files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.json')]
    results = []
    for f in files:
        if os.path.exists(f):
            try:
                res = evaluate_file(f)
                results.append(res)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    if not results:
        print("No valid files found.")
        return

    sums = defaultdict(float)
    counts = defaultdict(int)
    coherence_sums = defaultdict(float)
    coherence_counts = defaultdict(int)
    
    for res in results:
        for key, value in res.items():
            if isinstance(value, (int, float)):
                sums[key] += value
                counts[key] += 1
            elif key == "coherence" and isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        coherence_sums[subkey] += subvalue
                        coherence_counts[subkey] += 1

    avg = {
        "length": {},
        "n_gram": {},
        "Perplexity": {},
        "consistency(TRUNAJOD)": {},
        "NER(Jaccard)": {},
        "Thematic coherence": {},
        "NLI": {}
    }

    for key, total in sums.items():
        if counts[key] == 0:
            continue
        v = total / counts[key]

        if key in ("tokens",):
            avg["length"][key] = wrap(key, v)

        elif key in ("rep6", "rep10"):
            avg["n_gram"][key] = wrap(key, v)

        elif key in ("ppl3_add1",):
            avg["Perplexity"][key] = wrap(key, v)

        elif key in ("jac_median", "jac_low_ratio", "main_persist"):
            avg["NER(Jaccard)"][key] = wrap(key, v)

        elif key in ("global_coherence", "local_coherence"):
            avg["Thematic coherence"][key] = wrap(key, v)

        elif key.startswith("nli_"):
            avg["NLI"][key] = wrap(key, v)

    if coherence_sums:
        for ckey, ctotal in coherence_sums.items():
            if coherence_counts[ckey] > 0:
                v = ctotal / coherence_counts[ckey]
                avg["consistency(TRUNAJOD)"][ckey] = wrap(ckey, v)

    os.makedirs("out_co", exist_ok=True)
    with open("out_co/mem0_avgCo_404b.json", "w", encoding="utf-8") as f:
        json.dump(avg, f, ensure_ascii=False, indent=2)

    print("Averaged results saved to out_co/mem0_avgCo_404b.json")

if __name__ == "__main__":
    main()