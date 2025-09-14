import os
import sys
import json
import csv
from typing import List, Tuple, Dict
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# Optional TRUNAJOD coherence
_TRUNAJOD_AVAILABLE = True
try:
    from TRUNAJOD.entity_grid import EntityGrid, get_local_coherence
except Exception:
    _TRUNAJOD_AVAILABLE = False
from collections import Counter
import math

try:
    import spacy
    def _load_spacy():
        try:
            return spacy.load("en_core_web_md")
        except Exception:
            return spacy.load("en_core_web_sm")
    _NLP = _load_spacy()
except Exception:
    _NLP = None
    import re as _re_tok
    def _simple_tokenize(text: str):
        return _re_tok.findall(r"[A-Za-z0-9']+", text.lower())

# ---------------- I/O ----------------
def read_story_json(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chapters = [c.get("text", "") for c in data.get("chapters", []) if c.get("text")]
    return " ".join(chapters)

def read_chapters(path: str) -> List[str]:
    """Return a list of chapter texts from the story JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 僅回傳有文字的章節
    return [c.get("text", "") for c in data.get("chapters", []) if c.get("text")]

def scan_json_files(root: str) -> List[str]:
    targets = []
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".json"):
                    targets.append(os.path.join(dirpath, fn))
    else:
        targets.append(root)
    return sorted(targets)

# ---------------- Tokenization ----------------

def compute_coherence(text: str, nlp=None) -> dict:
    """Return TRUNAJOD local coherence metrics if TRUNAJOD is available, else {}."""
    if not _TRUNAJOD_AVAILABLE:
        return {}
    if nlp is None:
        # Try to reuse spaCy if already loaded
        try:
            nlp = _NLP
        except NameError:
            nlp = None
    if nlp is None:
        return {}
    doc = nlp(text)
    egrid = EntityGrid(doc)
    sc = get_local_coherence(egrid)
    return {
        "PU": sc[0],
        "PW": sc[1],
        "PACC": sc[2],
        "PU_dist": sc[3],
        "PW_dist": sc[4],
        "PACC_dist": sc[5],
    }

def tokenize(text: str) -> List[str]:
    if _NLP is None:
        return _simple_tokenize(text)
    doc = _NLP(text)
    toks = [t.lemma_.lower() if t.lemma_ else t.lower_ for t in doc if t.is_alpha]
    if len(toks) < 5:
        toks = [w for w in text.lower().split() if any(ch.isalnum() for ch in w)]
    return toks

# ---------------- Repetition (high-n) ----------------
def high_ngram_repetition_rate_tokens(tokens: List[str], n: int = 6) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    cnt = Counter(grams)
    total = len(grams)
    repeated = sum(c for c in cnt.values() if c > 1)
    return repeated / total if total else 0.0

# ---------------- N-gram LM (add-one) & Perplexity ----------------
def build_ngram_counts(tokens: List[str], n: int = 3) -> Tuple[Counter, Counter, int]:
    V = len(set(tokens)) or 1
    if n == 1:
        return Counter(tokens), Counter(), V
    pad = ["<s>"] * (n - 1)
    seq = pad + tokens
    ngrams = [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    histories = [g[:-1] for g in ngrams]
    return Counter(ngrams), Counter(histories), V

def ngram_perplexity_addone(train_tokens: List[str], test_tokens: List[str], n: int = 3) -> float:
    if len(test_tokens) == 0:
        return float("inf")
    ngrams_c, hist_c, V = build_ngram_counts(train_tokens, n=n)
    pad = ["<s>"] * (n - 1)
    seq = pad + test_tokens
    ngrams_test = [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    log2_sum = 0.0
    for g in ngrams_test:
        h = g[:-1]
        num = ngrams_c.get(g, 0) + 1
        den = hist_c.get(h, 0) + V
        p = num / den
        if p <= 0.0:
            p = 1.0 / (den if den > 0 else V)
        log2_sum += math.log2(p)
    M = len(ngrams_test)
    H = - log2_sum / max(M, 1)
    return 2.0 ** H

def heldout_split(tokens: List[str], ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    if len(tokens) < 10:
        return tokens, tokens 
    cut = int(len(tokens) * ratio)
    return tokens[:cut], tokens[cut:]
    
# --- Helpers for cross-chapter continuity ---
def chapter_entities(doc):
    keep = {"PERSON","ORG","GPE","LOC","FAC","WORK_OF_ART"}
    return set(ent.text for ent in doc.ents if ent.label_ in keep)

def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

def main_person(doc):
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if not persons: return None
    from collections import Counter
    return Counter(persons).most_common(1)[0][0]

def continuity_metrics(docs):
    ents = [chapter_entities(d) for d in docs]
    jac = [jaccard(ents[i], ents[i+1]) for i in range(len(ents)-1)]
    leaders = [main_person(d) for d in docs]
    persist = sum(1 for i in range(len(leaders)-1)
                  if leaders[i] and leaders[i]==leaders[i+1]) / max(1, len(leaders)-1)

    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics import roc_auc_score

        vecs = np.vstack([d.vector for d in docs])
        adj = [cosine_similarity(vecs[i:i+1], vecs[i+1:i+2])[0,0]
               for i in range(len(vecs)-1)]
        mean_adj = float(np.mean(adj)) if adj else 0.0

        pairs = [(i, j) for i in range(len(vecs)) for j in range(i+2, len(vecs))]
        if pairs:
            nonadj = [cosine_similarity(vecs[i:i+1], vecs[j:j+1])[0,0] for (i,j) in pairs]
            mean_nonadj = float(np.mean(nonadj))
        else:
            nonadj = []
            mean_nonadj = 0.0

        threadness_raw = mean_adj - mean_nonadj

    except Exception:
        threadness_raw = 0.0

    return {
        "jac_median": (sorted(jac)[len(jac)//2] if jac else 0.0),
        "jac_low_ratio": (sum(1 for x in jac if x < 0.10) / max(1, len(jac))),
        "main_persist": persist,
        "threadness": threadness_raw,  
    }

# ---------------- Consistency (NLI) helpers ----------------
def _majority_protagonist(docs):
    """用多數決找出章節中最常出現的主角名字（依你既有的 main_person）。"""
    names = [main_person(d) for d in docs]
    counts = {}
    for n in names:
        if not n:
            continue
        counts[n] = counts.get(n, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda x: x[1])[0]

def _split_sents(text: str, nlp=None):
    """優先用 spaCy 斷句；失敗改用簡單正則。"""
    if nlp:
        try:
            return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        except Exception:
            pass
    import re
    sents = re.split(r'(?<=[.!?。！？])\s+', text or "")
    return [s.strip() for s in sents if s.strip()]

def _extract_protagonist_claims(chap_texts, protagonist: str, nlp=None, max_per_chapter: int = 5):
    """Strict extraction: keep only static-attribute sentences about the protagonist."""
    import re

    def normalize_name(s: str, name: str) -> str:
        if not name:
            return s
        text = s
        nl = name.lower()
        if nl.startswith("thom"):
            text = re.sub(r"\bTomas\b", "Thomas", text, flags=re.IGNORECASE)
        return text

    prot_lower = protagonist.lower() if protagonist else ""
    per_chapter = []

    # 要排除：對話/報導動詞/效果詞/時間詞/動作動詞（避免劇情推進被當矛盾）
    dialogue_markers = [" said ", " says ", " told ", " asked ", " replied ", " whispered ", " shouted ", " exclaimed "]
    effect_markers   = [" seemed ", " appears ", " felt ", " glowing ", " light ", " breeze ", " wind ", " magic ", " portal "]
    time_markers     = [" then ", " later ", " after ", " afterwards ", " eventually ", " finally ", " soon "]
    action_verbs     = [" dropped ", " played ", " playing ", " began ", " opened ", " watched ", " entered ", " approached ",
                        " turned ", " walked ", " ran ", " shouted ", " fought ", " slammed ", " closed ", " pulled ", " pushed "]

    # 僅允許：家庭/職業/居住/出身/年齡等靜態線索
    family_terms     = [" brother", " sister", " father", " mother", " parents", " daughter", " son"]
    profession_terms = [" student", " teacher", " guard", " musician", " carpenter", " farmer", " baker", " knight", " lord", " maid"]
    location_patterns= [" lives in", " lives at", " is from", " comes from", " born in", " born at", " from "]
    age_patterns     = [" years old", " year-old"]

    for t in chap_texts:
        sents = _split_sents(t, nlp)
        picks = []
        for s in sents:
            s_norm = normalize_name(s, protagonist)
            sl = s_norm.lower()
            if '"' in s or '“' in s or '”' in s:
                continue
            if any(m in sl for m in (dialogue_markers + effect_markers + time_markers + action_verbs)):
                continue
            if not prot_lower or prot_lower not in sl:
                continue

            keep = False
            if (" is " in sl or " was " in sl):
                if not re.search(r"\bis\s+(playing|running|walking|fighting|opening|closing|watching)\b", sl):
                    keep = True

            if (" has " in sl or " have " in sl):
                keep = True

            if any(term in sl for term in (family_terms + profession_terms)):
                keep = True
            if any(p in sl for p in (location_patterns + age_patterns)):
                keep = True

            if keep:
                picks.append(s_norm)

            if len(picks) >= max_per_chapter:
                break

        per_chapter.append(picks)
    return per_chapter

# ---------------- Consistency (NLI) core ----------------
def nli_contradiction_metrics(chap_texts, docs) -> dict:
    protagonist = _majority_protagonist(docs)
    if not protagonist:
        return {}

    claims_per_ch = _extract_protagonist_claims(chap_texts, protagonist, _NLP, max_per_chapter=5)

    max_gap = 2
    pairs, idx_pairs = [], []
    for i in range(len(claims_per_ch)):
        for j in range(i+1, min(len(claims_per_ch), i+1+max_gap)):
            for p in claims_per_ch[i]:
                for h in claims_per_ch[j]:
                    pairs.append((p, h))
                    idx_pairs.append((i, j))
    if not pairs:
        return {}

    try:
        from transformers import pipeline
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

        for model_name in ["roberta-large-mnli", "distilroberta-base-mnli"]:
            try:
                nli = pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=None)
                break
            except Exception:
                nli = None
        if nli is None:
            return {}

        contradictions, total = 0, 0
        examples = []  # (score, i, j, prem, hyp)
        BATCH, TH = 16, 0.70

        for k in range(0, len(pairs), BATCH):
            chunk = pairs[k:k+BATCH]
            inputs = [{"text": prem, "text_pair": hyp} for prem, hyp in chunk]
            outputs = nli(inputs, truncation=True, max_length=256)
            for off, scores in enumerate(outputs):
                lab2p = {d["label"].upper(): d["score"] for d in scores}
                p_contra = lab2p.get("CONTRADICTION")
                if p_contra is None and "LABEL_0" in lab2p and "LABEL_2" in lab2p:
                    p_contra = lab2p["LABEL_0"]  # 多數 MNLI 標籤順序
                total += 1
                i, j = idx_pairs[k + off]
                prem, hyp = pairs[k + off]
                if p_contra is not None:
                    examples.append((p_contra, i, j, prem, hyp))
                    if p_contra >= TH:
                        contradictions += 1

        if total == 0:
            return {}

        examples.sort(key=lambda x: x[0], reverse=True)
        return {
            "nli_protagonist": protagonist,
            "nli_pairs": total,
            "nli_contradictions": contradictions,
            "nli_contradiction_rate": contradictions / total,
            "nli_top_contradictions": examples[:5],
        }
    except Exception:
        return {}

# ---------------- Reporting ----------------
def evaluate_file(path: str) -> Dict:
    text = read_story_json(path)
    toks = tokenize(text)

    # Repetition
    rep6  = high_ngram_repetition_rate_tokens(toks, n=6)
    rep10 = high_ngram_repetition_rate_tokens(toks, n=10)

    # PPL
    tr, te = heldout_split(toks, ratio=0.8)
    ppl3 = ngram_perplexity_addone(tr, te, n=3)

    # Coherence
    coh = compute_coherence(text, nlp=_NLP)

    # 章與章之間
    chap_texts = read_chapters(path)
    docs = [_NLP(t) for t in chap_texts] if _NLP else []
    cont = continuity_metrics(docs) if docs else {
        "jac_median": 0.0,
        "jac_low_ratio": 0.0,
        "main_persist": 0.0,
        "threadness": 0.0,
    }

    res = {
        "file": path,
        "tokens": len(toks),
        "rep6": rep6,
        "rep10": rep10,
        "ppl3_add1": ppl3,
        "coherence": coh,
    }
    res.update(cont) 
    try:
        nli_stats = nli_contradiction_metrics(chap_texts, docs) if docs else {}
        if nli_stats:
            res.update(nli_stats)
    except Exception:
        pass

    return res

def print_report(res: Dict, simple: bool = False):
    if simple:
        print(f"{res['file']}\trep6={res['rep6']:.3f}\trep10={res['rep10']:.3f}\tppl3_add1={res['ppl3_add1']:.2f}\ttokens={res['tokens']}")
        return
    print(f"\n=== {res['file']} ===")
    print("Repetition (6g / 10g) :", f"{res['rep6']:.3f} / {res['rep10']:.3f}")
    print("Perplexity:", f"{res['ppl3_add1']:.2f}")
    print(f"Entity-overlap Jaccard (median) : {res.get('jac_median',0.0):.3f}  | low<0.10 ratio: {res.get('jac_low_ratio',0.0):.2f}")
    print(f"Main-character persistence      : {res.get('main_persist',0.0):.2f}")
    print(f"Threadness (adj - nonadj sim)   : {res.get('threadness',0.0):.3f}")

    coh = res.get("coherence") or {}
    if coh:
        print("\n=== TRUNAJOD Local Coherence ===")
        for k in ["PU","PW","PACC","PU_dist","PW_dist","PACC_dist"]:
            print(f"{k:10s}: {coh.get(k):.3f}")

    if 'nli_contradiction_rate' in res:
        print("\n=== NLI Consistency ===")
        print(f"Protagonist                     : {res.get('nli_protagonist','?')}")
        print(f"Pairs evaluated                 : {res.get('nli_pairs',0)}")
        print(f"Contradictions                  : {res.get('nli_contradictions',0)}")
        print(f"Contradiction rate              : {res.get('nli_contradiction_rate',0.0):.3f}")
        
        tops = res.get("nli_top_contradictions") or []
            if tops:
                print("\nTop contradictions (score, ch_i→ch_j):")
                for sc, i, j, prem, hyp in tops:
                    print(f"  {sc:.2f}  ({i}->{j})")
                    print(f"     Prem: {prem[:120]}")
                    print(f"     Hyp : {hyp[:120]}")

def write_csv(rows: List[Dict], out_path: str):
    fieldnames = ["file", "tokens", "rep6", "rep10", "ppl3_add1"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_coherency.py <path-to-json-or-dir> [--simple]")
        sys.exit(1)
    simple = False
    if "--simple" in sys.argv:
        simple = True
        sys.argv.remove("--simple")

    target = sys.argv[1]
    files = scan_json_files(target)
    if not files:
        print("No JSON files found:", target)
        sys.exit(1)

    results = []
    for fpath in files:
        try:
            res = evaluate_file(fpath)
            print_report(res, simple=simple)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] {fpath}: {e}")

    if os.path.isdir(target):
        out_csv = os.path.join(target, "repetition_ppl_summary.csv")
        write_csv(results, out_csv)
        print(f"\nSummary written to: {out_csv}")

if __name__ == "__main__":
    main()
