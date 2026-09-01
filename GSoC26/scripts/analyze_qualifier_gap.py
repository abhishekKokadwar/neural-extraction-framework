#!/usr/bin/env python3
"""Offline audit of qualifier-suffix mismatches in the frozen v14-final sweep.

Counts gold triples that the pipeline missed ONLY because a URI was emitted
without its parenthesised disambiguator (or with a spurious one), and splits
them by whether the disambiguator word actually appears in the source sentence
-- i.e. whether it is recoverable from context at all.

No network, no LLM, no Redis. Reads results/final_clean/ only.

Run from GSoC26/:  python3 analyze_qualifier_gap.py
"""
import glob
import json
import os
import re
from collections import Counter

RESULTS = "results/final_clean"


def normalize_triple(sub, rel, obj):
    """Verbatim port of the official Text2KGBench run_eval.py normalisation,
    matching src/text2kg_harness.py:normalize_triple."""
    f = lambda x: re.sub(r"(_|\s+)", "", str(x)).lower()
    return f(sub) + f(rel) + f(obj)


def as_set(triples):
    return {normalize_triple(t["sub"], t["rel"], t["obj"]) for t in triples}


QUAL_RE = re.compile(r"^(.*)_\(([^)]*)\)$")


def main():
    missing = []   # gold wants _(qualifier), pred emitted bare
    spurious = []  # pred emitted _(qualifier), gold wants bare

    files = sorted(f for f in glob.glob(os.path.join(RESULTS, "results_*.json"))
                   if "__" not in os.path.basename(f))
    if not files:
        raise SystemExit(f"no result files under {RESULTS}/ -- run from GSoC26/")

    for path in files:
        domain = os.path.basename(path)[len("results_"):-len(".json")]
        for idx, row in enumerate(json.load(open(path, encoding="utf-8"))):
            gold_t, pred_t = row.get("gold", []), row.get("pred", [])
            gold, pred = as_set(gold_t), as_set(pred_t)
            sentence = row.get("sent", "")

            # gold carries a qualifier the prediction dropped
            for gt in gold_t:
                if normalize_triple(gt["sub"], gt["rel"], gt["obj"]) in pred:
                    continue
                for field in ("sub", "obj"):
                    m = QUAL_RE.match(str(gt[field]))
                    if not m:
                        continue
                    base, qual = m.group(1), m.group(2)
                    stripped = dict(gt, **{field: base})
                    if normalize_triple(stripped["sub"], stripped["rel"],
                                        stripped["obj"]) in pred:
                        in_sentence = qual.replace("_", " ").lower() in sentence.lower()
                        missing.append((domain, idx, field, base, gt[field],
                                        qual, gt["rel"], in_sentence))

            # prediction invented a qualifier gold does not have
            for pt in pred_t:
                if normalize_triple(pt["sub"], pt["rel"], pt["obj"]) in gold:
                    continue
                for field in ("sub", "obj"):
                    m = QUAL_RE.match(str(pt[field]))
                    if not m:
                        continue
                    stripped = dict(pt, **{field: m.group(1)})
                    if normalize_triple(stripped["sub"], stripped["rel"],
                                        stripped["obj"]) in gold:
                        spurious.append((domain, idx, field, pt[field],
                                         m.group(1), pt["rel"]))

    print(f"Frozen sweep: {len(files)} domains under {RESULTS}/\n")

    print(f"A. GOLD-QUALIFIED, PRED-BARE  ->  {len(missing)} gold triples missed")
    recoverable = [c for c in missing if c[7]]
    print(f"   of which the qualifier word appears in the sentence: "
          f"{len(recoverable)}\n")

    by_domain = Counter(c[0] for c in missing)
    print("   by domain:")
    for dom, n in by_domain.most_common():
        rec = sum(1 for c in missing if c[0] == dom and c[7])
        print(f"     {dom:26} {n:3}   recoverable from sentence: {rec}")

    print("\n   by qualifier token:")
    by_qual = Counter(c[5] for c in missing)
    for qual, n in by_qual.most_common():
        rec = sum(1 for c in missing if c[5] == qual and c[7])
        print(f"     _({qual}){'':<{max(0, 24 - len(qual))}} {n:3}   "
              f"recoverable: {rec}")

    print(f"\nB. PRED-QUALIFIED, GOLD-BARE  ->  {len(spurious)} triples lost to a "
          f"spurious suffix")
    for dom, n in Counter(c[0] for c in spurious).most_common():
        print(f"     {dom:26} {n:3}")

    print("\n--- sample of recoverable cases (A) ---")
    for c in recoverable[:12]:
        print(f"  {c[0]:24} #{c[1]:<4} {c[2]}: {c[3]}  ->gold->  {c[4]}   (rel={c[6]})")


if __name__ == "__main__":
    main()
