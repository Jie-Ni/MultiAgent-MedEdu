"""Post-experiment analysis: compute all fidelity metrics from generated dialogues."""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict, Counter
from loguru import logger
import numpy as np
from scipy import stats


def load_all_dialogues(results_dir: str) -> list:
    """Load all dialogue JSON files from results directory."""
    files = glob.glob(os.path.join(results_dir, "dialogues_*.json"))
    all_dialogues = []
    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)
            all_dialogues.extend(data)
    logger.info(f"Loaded {len(all_dialogues)} dialogues from {len(files)} files")
    return all_dialogues


def compute_linguistic_fidelity(dialogue: dict) -> dict:
    """Compute linguistic metrics for student utterances."""
    student_msgs = [m["content"] for m in dialogue["messages"] if m["role"] == "student"]
    if not student_msgs:
        return {"ttr": 0, "med_term_density": 0, "avg_sentence_len": 0, "question_ratio": 0}

    all_text = " ".join(student_msgs)
    words = re.findall(r'\b\w+\b', all_text.lower())

    # Type-Token Ratio
    ttr = len(set(words)) / max(len(words), 1)

    # Medical terminology density (simplified: count common medical terms)
    med_terms = {
        'diagnosis', 'symptom', 'patient', 'treatment', 'disease', 'condition',
        'chronic', 'acute', 'pathology', 'etiology', 'prognosis', 'differential',
        'inflammation', 'infection', 'neoplasm', 'hemorrhage', 'edema', 'ischemia',
        'hypertension', 'tachycardia', 'bradycardia', 'dyspnea', 'cyanosis',
        'hepatomegaly', 'splenomegaly', 'carcinoma', 'metastasis', 'biopsy',
        'antibiotic', 'analgesic', 'corticosteroid', 'immunosuppressant'
    }
    med_count = sum(1 for w in words if w in med_terms)
    sentences = re.split(r'[.!?]+', all_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    med_density = med_count / max(len(sentences), 1)

    # Average sentence length
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

    # Question ratio
    questions = sum(1 for s in sentences if '?' in s or s.strip().startswith(('what', 'how', 'why', 'when', 'where', 'which', 'is', 'are', 'do', 'does', 'can', 'could')))
    question_ratio = questions / max(len(sentences), 1)

    return {
        "ttr": round(ttr, 4),
        "med_term_density": round(med_density, 4),
        "avg_sentence_len": round(avg_sent_len, 2),
        "question_ratio": round(question_ratio, 4),
        "n_words": len(words),
    }


def compute_behavioral_fidelity(dialogue: dict) -> dict:
    """Compute behavioral metrics."""
    messages = dialogue["messages"]
    student_msgs = [m for m in messages if m["role"] == "student"]
    n_turns = dialogue["turn_count"]
    diagnosis = dialogue["diagnosis_reached"]

    # Help-seeking: student messages containing questions
    help_seeking = sum(1 for m in student_msgs if '?' in m["content"])

    # Self-correction: student referencing own previous error
    self_corrections = 0
    for i, m in enumerate(student_msgs):
        text = m["content"].lower()
        if any(phrase in text for phrase in ['actually', 'wait', 'i was wrong', 'let me reconsider', 'on second thought', 'correction']):
            self_corrections += 1

    return {
        "n_turns": n_turns,
        "diagnosis_reached": diagnosis,
        "help_seeking_freq": help_seeking / max(len(student_msgs), 1),
        "self_correction_rate": self_corrections / max(len(student_msgs), 1),
        "n_student_msgs": len(student_msgs),
    }


def compute_cognitive_fidelity(dialogue: dict) -> dict:
    """Compute cognitive metrics from assessment data."""
    assessments = dialogue.get("assessments", [])
    if not assessments:
        return {"blooms_distribution": {}, "coi_distribution": {}, "avg_reasoning_quality": 0,
                "n_boundary_violations": 0, "n_misconceptions": 0}

    blooms = Counter()
    coi = Counter()
    reasoning_scores = []
    n_violations = 0
    n_misconceptions = 0

    for a in assessments:
        bl = a.get("blooms_level", "unknown")
        if bl != "unknown":
            blooms[bl] += 1
        cp = a.get("coi_phase", "unknown")
        if cp != "unknown":
            coi[cp] += 1
        rq = a.get("reasoning_quality", 0)
        try:
            rq = int(rq) if isinstance(rq, str) and rq.isdigit() else (float(rq) if isinstance(rq, (int, float)) else 0)
        except (ValueError, TypeError):
            rq = 0
        if rq > 0:
            reasoning_scores.append(rq)
        bv = a.get("boundary_violations", [])
        n_violations += len(bv) if isinstance(bv, list) else 0
        mc = a.get("misconceptions", [])
        n_misconceptions += len(mc) if isinstance(mc, list) else 0

    return {
        "blooms_distribution": dict(blooms),
        "coi_distribution": dict(coi),
        "avg_reasoning_quality": round(np.mean(reasoning_scores), 2) if reasoning_scores else 0,
        "n_boundary_violations": n_violations,
        "n_misconceptions": n_misconceptions,
        "n_valid_assessments": sum(1 for a in assessments if a.get("blooms_level", "unknown") != "unknown"),
    }


def compute_interaction_quality(dialogue: dict) -> dict:
    """Compute educational interaction quality metrics."""
    tutor_msgs = [m["content"] for m in dialogue["messages"] if m["role"] == "tutor"]
    if not tutor_msgs:
        return {"socratic_ratio": 0, "n_tutor_msgs": 0}

    # Socratic ratio: proportion of tutor messages containing questions
    socratic = sum(1 for m in tutor_msgs if '?' in m)
    return {
        "socratic_ratio": round(socratic / len(tutor_msgs), 4),
        "n_tutor_msgs": len(tutor_msgs),
    }


def run_anova(dialogues: list, metric_func, metric_key: str):
    """Run 2-way ANOVA: condition × level on a specific metric."""
    groups = defaultdict(list)
    for d in dialogues:
        cond = d["condition"]
        level = d["student_level"]
        metrics = metric_func(d)
        val = metrics.get(metric_key, 0)
        if isinstance(val, (int, float)):
            groups[(cond, level)].append(val)

    # One-way ANOVA across conditions
    conditions = set(d["condition"] for d in dialogues)
    cond_groups = defaultdict(list)
    for d in dialogues:
        metrics = metric_func(d)
        val = metrics.get(metric_key, 0)
        if isinstance(val, (int, float)):
            cond_groups[d["condition"]].append(val)

    if len(cond_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*[v for v in cond_groups.values() if len(v) > 1])
        return {"f_statistic": round(f_stat, 4), "p_value": round(p_val, 6), "metric": metric_key}
    return {"f_statistic": 0, "p_value": 1.0, "metric": metric_key}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    dialogues = load_all_dialogues(args.results_dir)
    if not dialogues:
        logger.error("No dialogues found!")
        return

    # Compute all metrics for each dialogue
    all_metrics = []
    for d in dialogues:
        m = {
            "case_id": d["case_id"],
            "condition": d["condition"],
            "student_level": d["student_level"],
            "seed": d["seed"],
            **compute_linguistic_fidelity(d),
            **compute_behavioral_fidelity(d),
            **compute_cognitive_fidelity(d),
            **compute_interaction_quality(d),
        }
        all_metrics.append(m)

    # Save per-dialogue metrics
    metrics_file = os.path.join(output_dir, "all_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved per-dialogue metrics to {metrics_file}")

    # Aggregate by condition × level
    print("\n" + "=" * 80)
    print("FIDELITY ANALYSIS REPORT")
    print("=" * 80)

    agg = defaultdict(lambda: defaultdict(list))
    for m in all_metrics:
        key = (m["condition"], m["student_level"])
        for k, v in m.items():
            if isinstance(v, (int, float)):
                agg[key][k].append(v)

    print(f"\n{'Condition':<15} {'Level':<18} {'TTR':>6} {'MedDens':>8} {'Turns':>6} {'DiagRate':>9} {'Socratic':>9} {'BndryViol':>10}")
    print("-" * 80)
    for (cond, level), metrics in sorted(agg.items()):
        ttr = np.mean(metrics.get("ttr", [0]))
        med = np.mean(metrics.get("med_term_density", [0]))
        turns = np.mean(metrics.get("n_turns", [0]))
        diag = np.mean(metrics.get("diagnosis_reached", [0]))
        socr = np.mean(metrics.get("socratic_ratio", [0]))
        viol = np.mean(metrics.get("n_boundary_violations", [0]))
        print(f"{cond:<15} {level:<18} {ttr:>6.3f} {med:>8.3f} {turns:>6.1f} {diag:>9.1%} {socr:>9.3f} {viol:>10.1f}")

    # ANOVA tests
    print(f"\n{'='*80}")
    print("STATISTICAL TESTS (One-way ANOVA across conditions)")
    print(f"{'='*80}")
    for metric_key in ["ttr", "med_term_density", "n_turns", "socratic_ratio", "n_boundary_violations", "avg_reasoning_quality"]:
        result = run_anova(dialogues, compute_linguistic_fidelity if "ttr" in metric_key or "med" in metric_key
                          else compute_behavioral_fidelity if "turns" in metric_key
                          else compute_interaction_quality if "socratic" in metric_key
                          else compute_cognitive_fidelity, metric_key)
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
        print(f"  {metric_key:<30} F={result['f_statistic']:>8.3f}  p={result['p_value']:>.6f}  {sig}")

    # Save summary
    summary = {
        "n_dialogues": len(dialogues),
        "conditions": list(set(d["condition"] for d in dialogues)),
        "levels": list(set(d["student_level"] for d in dialogues)),
        "aggregate": {str(k): {kk: round(np.mean(vv), 4) for kk, vv in v.items()} for k, v in agg.items()},
    }
    summary_file = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_file}")

    print(f"\n{'='*80}")
    print(f"Analysis complete. {len(dialogues)} dialogues processed.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
