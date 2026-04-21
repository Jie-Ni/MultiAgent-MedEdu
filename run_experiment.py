"""Main experiment runner: generates all dialogues across conditions and student levels."""
import argparse
import json
import os
import sys
import time
import yaml
from loguru import logger

sys.path.insert(0, os.path.dirname(__file__))
from agents.base import LLMBackend, DialogueState
from agents.multi_agent import MultiAgentOrchestrator, SingleAgentSimulator, DirectQASimulator


def load_cases(data_dir: str, n_cases: int = 30) -> list:
    """Load clinical cases from MedQA data."""
    medqa_path = os.path.join(data_dir, "medqa_pilot.json")
    if not os.path.exists(medqa_path):
        logger.error(f"MedQA data not found at {medqa_path}")
        sys.exit(1)

    with open(medqa_path, "r") as f:
        all_cases = json.load(f)

    # Select n_cases, trying to balance by meta_info (step1/step2)
    cases = all_cases[:n_cases]
    logger.info(f"Loaded {len(cases)} clinical cases")
    return cases


def run_single_dialogue(
    llm: LLMBackend,
    case: dict,
    case_idx: int,
    condition: str,
    student_level: str,
    student_config: dict,
    seed: int,
    max_turns: int,
) -> dict:
    """Run one dialogue and return results."""
    state = DialogueState(
        case_id=f"case_{case_idx}",
        condition=condition,
        student_level=student_level,
        seed=seed,
        max_turns=max_turns,
    )

    if condition == "multi_agent":
        orchestrator = MultiAgentOrchestrator(llm, case, student_config)
    elif condition == "single_agent":
        orchestrator = SingleAgentSimulator(llm, case, student_config)
    elif condition == "direct_qa":
        orchestrator = DirectQASimulator(llm, case, student_config)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    try:
        state = orchestrator.run_dialogue(state)
    except Exception as e:
        logger.error(f"Dialogue failed: {condition}/{student_level}/case_{case_idx}/seed_{seed}: {e}")
        state.add_message("system", f"ERROR: {str(e)}")

    return state.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent medical education experiment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", required=True, help="Directory with medqa_pilot.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", default=None, help="Run single condition (multi_agent|single_agent|direct_qa)")
    parser.add_argument("--level", default=None, help="Run single student level")
    parser.add_argument("--n-cases", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    n_cases = args.n_cases or config["experiment"]["n_cases"]
    n_seeds = args.n_seeds or config["experiment"]["n_seeds"]
    max_turns = config["llm"]["max_dialogue_turns"]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load cases
    cases = load_cases(args.data_dir, n_cases)

    # Initialize LLM
    model_name = config["llm"]["model"]
    llm = LLMBackend(model_name, config_path)
    llm.initialize()

    # Determine conditions and levels to run
    conditions = [args.condition] if args.condition else list(config["conditions"].keys())
    levels = [args.level] if args.level else list(config["student_levels"].keys())

    total = len(conditions) * len(levels) * len(cases) * n_seeds
    logger.info(
        f"Starting experiment: {len(conditions)} conditions × {len(levels)} levels "
        f"× {len(cases)} cases × {n_seeds} seeds = {total} dialogues"
    )

    all_results = []
    completed = 0
    t0 = time.time()

    for condition in conditions:
        for level_name in levels:
            level_config = config["student_levels"][level_name]
            for case_idx, case in enumerate(cases):
                for seed in range(n_seeds):
                    result = run_single_dialogue(
                        llm, case, case_idx, condition, level_name, level_config, seed, max_turns
                    )
                    all_results.append(result)
                    completed += 1

                    if completed % 10 == 0:
                        elapsed = time.time() - t0
                        rate = completed / elapsed
                        eta = (total - completed) / rate if rate > 0 else 0
                        logger.info(
                            f"Progress: {completed}/{total} ({completed/total*100:.1f}%) "
                            f"- {rate:.2f} dialogues/s - ETA: {eta/60:.1f}min"
                        )

    # Save results
    timestamp = int(time.time())
    out_file = os.path.join(args.output_dir, f"dialogues_{timestamp}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Experiment complete! {len(all_results)} dialogues saved to {out_file}")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for condition in conditions:
        cond_results = [r for r in all_results if r["condition"] == condition]
        diag_rate = sum(1 for r in cond_results if r["diagnosis_reached"]) / max(len(cond_results), 1)
        avg_turns = sum(r["turn_count"] for r in cond_results) / max(len(cond_results), 1)
        print(f"  {condition:15s}: {len(cond_results)} dialogues, "
              f"diagnosis rate: {diag_rate:.1%}, avg turns: {avg_turns:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
