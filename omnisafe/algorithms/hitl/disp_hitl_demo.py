import os
import csv
from typing import Optional


def _is_truthy(val: Optional[str]) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def print_hitl_demo_stats():
    """
    Print statistics for the HITL demo episodes from CSV logs, including:
    - Total steps per episode
    - Human intervened steps per episode (via act_overlaid if present, otherwise act != act_agent)
    - Overall statistics across all episodes
    """
    # Resolve base path relative to this file so it works when run from repo root
    base_path = os.path.join(os.path.dirname(__file__), "evaluations", "hitl_demo")

    total_steps = 0
    total_interventions = 0

    print("\nHITL Demo Statistics:")
    print("-" * 50)

    # Iterate through a reasonable range of episodes until files stop existing
    found_any = False
    for ep_num in range(1000):  # upper bound to allow many episodes while staying safe
        filename = f"medium_hitlTrue_lossNone_episode{ep_num:03d}.csv"
        file_path = os.path.join(base_path, filename)
        if not os.path.exists(file_path):
            # For initial gaps, skip; break after the first missing following a found block
            if found_any:
                break
            else:
                continue

        found_any = True

        ep_total_steps = 0
        intervention_count = 0

        try:
            with open(file_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                # Normalize fieldnames to known keys
                has_act_overlaid = "act_overlaid" in (reader.fieldnames or [])
                has_act = "act" in (reader.fieldnames or [])
                has_act_agent = "act_agent" in (reader.fieldnames or [])

                for row in reader:
                    ep_total_steps += 1

                    intervened = False
                    if has_act_overlaid:
                        intervened = _is_truthy(row.get("act_overlaid"))
                    if not intervened and has_act and has_act_agent:
                        # Compare as raw strings to avoid heavy parsing; robust to whitespace quotes
                        intervened = (str(row.get("act")).strip() != str(row.get("act_agent")).strip())

                    if intervened:
                        intervention_count += 1
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            continue

        total_steps += ep_total_steps
        total_interventions += intervention_count

        # Print episode statistics
        print(f"\nEpisode {ep_num:03d} ({filename}):")
        print(f"  Total Steps: {ep_total_steps}")
        print(f"  Human Interventions: {intervention_count}")
        rate = (intervention_count / ep_total_steps * 100.0) if ep_total_steps else 0.0
        print(f"  Intervention Rate: {rate:.2f}%")

    # Print overall statistics
    print("\nOverall Statistics:")
    print("-" * 50)
    print(f"Total Steps Across All Episodes: {total_steps}")
    print(f"Total Human Interventions: {total_interventions}")
    overall_rate = (total_interventions / total_steps * 100.0) if total_steps else 0.0
    print(f"Overall Intervention Rate: {overall_rate:.2f}%")


if __name__ == "__main__":
    print_hitl_demo_stats()
