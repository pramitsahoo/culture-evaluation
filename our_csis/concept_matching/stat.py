import os
import json
import csv

def load_json(filepath):
    """Load JSON content from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

def main():
    # Set the directory where your concept matching scores JSON files are saved.
    scores_dir = "culture-evaluation/our_csis/concept_matching"
    
    # Get a list of all files starting with "concept_matching_scores" and ending with ".json"
    json_files = [f for f in os.listdir(scores_dir)
                  if f.startswith("concept_matching_scores") and f.endswith(".json")]
    
    # Lists to hold aggregated statistics
    dataset_stats_list = []  # One row per model for dataset-level statistics
    facet_stats_list = []    # One row per model and per facet
    state_stats_list = []    # One row per model and per state

    for filename in json_files:
        filepath = os.path.join(scores_dir, filename)
        data = load_json(filepath)
        if not data:
            continue

        # Extract model name from the filename.
        # For example: "concept_matching_scores_mistral_gsm_8k_test.json" -> "mistral_gsm_8k_test"
        model = filename.replace("concept_matching_scores_", "").replace(".json", "")
        
        # Get dataset-level statistics
        ds_stats = data.get("dataset_statistics", {})
        ds_stats["model"] = model
        dataset_stats_list.append(ds_stats)
        
        # Get facet statistics (each facet becomes one row)
        facet_stats = data.get("facet_statistics", {})
        for facet, stats in facet_stats.items():
            facet_stats_list.append({
                "model": model,
                "facet": facet,
                "count": stats.get("count", 0),
                "percentage": stats.get("percentage", 0)
            })
        
        # Get state statistics (each state becomes one row)
        state_stats = data.get("state_statistics", {})
        for state, stats in state_stats.items():
            state_stats_list.append({
                "model": model,
                "state": state,
                "count": stats.get("count", 0),
                "percentage": stats.get("percentage", 0)
            })

    # Write dataset statistics CSV
    dataset_csv = os.path.join(scores_dir, "dataset_statistics_comparison.csv")
    with open(dataset_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "model",
            "total_sentences",
            "entries_with_replacements",
            "total_strict_matches",
            "total_singular_plural_matches",
            "average_strict_match_per_sentence",
            "average_singular_plural_match_per_sentence"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset_stats_list:
            writer.writerow(row)

    # Write facet statistics CSV
    facet_csv = os.path.join(scores_dir, "facet_statistics_comparison.csv")
    with open(facet_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["model", "facet", "count", "percentage"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in facet_stats_list:
            writer.writerow(row)

    # Write state statistics CSV
    state_csv = os.path.join(scores_dir, "state_statistics_comparison.csv")
    with open(state_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["model", "state", "count", "percentage"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in state_stats_list:
            writer.writerow(row)

    print("CSV files created:")
    print(f" - {dataset_csv}")
    print(f" - {facet_csv}")
    print(f" - {state_csv}")

if __name__ == "__main__":
    main()
