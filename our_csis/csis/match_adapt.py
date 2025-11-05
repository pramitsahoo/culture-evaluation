import os
import glob
import json
import csv

def load_json(filename):
    """Load JSON data from a file, trying different encodings if necessary."""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1']
    for enc in encodings:
        try:
            with open(filename, 'r', encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError as e:
            print(f"Error reading {filename} with encoding {enc}: {e}")
    raise Exception(f"Could not read {filename} using any of the encodings: {encodings}")

def get_replacement_values(replaced_data):
    """
    Given a list of dictionaries mapping original words to replacement words,
    return a set of normalized (lower-case and stripped) replacement words.
    """
    replacement_values = set()
    for mapping in replaced_data:
        for original, replacement in mapping.items():
            normalized = str(replacement).strip().lower()
            replacement_values.add(normalized)
    return replacement_values

def compute_statistics(concepts_data, replacement_values):
    """
    For each state in the cultural concepts data, compute:
      - total number of concepts,
      - number of adapted concepts (whose normalized concept name is in replacement_values),
      - record the original names that matched.
    
    Assumes each state is a dict with keys:
      - "name": the state name,
      - "items": a list of dictionaries. The first dictionary's first key is used
                 as the key for the concept name (e.g., "Dance Form Name").
    
    Returns:
      stats: dict keyed by state name with keys "total", "adapted", "not_adapted", "matches"
      total_adapted: overall adapted count (across all states for this model)
      total_concepts: overall total count (across all states)
    """
    stats = {}
    total_adapted = 0
    total_concepts = 0

    for state in concepts_data:
        state_name = state.get("name", "Unknown")
        items = state.get("items", [])
        count_total = len(items)
        count_adapted = 0
        matched_concepts = []
        
        # Determine which key holds the concept name by looking at the first item (if available)
        concept_key = None
        if items:
            first_item = items[0]
            # Use the first key of the first dictionary as the concept name key.
            concept_key = list(first_item.keys())[0]
        
        for concept in items:
            if concept_key:
                concept_name = concept.get(concept_key, "").strip()
                normalized_name = concept_name.lower()
                # Compare normalized concept name with replacement values.
                if normalized_name in replacement_values:
                    count_adapted += 1
                    matched_concepts.append(concept_name)
        
        stats[state_name] = {
            "total": count_total,
            "adapted": count_adapted,
            "not_adapted": count_total - count_adapted,
            "matches": matched_concepts
        }
        total_concepts += count_total
        total_adapted += count_adapted

    return stats, total_adapted, total_concepts

def write_stats_to_file(stats, total_adapted, total_concepts, output_filename):
    """Write the computed per‑state statistics and overall stats to a text file."""
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("Adaptation Statistics Per State:\n")
        for state, data in stats.items():
            f.write(f"State: {state}\n")
            f.write(f"  Total Concepts   : {data['total']}\n")
            f.write(f"  Adapted Concepts : {data['adapted']}\n")
            f.write(f"  Not Adapted      : {data['not_adapted']}\n")
            if data['matches']:
                f.write("  Matched Concepts : " + ", ".join(data['matches']) + "\n")
            else:
                f.write("  Matched Concepts : None\n")
            f.write("\n")
        
        overall_ratio = total_adapted / total_concepts if total_concepts > 0 else 0
        f.write("Overall Statistics:\n")
        f.write(f"Total Concepts         : {total_concepts}\n")
        f.write(f"Total Adapted Concepts : {total_adapted}\n")
        f.write(f"Total Not Adapted      : {total_concepts - total_adapted}\n")
        f.write(f"Overall Adaptation Score: {overall_ratio:.2f}\n")
    print(f"Statistics written to '{output_filename}'.")

def main():
    # --- SETTINGS ---
    # Path to the cultural JSON file (for example, one of your files like Dance_Forms.json, Arts.json, etc.)
    # For this example, we will process every JSON file in the directory except the one to be excluded.
    cultural_json_files = glob.glob("*.json")
    exclude_file = "india_cultural_features.json"  # adjust if needed
    cultural_json_files = [f for f in cultural_json_files if os.path.basename(f) != exclude_file]
    
    # Directory where all replaced‑concepts JSON files (for different models) reside.
    replaced_dir = "/u/student/2023/ai23mtech14004/culture-evaluation/output/new_prompt_english/replaced_words"
    replaced_pattern = os.path.join(replaced_dir, "*_replaced_concepts.json")
    
    # Mapping dictionary to rename models based on their file names.
    model_rename = {
        "gemma-2-2b":   "Gemma 2 2B Instruct",
        "llama-3.2-1b": "Llama 3.2 1B Instruct",
        "mistral":      "Mistral 7BInstruct v0.3",
        "gemma-2":      "Gemma 2 9B Instruct",
        "llama-3.2-3b": "Llama 3.2 3B Instruct",
        "llama-2":      "Llama 2 7B Chat",
        "llama-3":      "Llama 3.1 8B Instruct"
    }
    
    # Process each cultural JSON file in the current directory (except the excluded one)
    for cultural_file in cultural_json_files:
        print(f"Processing {cultural_file}...")
        try:
            concepts_data = load_json(cultural_file)
        except Exception as e:
            print(f"Error loading {cultural_file}: {e}")
            continue
        
        # Get the list of states from the cultural JSON file.
        state_names = [state.get("name", "Unknown") for state in concepts_data]
        
        # Dictionary to store adaptation scores per model for this cultural aspect.
        adaptation_scores = {}
        
        # Process each replaced‑concepts file.
        replaced_files = glob.glob(replaced_pattern)
        if not replaced_files:
            print("No replaced concepts JSON files found in", replaced_dir)
            return
        
        for replaced_file in replaced_files:
            base = os.path.basename(replaced_file)
            model_name = base.replace("_replaced_concepts.json", "")
            display_name = model_rename.get(model_name, model_name)
            
            try:
                replaced_data = load_json(replaced_file)
            except Exception as e:
                print(f"Error loading {replaced_file}: {e}")
                continue
            
            replacement_values = get_replacement_values(replaced_data)
            stats, total_adapted, total_concepts = compute_statistics(concepts_data, replacement_values)
            
            # Write per-model stats to a text file.
            stats_output_filename = f"{display_name}_{os.path.splitext(cultural_file)[0]}_stats.txt"
            write_stats_to_file(stats, total_adapted, total_concepts, stats_output_filename)
            
            # Compute adaptation ratios per state for this model.
            model_scores = {}
            for state in state_names:
                state_stat = stats.get(state, {"adapted": 0, "total": 0})
                adapted = state_stat.get("adapted", 0)
                ratio = adapted / total_adapted if total_adapted > 0 else 0
                model_scores[state] = ratio
            overall_ratio = total_adapted / total_concepts if total_concepts > 0 else 0
            model_scores["Overall"] = overall_ratio
            
            adaptation_scores[display_name] = model_scores
        
        # --- WRITE THE CSV FILE FOR THIS CULTURAL ASPECT ---
        # Build header: first column "States", then one column per model.
        model_names = sorted(adaptation_scores.keys())
        header = ["States"] + model_names

        csv_rows = []
        for state in state_names:
            row = [state]
            for model in model_names:
                score = adaptation_scores[model].get(state, 0)
                row.append(f"{score:.2f}")
            csv_rows.append(row)
        
        # Add an extra row for overall adaptation scores.
        overall_row = ["Overall"]
        for model in model_names:
            overall_score = adaptation_scores[model].get("Overall", 0)
            overall_row.append(f"{overall_score:.2f}")
        csv_rows.append(overall_row)
        
        output_csv = f"{os.path.splitext(cultural_file)[0]}_adaptation_scores.csv"
        try:
            with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(csv_rows)
            print(f"CSV file written to '{output_csv}'.")
        except Exception as e:
            print(f"Error writing CSV file {output_csv}: {e}")

if __name__ == "__main__":
    main()
