import os
import json
from glob import glob
import sys

def extract_replaced_concepts(file_path):
    """
    Extracts replaced concepts from cultural adaptation JSON files.
    Each file contains a list of strings, where each string is a JSON array
    containing cultural adaptations.
    """
    print(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded data from {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    all_replacements = []
    
    if isinstance(data, list):
        for i, item in enumerate(data):
            try:
                if isinstance(item, str):
                    # Parse the string as JSON
                    item_data = json.loads(item)
                    
                    # Extract replaced_concepts if it exists
                    if isinstance(item_data, list):
                        for entry in item_data:
                            if isinstance(entry, dict) and 'replaced_concepts' in entry:
                                replacement = entry['replaced_concepts']
                                if replacement:  # Only add if not empty
                                    all_replacements.append(replacement)
                                    print(f"Found replacement: {replacement}")
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue

    return all_replacements

def main():
    # Directory containing the JSON files
    input_dir = "/u/student/2023/ai23mtech14004/culture-evaluation/output/new_prompt_english"  # Current directory
    
    # Create output directory
    output_dir = os.path.join("/u/student/2023/ai23mtech14004/culture-evaluation/output/new_prompt_english/replaced_words")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all JSON files
    json_files = glob(os.path.join(input_dir, "*_gsm_8k_test.json"))
    
    if not json_files:
        print("No JSON files found!")
        return
    
    print(f"Found {len(json_files)} files to process")
    
    for file_path in json_files:
        # Extract model name from filename
        model_name = os.path.basename(file_path).split('_gsm_')[0]
        print(f"\nProcessing model: {model_name}")
        
        # Extract concepts
        replaced_concepts = extract_replaced_concepts(file_path)
        
        if replaced_concepts:
            # Create output file path
            output_file = os.path.join(output_dir, f"{model_name}_replaced_concepts.json")
            
            try:
                # Save to file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(replaced_concepts, f, indent=4, ensure_ascii=False)
                print(f"Saved {len(replaced_concepts)} replacements to {output_file}")
            except Exception as e:
                print(f"Error saving file {output_file}: {e}")
        else:
            print(f"No replacements found for {model_name}")

if __name__ == '__main__':
    main()