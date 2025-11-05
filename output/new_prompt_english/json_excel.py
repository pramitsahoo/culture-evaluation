import json
import pandas as pd
import os
import glob
import sys

def extract_adapted_texts(file_path):
    """Extract all culturally adapted texts from a JSON file with the specific structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        adapted_texts = []
        
        # Handle the specific structure: list of JSON *strings*
        if isinstance(data, list):
            for item in data:
                # Each 'item' should be a JSON string, e.g. "[{\"cultural_adapted_text\": ... }]"
                if isinstance(item, str):
                    try:
                        # Parse the JSON string again
                        inner_data = json.loads(item)
                        
                        # Case 1: It's a list of dictionaries
                        if isinstance(inner_data, list):
                            for inner_item in inner_data:
                                if isinstance(inner_item, dict) and "cultural_adapted_text" in inner_item:
                                    adapted_texts.append(inner_item["cultural_adapted_text"])
                        # Case 2: It's a single dictionary
                        elif isinstance(inner_data, dict) and "cultural_adapted_text" in inner_data:
                            adapted_texts.append(inner_data["cultural_adapted_text"])
                    
                    except json.JSONDecodeError as e:
                        # Print the exact failing string and the error details
                        print(f"Error parsing JSON string in file {file_path}:\n  --> {item}")
                        print(f"  Error details: {e}\n")
                        continue
        
        return adapted_texts
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

def extract_original_questions_from_jsonl(file_path):
    """Extract original questions from a JSONL file (one JSON object per line)."""
    questions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "question" in data:
                        questions.append(data["question"])
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line in {file_path}:\n  --> {line.strip()}")
                    print(f"  Error details: {e}\n")
                    continue
        return questions
    except Exception as e:
        print(f"Error extracting original questions from {file_path}: {e}")
        return []

def main():
    # Get directory path from command line or ask user
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = input("Enter the directory path containing your JSON files: ")
        if not directory:
            directory = os.getcwd()
    
    # Make sure the directory path exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    # Specific path for original questions (update this if needed)
    original_questions_path = "/u/student/2023/ai23mtech14004/culture-evaluation/dataset/gsm_8k/test.jsonl"
    
    # Grab all JSON files in the specified directory
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    # Optionally skip certain files if needed
    json_files = [
        f for f in json_files
        if not os.path.basename(f).startswith('deepseek') and not f.endswith('.py')
    ]
    
    # Map filenames to "readable" names for the DataFrame columns
    model_name_mapping = {
        'gemma-2-2b': 'Gemma-2-2B',
        'gemma-2': 'Gemma-2',
        'llama-2': 'Llama-2',
        'llama-3': 'Llama-3',
        'llama-3.2-1b': 'Llama-3.2-1B',
        'llama-3.2-3b': 'Llama-3.2-3B',
        'mistral': 'Mistral'
    }
    
    # Extract the original questions from JSONL
    original_questions = extract_original_questions_from_jsonl(original_questions_path)
    if not original_questions:
        print(f"Failed to extract original questions from {original_questions_path}")
        return
    
    # Prepare DataFrame with columns for each model
    columns = ["Sl. No.", "Original"]
    model_names = sorted(model_name_mapping.values())
    for model in model_names:
        columns.append(f"{model} Generated")
    
    df = pd.DataFrame(columns=columns)
    
    # Gather each model's adapted texts in a dict: { modelName -> list_of_texts }
    model_adapted_texts = {}
    
    for file in json_files:
        filename = os.path.basename(file)
        
        # Identify which model name is in the filename
        model_key = None
        for key in model_name_mapping.keys():
            if key in filename:
                model_key = key
                break
        
        if model_key:
            model_name = model_name_mapping[model_key]
            texts = extract_adapted_texts(file)
            if texts:
                model_adapted_texts[model_name] = texts
    
    # Debug info: see how many files processed, which models found, how many lines each
    print("Files processed:", len(json_files))
    print("Models found:", list(model_adapted_texts.keys()))
    for model, texts in model_adapted_texts.items():
        print(f"{model}: {len(texts)} adapted texts")
    
    # Build the DataFrame row by row
    for i, question in enumerate(original_questions):
        row = {
            "Sl. No.": i + 1,
            "Original": question
        }
        
        # Fill in each model's generated text for this question index
        for model_name in model_names:
            col_name = f"{model_name} Generated"
            texts = model_adapted_texts.get(model_name, [])
            if texts and i < len(texts):
                row[col_name] = texts[i]
        
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    # Save to Excel
    output_path = os.path.join(directory, "adapted_texts_comparison.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Excel file created successfully at {output_path}")

if __name__ == "__main__":
    main()
