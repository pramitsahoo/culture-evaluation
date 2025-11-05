import spacy
import json
import csv
import os

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

def is_person(text):
    # Ensure the input is a string
    text = str(text)
    doc = nlp(text)
    return any(ent.label_ == "PERSON" for ent in doc.ents)

# List of JSON files (adjust paths if necessary)
json_files = [
    "gemma-2-2b_replaced_concepts.json",
    "llama-3.2-1b_replaced_concepts.json",
    "mistral_replaced_concepts.json",
    "gemma-2_replaced_concepts.json",
    "llama-3.2-3b_replaced_concepts.json",
    "llama-2_replaced_concepts.json",
    "llama-3_replaced_concepts.json"
]

for filename in json_files:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    extracted_pairs = []
    # Each JSON file contains an array of dictionaries.
    for entry in data:
        for key, value in entry.items():
            # Convert key and value to strings and then check if both are recognized as persons.
            if is_person(key) and is_person(value):
                extracted_pairs.append({"Original": key, "Adapted": value})
    
    # Create a CSV filename based on the JSON filename.
    csv_filename = os.path.splitext(filename)[0] + "_names_spacy.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Original", "Adapted"])
        writer.writeheader()
        writer.writerows(extracted_pairs)
    
    print(f"Saved extracted names to {csv_filename}")
