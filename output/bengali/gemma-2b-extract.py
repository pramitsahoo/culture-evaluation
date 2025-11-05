import json
import sys
import re

def clean_and_parse(item_str):
    """
    Remove markdown code block markers (if any) and parse the JSON string.
    This version removes any leading/trailing markdown fences regardless
    of whether the closing marker is present.
    """
    # Remove ```json (or ``` if present) from the start of the string.
    item_str = re.sub(r"^```(?:json)?\s*", "", item_str, flags=re.MULTILINE)
    # Remove any trailing ``` (or extra whitespace after it) from the end.
    item_str = re.sub(r"\s*```$", "", item_str, flags=re.MULTILINE)
    # It's also possible there are stray markdown fences in the middle;
    # if needed, you can remove them too.
    item_str = re.sub(r"```", "", item_str)
    return json.loads(item_str)

def extract_replaced_concepts(file_path, output_path):
    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []       # To store the extracted replaced_concepts along with entry IDs.
    output_lines = []  # To accumulate a text version of the output.
    
    # Process each entry in the outer JSON list.
    for i, item in enumerate(data, start=1):
        # If the item is a string, attempt to parse it.
        if isinstance(item, str):
            try:
                item = clean_and_parse(item)
            except Exception as e:
                msg = f"Error parsing item {i}: {e}"
                print(msg)
                output_lines.append(msg)
                continue
        
        # Process the item if it is a list of dictionaries or a single dictionary.
        if isinstance(item, list):
            for j, inner_item in enumerate(item, start=1):
                concepts = inner_item.get("replaced_concepts")
                entry_id = f"{i}-{j}"
                if concepts:
                    entry_output = f"Entry {entry_id} replaced concepts:\n" + json.dumps(concepts, indent=2, ensure_ascii=False)
                    print(entry_output)
                    output_lines.append(entry_output)
                    output_lines.append("-" * 40)
                    results.append({"entry": entry_id, "replaced_concepts": concepts})
                else:
                    msg = f"Entry {entry_id} has no 'replaced_concepts' key."
                    print(msg)
                    output_lines.append(msg)
        elif isinstance(item, dict):
            concepts = item.get("replaced_concepts")
            entry_id = f"{i}"
            if concepts:
                entry_output = f"Entry {entry_id} replaced concepts:\n" + json.dumps(concepts, indent=2, ensure_ascii=False)
                print(entry_output)
                output_lines.append(entry_output)
                output_lines.append("-" * 40)
                results.append({"entry": entry_id, "replaced_concepts": concepts})
            else:
                msg = f"Entry {entry_id} has no 'replaced_concepts' key."
                print(msg)
                output_lines.append(msg)
        else:
            msg = f"Unexpected type at entry {i}: {type(item)}"
            print(msg)
            output_lines.append(msg)
    
    # Save the extracted results to a JSON file.
    try:
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(results, out_file, indent=2, ensure_ascii=False)
        print(f"\nExtracted replaced concepts saved to {output_path}")
    except Exception as e:
        print(f"Error saving output to file: {e}")
    
    # Also save a text version of the output.
    text_output_path = output_path.rsplit('.', 1)[0] + ".txt"
    try:
        with open(text_output_path, 'w', encoding='utf-8') as text_file:
            text_file.write("\n".join(output_lines))
        print(f"Text output saved to {text_output_path}")
    except Exception as e:
        print(f"Error saving text output to file: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python gemma_2b_exatraction.py path_to_file.json [output_file.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "extracted_concepts.json"
    extract_replaced_concepts(input_file, output_file)
