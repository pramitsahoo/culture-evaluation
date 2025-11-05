import json
import sys

def clean_and_parse(item_str):
    """
    Remove markdown code block markers (if any) and try to parse the JSON string.
    """
    if item_str.startswith("```json"):
        item_str = item_str[len("```json"):].strip()
    if item_str.endswith("```"):
        item_str = item_str[:-3].strip()
    return json.loads(item_str)

def extract_replaced_concepts_from_text(text):
    """
    Extract replaced concepts from text that contains a "Replaced concepts:" section.
    First, attempt to extract an embedded JSON snippet. If not found, look for bullet point lines.
    """
    # Attempt to extract an embedded JSON snippet.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict) and "replaced_concepts" in obj:
                return obj.get("replaced_concepts")
        except Exception:
            pass

    # Look for "Replaced concepts:" section.
    if "Replaced concepts:" not in text:
        return None

    parts = text.split("Replaced concepts:", 1)[1]
    lines = parts.splitlines()
    mapping = {}

    for line in lines:
        line = line.strip()
        if not line.startswith("*"):
            continue
        # Remove the leading "*" and extra spaces.
        line_content = line[1:].strip()
        # Support two formats: using "->" or "is replaced with"
        if "->" in line_content:
            parts_line = line_content.split("->", 1)
        elif "is replaced with" in line_content:
            parts_line = line_content.split("is replaced with", 1)
        else:
            continue

        # Clean up the strings by stripping quotes and extra spaces.
        orig = parts_line[0].strip().strip('"')
        repl = parts_line[1].strip()
        if "(" in repl:
            repl = repl.split("(", 1)[0].strip()
        repl = repl.strip('"')
        mapping[orig] = repl

    return mapping if mapping else None

def extract_replaced_concepts(file_path, output_path):
    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []       # Store extracted replaced_concepts with entry IDs.
    output_lines = []  # To accumulate text output.

    # Process each entry in the outer JSON list.
    for i, item in enumerate(data, start=1):
        # If the item is a string, try to parse it.
        if isinstance(item, str):
            try:
                parsed_item = clean_and_parse(item)
                item = parsed_item
            except Exception as e:
                # If JSON parsing fails, try to extract from text.
                concepts = extract_replaced_concepts_from_text(item)
                entry_id = f"{i}"
                if concepts:
                    entry_output = f"Entry {entry_id} replaced concepts:\n" + json.dumps(concepts, indent=2, ensure_ascii=False)
                    print(entry_output)
                    output_lines.append(entry_output)
                    output_lines.append("-" * 40)
                    results.append({"entry": entry_id, "replaced_concepts": concepts})
                else:
                    msg = f"Entry {i} has no 'replaced_concepts' key."
                    print(msg)
                    output_lines.append(msg)
                continue

        # If the item is a list, process each inner entry.
        if isinstance(item, list):
            for j, inner_item in enumerate(item, start=1):
                if isinstance(inner_item, str):
                    try:
                        parsed_inner = clean_and_parse(inner_item)
                        inner_item = parsed_inner
                    except Exception:
                        concepts = extract_replaced_concepts_from_text(inner_item)
                        entry_id = f"{i}-{j}"
                        if concepts:
                            entry_output = f"Entry {entry_id} replaced concepts:\n" + json.dumps(concepts, indent=2, ensure_ascii=False)
                            print(entry_output)
                            output_lines.append(entry_output)
                            output_lines.append("-" * 40)
                            results.append({"entry": entry_id, "replaced_concepts": concepts})
                        else:
                            msg = f"Entry {i}-{j} has no 'replaced_concepts' key."
                            print(msg)
                            output_lines.append(msg)
                        continue

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
        print("Usage: python extract_concepts.py path_to_file.json [output_file.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "extracted_concepts_llama2.json"
    extract_replaced_concepts(input_file, output_file)
