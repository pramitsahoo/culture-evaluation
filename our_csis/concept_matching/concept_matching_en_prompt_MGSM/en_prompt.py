import re
import os
import json
import logging
import inflect
from rapidfuzz import fuzz, process  # Added for fuzzy matching

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the inflect engine (for singular/plural conversions)
inflect_engine = inflect.engine()

def normalize(word, allow_bengali=True):
    """
    Normalize a word by removing unwanted characters and converting to lowercase.
    Allows Bengali characters (Unicode range \u0980-\u09FF), ASCII letters, digits, ₹, and whitespace.
    """
    if not word:
        return ""
    try:
        word = str(word)
        if allow_bengali:
            # Allow Bengali characters along with a-z, A-Z, digits, ₹, and whitespace.
            word = re.sub(r'[^\u0980-\u09FFa-zA-Z0-9₹\s]', '', word)
        else:
            word = re.sub(r'[^a-zA-Z0-9₹\s]', '', word)
        return word.lower().strip()
    except Exception as e:
        logging.error(f"Error processing normalization: {word}, Error: {e}")
        return word

def strict_exact_match(word, concept_set, allow_bengali=True):
    """
    Return the normalized word if it exactly matches one in concept_set.
    """
    normalized_word = normalize(word, allow_bengali=allow_bengali)
    if normalized_word in concept_set:
        return normalized_word
    return None

def singular_plural_match(word, concept_set, allow_bengali=True):
    """
    Return a matching concept using singular/plural forms.
    (Note: This works best for English.)
    """
    normalized_word = normalize(word, allow_bengali=allow_bengali)
    if not normalized_word:
        return None
    try:
        singular = inflect_engine.singular_noun(normalized_word)
        if singular and singular in concept_set:
            return singular
        plural = inflect_engine.plural_noun(normalized_word)
        if plural and plural in concept_set:
            return plural
    except Exception as e:
        logging.error(f"Error in singular/plural conversion for '{normalized_word}': {e}")
    return None

def fuzzy_match(word, concept_set, threshold=80, allow_bengali=True):
    """
    Return a matching concept using fuzzy matching.
    Uses RapidFuzz's token sort ratio to find the best match.
    If the best match's score meets or exceeds the threshold, return the matched concept.
    """
    if not word or not concept_set:
        return None
    try:
        best_match = process.extractOne(normalize(word, allow_bengali=allow_bengali), 
                                        concept_set, scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= threshold:
            return best_match[0]
        return None
    except Exception as e:
        logging.error(f"Error in fuzzy matching for '{word}': {e}")
        return None

def read_output_file(filepath):
    """
    Read and parse a JSON output file.
    Expects a list of JSON objects or a nested list structure.
    """
    if not os.path.exists(filepath):
        logging.error(f"Output file not found: {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        parsed_data = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    try:
                        parsed_item = json.loads(item)
                        if isinstance(parsed_item, list):
                            parsed_data.extend([entry for entry in parsed_item if isinstance(entry, dict)])
                        elif isinstance(parsed_item, dict):
                            parsed_data.append(parsed_item)
                    except json.JSONDecodeError:
                        continue
                elif isinstance(item, list):
                    parsed_data.extend([entry for entry in item if isinstance(entry, dict)])
                elif isinstance(item, dict):
                    parsed_data.append(item)
        return parsed_data
    except Exception as e:
        logging.error(f"Error reading output file: {filepath}, Error: {e}")
        return []

def merge_output_files(directory):
    """
    Merge all JSON output files in the given directory into a single list.
    """
    merged = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            file_data = read_output_file(filepath)
            merged.extend(file_data)
    return merged

def read_concept_list(filepath, allow_bengali=True):
    """
    Read and parse the concept file.
    Expected structure:
      {
         "textiles": [ { "state": "...", "name": "Pochampally Ikat", ... }, ... ],
         "architectures": [ { "name": "Some Architecture Name", ... }, ... ],
         ...
      }
    For each facet, from each entry’s "items" list (if present) or directly,
    the first key that contains "name" (case-insensitive) is used.
    Returns:
      facet_concepts: dict mapping facet to list of concept names (raw)
      concept_metadata: dict mapping normalized concept name -> metadata
    """
    if not os.path.exists(filepath):
        logging.error(f"Concept file not found: {filepath}")
        return {}, {}
    facet_concepts = {}
    concept_metadata = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Data may be a dict with multiple facets
        if isinstance(data, dict):
            for facet, entries in data.items():
                facet_concepts.setdefault(facet, [])
                if isinstance(entries, list):
                    for entry in entries:
                        # If the entry has an "items" list, use that; otherwise, use the entry itself.
                        items = entry.get("items", [entry])
                        for item in items:
                            concept_name = None
                            # Look for a key that includes "name" (case-insensitive)
                            for key, value in item.items():
                                if "name" in key.lower() and value:
                                    concept_name = value
                                    break
                            if concept_name:
                                norm_name = normalize(concept_name, allow_bengali=allow_bengali)
                                facet_concepts[facet].append(norm_name)
                                # Also store metadata (state, description, source) if available
                                concept_metadata[norm_name] = {
                                    "facet": facet,
                                    "state": item.get("state", ""),
                                    "original_name": concept_name,
                                    "description": item.get("description", ""),
                                    "source": item.get("source", "")
                                }
        elif isinstance(data, list):
            # In case the file is a list of entries
            for entry in data:
                facet = entry.get("facet") or entry.get("name", "unknown")
                facet_concepts.setdefault(facet, [])
                items = entry.get("items", [entry])
                for item in items:
                    concept_name = None
                    for key, value in item.items():
                        if "name" in key.lower() and value:
                            concept_name = value
                            break
                    if concept_name:
                        norm_name = normalize(concept_name, allow_bengali=allow_bengali)
                        facet_concepts[facet].append(norm_name)
                        concept_metadata[norm_name] = {
                            "facet": facet,
                            "state": item.get("state", ""),
                            "original_name": concept_name,
                            "description": item.get("description", ""),
                            "source": item.get("source", "")
                        }
        return facet_concepts, concept_metadata
    except Exception as e:
        logging.error(f"Error reading concept file: {filepath}, Error: {e}")
        return {}, {}

def save_scores_to_file(scores, total_sentences, filepath):
    """
    Save the adaptation scores and dataset-wide statistics to a JSON file.
    Bengali characters are preserved.
    """
    try:
        total_strict_matches = sum(len(score["strict_matches"]) for score in scores)
        total_singular_plural_matches = sum(len(score["singular_plural_matches"]) for score in scores)
        total_fuzzy_matches = sum(len(score["fuzzy_matches"]) for score in scores)
        avg_strict = total_strict_matches / total_sentences if total_sentences > 0 else 0
        avg_singular_plural = total_singular_plural_matches / total_sentences if total_sentences > 0 else 0
        avg_fuzzy = total_fuzzy_matches / total_sentences if total_sentences > 0 else 0
        final_output = {
            "dataset_statistics": {
                "total_sentences": total_sentences,
                "entries_with_replacements": len(scores),
                "total_strict_matches": total_strict_matches,
                "total_singular_plural_matches": total_singular_plural_matches,
                "total_fuzzy_matches": total_fuzzy_matches,
                "average_strict_match_per_sentence": avg_strict,
                "average_singular_plural_match_per_sentence": avg_singular_plural,
                "average_fuzzy_match_per_sentence": avg_fuzzy
            },
            "detailed_scores": scores
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        logging.info(f"Scores saved to {filepath}")
        return avg_strict, avg_singular_plural
    except Exception as e:
        logging.error(f"Error saving scores to file: {filepath}, Error: {e}")
        return 0, 0

def calculate_adaptation_score(output_data, concept_file_path, allow_bengali=True):
    """
    Calculate adaptation scores given a list of output entries and a concept file.
    The function uses strict, singular/plural, and fuzzy matching against the global concept set.
    """
    facet_concepts, concept_metadata = read_concept_list(concept_file_path, allow_bengali=allow_bengali)
    # Merge all concept names from all facets into a global set (normalized)
    all_concepts = set()
    for concept_list in facet_concepts.values():
        for concept in concept_list:
            all_concepts.add(normalize(concept, allow_bengali=allow_bengali))
    
    scores = []
    for entry in output_data:
        # Assume each entry has a dictionary "replaced_concepts" mapping original -> replaced words
        replaced_words = {
            normalize(replaced, allow_bengali=allow_bengali)
            for original, replaced in entry.get("replaced_concepts", {}).items()
            if original != replaced and replaced
        }
        if not replaced_words:
            continue
        
        strict_matches = set()
        singular_plural_matches = set()
        fuzzy_matches = set()

        # Process strict matches.
        for word in replaced_words:
            strict = strict_exact_match(word, all_concepts, allow_bengali=allow_bengali)
            if strict:
                strict_matches.add(strict)

        remaining_words = replaced_words - strict_matches
        # Process singular/plural matches.
        for word in remaining_words:
            sp_match = singular_plural_match(word, all_concepts, allow_bengali=allow_bengali)
            if sp_match:
                singular_plural_matches.add(sp_match)
        
        # Process fuzzy matching on remaining words.
        remaining_for_fuzzy = remaining_words - singular_plural_matches
        for word in remaining_for_fuzzy:
            fuzzy_result = fuzzy_match(word, all_concepts, threshold=80, allow_bengali=allow_bengali)
            if fuzzy_result:
                fuzzy_matches.add(fuzzy_result)
        
        total = len(replaced_words)
        strict_match_score = len(strict_matches) / total if total > 0 else 0
        singular_plural_score = len(singular_plural_matches) / total if total > 0 else 0
        fuzzy_match_score = len(fuzzy_matches) / total if total > 0 else 0

        score_entry = {
            "adapted_text": entry.get("cultural_adapted_text", ""),
            "strict_matches": list(strict_matches),
            "singular_plural_matches": list(singular_plural_matches),
            "fuzzy_matches": list(fuzzy_matches),
            "total_replaced_words": total,
            "strict_match_score": strict_match_score,
            "singular_plural_score": singular_plural_score,
            "fuzzy_match_score": fuzzy_match_score,
            "replaced_words": list(replaced_words)
        }
        scores.append(score_entry)
    return scores, len(output_data)

def main():
    # Update these paths as needed.
    # For example, here we process English output files.
    output_dir = "/u/student/2023/ai23mtech14004/culture-evaluation/MGSM/output_en_prompt"
    # The concept file can contain multiple facets (e.g., textiles, architectures, etc.)
    concept_file_path = "/u/student/2023/ai23mtech14004/culture-evaluation/our_csis/csis/india_cultural_features.json"
    # Directory to save adaptation score files
    scores_dir = "/u/student/2023/ai23mtech14004/culture-evaluation/our_csis/concept_matching/concept_matching_en_prompt_MGSM"
    os.makedirs(scores_dir, exist_ok=True)
    
    # Process each .json output file separately.
    model_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    overall_scores = []
    overall_total_sentences = 0
    
    for filename in model_files:
        output_file_path = os.path.join(output_dir, filename)
        logging.info(f"Processing file: {output_file_path}")
        output_data = read_output_file(output_file_path)
        scores, total_sentences = calculate_adaptation_score(output_data, concept_file_path, allow_bengali=True)
        overall_scores.extend(scores)
        overall_total_sentences += total_sentences
        
        # Save scores for this individual model output.
        model_scores_path = os.path.join(scores_dir, f"adaptation_scores_{filename}.json")
        avg_strict, avg_singular_plural = save_scores_to_file(scores, total_sentences, model_scores_path)
        print(f"\nModel File: {filename}")
        print(f"  Total Sentences: {total_sentences}")
        print(f"  Entries with Replacements: {len(scores)}")
        print(f"  Average Strict Matches per Sentence: {avg_strict:.4f}")
        print(f"  Average Singular/Plural Matches per Sentence: {avg_singular_plural:.4f}")
        print(f"  Scores saved to: {model_scores_path}")
    
    # Save a merged adaptation scores file across all models.
    merged_scores_path = os.path.join(scores_dir, "merged_adaptation_scores_bengali.json")
    avg_strict_overall, avg_singular_plural_overall = save_scores_to_file(overall_scores, overall_total_sentences, merged_scores_path)
    print("\n=== Merged Dataset-wide Statistics ===")
    print(f"Total Sentences (All Models): {overall_total_sentences}")
    print(f"Entries with Replacements (All Models): {len(overall_scores)}")
    print(f"Average Strict Matches per Sentence: {avg_strict_overall:.4f}")
    print(f"Average Singular/Plural Matches per Sentence: {avg_singular_plural_overall:.4f}")
    print(f"Merged scores saved to: {merged_scores_path}")

if __name__ == "__main__":
    main()
