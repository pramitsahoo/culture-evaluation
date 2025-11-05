import re
import os
import json
import logging
import inflect
from rapidfuzz import fuzz, process  # For fuzzy matching

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the inflect engine
inflect_engine = inflect.engine()

def normalize(word):
    """
    Normalize a word by removing unwanted characters and converting to lowercase.
    This version allows Bengali characters (Unicode range \u0980-\u09FF) along with
    ASCII letters, digits, the rupee symbol, and whitespace.
    """
    if not word:
        return ""
    try:
        word = str(word)
        # Allow Bengali characters (U+0980-U+09FF), a-z, A-Z, digits, ₹, and whitespace.
        word = re.sub(r'[^\u0980-\u09FFa-zA-Z0-9₹\s]', '', word).lower().strip()
        return word
    except Exception as e:
        logging.error(f"Error processing normalization: {word}, Error: {e}")
        return word

def strict_exact_match(word, concept_set):
    """
    Find only exact matches between a word and the set of concepts.
    """
    normalized_word = normalize(word)
    if normalized_word in concept_set:
        return normalized_word
    return None

def singular_plural_match(word, concept_set):
    """
    Find matches between a word and concepts using singular/plural forms.
    """
    normalized_word = normalize(word)
    if not normalized_word:
        return None
    try:
        # Try getting singular form
        singular = inflect_engine.singular_noun(normalized_word)
        if singular and singular in concept_set:
            return singular
        # Try getting plural form
        plural = inflect_engine.plural_noun(normalized_word)
        if plural and plural in concept_set:
            return plural
    except Exception as e:
        logging.error(f"Error in singular/plural conversion for '{normalized_word}': {e}")
    return None

def fuzzy_match(word, concept_set, threshold=80):
    """
    Return a matching concept using fuzzy matching.
    Uses RapidFuzz's token sort ratio to find the best match.
    If the best match's score meets or exceeds the threshold, return the matched concept.
    """
    if not word or not concept_set:
        return None
    try:
        normalized_word = normalize(word)
        best_match = process.extractOne(normalized_word, concept_set, scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= threshold:
            return best_match[0]
        return None
    except Exception as e:
        logging.error(f"Error in fuzzy matching for '{word}': {e}")
        return None

def read_output_file(filepath):
    """
    Read and parse the txt output file with the given structure.
    Extracts JSON blocks following lines that start with "Entry" and returns a list of entries.
    """
    if not os.path.exists(filepath):
        logging.error(f"Output file not found: {filepath}")
        return []

    entries = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_json_lines = []
        reading_json = False
        
        for line in lines:
            stripped = line.strip()
            # Start a new entry when a line starts with "Entry" and contains "replaced concepts:"
            if stripped.startswith("Entry") and "replaced concepts:" in stripped:
                # If we were reading a previous JSON block, try to parse it.
                if current_json_lines:
                    json_str = "\n".join(current_json_lines)
                    try:
                        entry = json.loads(json_str)
                        entries.append({"replaced_concepts": entry})
                    except json.JSONDecodeError as e:
                        logging.error(f"Error parsing JSON block: {e}")
                    current_json_lines = []
                reading_json = True  # Next lines should be part of JSON block
                continue

            # Stop reading JSON when hitting a separator or error message.
            if reading_json and (set(stripped) == {"-"} or stripped.startswith("Error")):
                if current_json_lines:
                    json_str = "\n".join(current_json_lines)
                    try:
                        entry = json.loads(json_str)
                        entries.append({"replaced_concepts": entry})
                    except json.JSONDecodeError as e:
                        logging.error(f"Error parsing JSON block: {e}")
                    current_json_lines = []
                reading_json = False
                continue

            # If currently reading a JSON block, collect the lines.
            if reading_json:
                current_json_lines.append(line.rstrip())

        # In case the file does not end with a separator, try to parse remaining lines.
        if current_json_lines:
            json_str = "\n".join(current_json_lines)
            try:
                entry = json.loads(json_str)
                entries.append({"replaced_concepts": entry})
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing final JSON block: {e}")
        
        return entries

    except Exception as e:
        logging.error(f"Error reading output file: {filepath}, Error: {e}")
        return []

def read_concept_list(filepath):
    """
    Read and parse the Bengali concept file that may contain multiple facets.
    Expected format: a JSON object with one or more top-level keys.
    Each top-level key (e.g., "architectures", "festivals", etc.) contains a list of region entries.
    Each region entry is a dict with keys like "name", "type", and "items".
    From each entry’s "items" list, the first key that includes "name" (case-insensitive)
    is used as the concept.
    """
    if not os.path.exists(filepath):
        logging.error(f"Concept file not found: {filepath}")
        return {}
    facet_concepts = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            for facet, entries in data.items():
                facet_concepts.setdefault(facet, [])
                if isinstance(entries, list):
                    for entry in entries:
                        items = entry.get("items", [])
                        for item in items:
                            for key, value in item.items():
                                if "name" in key.lower() and value:
                                    facet_concepts[facet].append(value)
                                    break  # take the first matching key per item
        elif isinstance(data, list):
            for entry in data:
                facet = entry.get("facet") or entry.get("name")
                if facet:
                    facet_concepts.setdefault(facet, [])
                    items = entry.get("items", [])
                    for item in items:
                        for key, value in item.items():
                            if "name" in key.lower() and value:
                                facet_concepts[facet].append(value)
                                break
        return facet_concepts
    except Exception as e:
        logging.error(f"Error reading concept file: {filepath}, Error: {e}")
        return {}

def save_scores_to_file(scores, total_sentences, filepath):
    """
    Save the adaptation scores and dataset-wide averages to a file in JSON format.
    Bengali characters are preserved (ensure_ascii=False).
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
        return avg_strict, avg_singular_plural, avg_fuzzy
    except Exception as e:
        logging.error(f"Error saving scores to file: {filepath}, Error: {e}")
        return 0, 0, 0

def calculate_adaptation_score(output_file_path, concept_file_path):
    """
    Calculate adaptation scores using strict, singular/plural, and fuzzy matching.
    """
    output_data = read_output_file(output_file_path)
    facet_concepts = read_concept_list(concept_file_path)
    
    # Merge all concepts from all facets into a single set (normalized)
    all_concepts = set()
    for concept_set in facet_concepts.values():
        all_concepts.update(concept_set)
    
    scores = []
    for entry in output_data:
        # Expect each entry to have a "replaced_concepts" dict
        replaced_words = {
            normalize(replaced)
            for original, replaced in entry.get("replaced_concepts", {}).items()
            if original != replaced and replaced
        }
        
        if not replaced_words:
            continue
            
        strict_matches = set()
        singular_plural_matches = set()
        fuzzy_matches = set()
        
        # Process strict matches
        for word in replaced_words:
            strict = strict_exact_match(word, all_concepts)
            if strict:
                strict_matches.add(strict)
                
        remaining_words = replaced_words - strict_matches
        
        # Process singular/plural matches
        for word in remaining_words:
            sp_match = singular_plural_match(word, all_concepts)
            if sp_match:
                singular_plural_matches.add(sp_match)
        
        # Process fuzzy matches on remaining words
        remaining_for_fuzzy = remaining_words - singular_plural_matches
        for word in remaining_for_fuzzy:
            fm = fuzzy_match(word, all_concepts)
            if fm:
                fuzzy_matches.add(fm)
        
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
    # Update these file paths with your actual file locations.
    output_file_path = "culture-evaluation/output/bengali/extracted_concepts.txt"
    concept_file_path = "culture-evaluation/our_csis/csis/bengali_csis/india_cultural_features.json"
    scores_file_path = "culture-evaluation/our_csis/concept_matching/concept_matching_bengali_GSM/gemma-2-2b-adaptation_scores_bengali.txt"
    
    print("Starting adaptation score calculation...")
    scores, total_sentences = calculate_adaptation_score(output_file_path, concept_file_path)
    
    if scores:
        avg_strict, avg_singular_plural, avg_fuzzy = save_scores_to_file(scores, total_sentences, scores_file_path)
        
        print("\n=== Dataset-wide Statistics ===")
        print(f"Total Sentences in Dataset: {total_sentences}")
        print(f"Entries with Replacements: {len(scores)}")
        print(f"Total Strict Matches: {sum(len(score['strict_matches']) for score in scores)}")
        print(f"Total Singular/Plural Matches: {sum(len(score['singular_plural_matches']) for score in scores)}")
        print(f"Total Fuzzy Matches: {sum(len(score['fuzzy_matches']) for score in scores)}")
        print(f"Average Strict Matches per Sentence: {avg_strict:.4f}")
        print(f"Average Singular/Plural Matches per Sentence: {avg_singular_plural:.4f}")
        print(f"Average Fuzzy Matches per Sentence: {avg_fuzzy:.4f}")
        
        print("\n=== Detailed Statistics ===")
        for i, score in enumerate(scores, 1):
            print(f"\nEntry {i}:")
            print(f"Total Replaced Words: {score['total_replaced_words']}")
            print(f"Strict Matches: {len(score['strict_matches'])}")
            print(f"Singular/Plural Matches: {len(score['singular_plural_matches'])}")
            print(f"Fuzzy Matches: {len(score['fuzzy_matches'])}")
    else:
        print("No adaptation scores were calculated.")

if __name__ == "__main__":
    main()
