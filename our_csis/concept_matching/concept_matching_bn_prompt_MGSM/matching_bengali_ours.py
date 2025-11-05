import re
import os
import json
import logging
import inflect
from rapidfuzz import fuzz, process  # Added for fuzzy matching

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the inflect engine
inflect_engine = inflect.engine()

def normalize(word):
    """
    Normalize a word by removing unwanted characters and converting to lowercase.
    Allows Bengali characters (Unicode range \u0980-\u09FF), ASCII letters, digits, ₹, and whitespace.
    """
    if not word:
        return ""
    try:
        word = str(word)
        word = re.sub(r'[^\u0980-\u09FFa-zA-Z0-9₹\s]', '', word).lower().strip()
        return word
    except Exception as e:
        logging.error(f"Error processing normalization: {word}, Error: {e}")
        return word

def strict_exact_match(word, concept_set):
    """
    Return the normalized word if it exactly matches one in the concept_set.
    """
    normalized_word = normalize(word)
    if normalized_word in concept_set:
        return normalized_word
    return None

def singular_plural_match(word, concept_set):
    """
    Return a matching concept using singular/plural forms.
    """
    normalized_word = normalize(word)
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
    Read and parse a JSON output file.
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
    Merge all JSON output files from the given directory into a single list.
    """
    merged = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            file_data = read_output_file(filepath)
            merged.extend(file_data)
    return merged

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
    Save the adaptation scores and dataset-wide averages to a JSON file.
    Bengali characters are preserved (ensure_ascii=False).
    """
    try:
        total_strict_matches = sum(len(score["strict_matches"]) for score in scores)
        total_singular_plural_matches = sum(len(score["singular_plural_matches"]) for score in scores)
        total_fuzzy_matches = sum(len(score.get("fuzzy_matches", [])) for score in scores)
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

def calculate_adaptation_score(output_data, concept_file_path):
    """
    Given a list of output entries (merged from one or more files) and a Bengali concept file,
    calculate adaptation scores using strict, singular/plural, and fuzzy matching.
    """
    facet_concepts = read_concept_list(concept_file_path)
    # Merge all concepts from all facets into a single set (normalized)
    all_concepts = set()
    for concept_list in facet_concepts.values():
        for concept in concept_list:
            all_concepts.add(normalize(concept))
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
        # Process strict matches.
        for word in replaced_words:
            strict = strict_exact_match(word, all_concepts)
            if strict:
                strict_matches.add(strict)
        remaining_words = replaced_words - strict_matches
        # Process singular/plural matches.
        for word in remaining_words:
            sp_match = singular_plural_match(word, all_concepts)
            if sp_match:
                singular_plural_matches.add(sp_match)
        # Process fuzzy matches on words not already matched.
        remaining_for_fuzzy = remaining_words - singular_plural_matches
        for word in remaining_for_fuzzy:
            fuzzy_result = fuzzy_match(word, all_concepts)
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
    # Paths – update these as needed.
    output_dir = "culture-evaluation/MGSM/output_bn_prompt"
    concept_file_path = "culture-evaluation/our_csis/csis/bengali_csis/india_cultural_features.json"
    scores_dir = "culture-evaluation/our_csis/concept_matching/concept_matching_bn_prompt_MGSM"
    os.makedirs(scores_dir, exist_ok=True)
    
    # Process each model's output file separately.
    model_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    overall_scores = []
    overall_total = 0

    for filename in model_files:
        output_file_path = os.path.join(output_dir, filename)
        logging.info(f"Processing model file: {output_file_path}")
        output_data = read_output_file(output_file_path)
        scores, total_sentences = calculate_adaptation_score(output_data, concept_file_path)
        overall_scores.extend(scores)
        overall_total += total_sentences
        
        # Save adaptation scores for this model in a JSON file
        model_scores_path = os.path.join(scores_dir, f"adaptation_scores_{filename}.json")
        avg_strict, avg_singular_plural = save_scores_to_file(scores, total_sentences, model_scores_path)
        print(f"\nModel File: {filename}")
        print(f"  Total Sentences: {total_sentences}")
        print(f"  Entries with Replacements: {len(scores)}")
        print(f"  Average Strict Matches per Sentence: {avg_strict:.4f}")
        print(f"  Average Singular/Plural Matches per Sentence: {avg_singular_plural:.4f}")
        print(f"  Scores saved to: {model_scores_path}")
    
    # Also create a merged adaptation scores file across all models.
    merged_scores_path = os.path.join(scores_dir, "merged_adaptation_scores_bengali.json")
    avg_strict_overall, avg_singular_plural_overall = save_scores_to_file(overall_scores, overall_total, merged_scores_path)
    print("\n=== Merged Dataset-wide Statistics ===")
    print(f"Total Sentences (All Models): {overall_total}")
    print(f"Entries with Replacements (All Models): {len(overall_scores)}")
    print(f"Average Strict Matches per Sentence: {avg_strict_overall:.4f}")
    print(f"Average Singular/Plural Matches per Sentence: {avg_singular_plural_overall:.4f}")
    print(f"Merged scores saved to: {merged_scores_path}")

if __name__ == "__main__":
    main()
