import re
import os
import json
import logging
import inflect
from collections import defaultdict
from rapidfuzz import fuzz, process

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the inflect engine
inflect_engine = inflect.engine()

def normalize(word):
    """
    Normalize a word by removing unwanted characters and converting to lowercase.
    """
    if not word:
        return ""
    try:
        word = str(word)
        word = re.sub(r'[^a-zA-Z0-9₹\s]', '', word).lower().strip()
        return word
    except Exception as e:
        logging.error(f"Error processing normalization: {word}, Error: {e}")
        return word

def read_output_file(filepath):
    """
    Read and parse the JSON output file.
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
                            for entry in parsed_item:
                                if isinstance(entry, dict):
                                    parsed_data.append(entry)
                        elif isinstance(parsed_item, dict):
                            parsed_data.append(parsed_item)
                    except json.JSONDecodeError:
                        continue
                elif isinstance(item, list):
                    for entry in item:
                        if isinstance(entry, dict):
                            parsed_data.append(entry)
                elif isinstance(item, dict):
                    parsed_data.append(item)
        return parsed_data
    except Exception as e:
        logging.error(f"Error reading output file: {filepath}, Error: {e}")
        return []

def read_concept_list(filepath):
    """
    Read and parse the concept file structure.
    """
    if not os.path.exists(filepath):
        logging.error(f"Concept file not found: {filepath}")
        return {}, {}, {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        facet_concepts = {}
        state_concepts = defaultdict(list)
        concept_metadata = {}
        
        for facet, items in data.items():
            facet_concepts[facet] = []
            for item in items:
                name = item.get("name")
                state = item.get("state")
                if name and state:
                    normalized_name = normalize(name)
                    facet_concepts[facet].append(normalized_name)
                    state_concepts[state].append(normalized_name)
                    concept_metadata[normalized_name] = {
                        "facet": facet,
                        "state": state,
                        "original_name": name
                    }
                    
        return facet_concepts, state_concepts, concept_metadata
    except Exception as e:
        logging.error(f"Error reading concept file: {filepath}, Error: {e}")
        return {}, {}, {}

def fuzzy_match(word, concept_set, threshold=80):
    """
    Find fuzzy matches between a word and concepts using token sort ratio.
    Returns the best match if it exceeds the threshold, otherwise None.
    """
    if not word or not concept_set:
        return None
        
    try:
        # Use process.extractOne to find the best match
        best_match = process.extractOne(
            normalize(word),
            concept_set,
            scorer=fuzz.token_sort_ratio
        )
        
        if best_match and best_match[1] >= threshold:
            return best_match[0], best_match[1]
        return None
        
    except Exception as e:
        logging.error(f"Error in fuzzy matching for '{word}': {e}")
        return None

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

def calculate_adaptation_score(output_file_path, concept_file_path):
    """
    Calculate adaptation scores with facet, state statistics and fuzzy matching.
    """
    output_data = read_output_file(output_file_path)
    facet_concepts, state_concepts, concept_metadata = read_concept_list(concept_file_path)
    
    all_concepts = set()
    for concept_set in facet_concepts.values():
        all_concepts.update(concept_set)
    
    scores = []
    facet_matches = defaultdict(int)
    state_matches = defaultdict(int)
    
    for entry in output_data:
        replaced_words = {
            normalize(replaced)
            for original, replaced in entry.get("replaced_concepts", {}).items()
            if original != replaced and replaced
        }
        
        if not replaced_words:
            continue
            
        strict_matches = set()
        singular_plural_matches = set()
        fuzzy_matches = []
        
        # Process strict matches first
        for word in replaced_words:
            strict = strict_exact_match(word, all_concepts)
            if strict:
                strict_matches.add(strict)
                metadata = concept_metadata.get(strict)
                if metadata:
                    facet_matches[metadata["facet"]] += 1
                    state_matches[metadata["state"]] += 1
                
        remaining_words = replaced_words - strict_matches
        
        # Process singular/plural matches
        for word in remaining_words:
            sp_match = singular_plural_match(word, all_concepts)
            if sp_match:
                singular_plural_matches.add(sp_match)
                metadata = concept_metadata.get(sp_match)
                if metadata:
                    facet_matches[metadata["facet"]] += 1
                    state_matches[metadata["state"]] += 1
                    
        # Process fuzzy matches for remaining words
        remaining_for_fuzzy = remaining_words - singular_plural_matches
        for word in remaining_for_fuzzy:
            fuzzy_result = fuzzy_match(word, all_concepts)
            if fuzzy_result:
                matched_word, score = fuzzy_result
                fuzzy_matches.append({
                    "original": word,
                    "matched": matched_word,
                    "score": score
                })
                metadata = concept_metadata.get(matched_word)
                if metadata:
                    facet_matches[metadata["facet"]] += 1
                    state_matches[metadata["state"]] += 1
        
        total = len(replaced_words)
        strict_match_score = len(strict_matches) / total if total > 0 else 0
        singular_plural_score = len(singular_plural_matches) / total if total > 0 else 0
        fuzzy_match_score = len(fuzzy_matches) / total if total > 0 else 0
        
        score_entry = {
            "adapted_text": entry.get("cultural_adapted_text", ""),
            "original_text": entry.get("original_text", ""),
            "strict_matches": list(strict_matches),
            "singular_plural_matches": list(singular_plural_matches),
            "fuzzy_matches": fuzzy_matches,
            "total_replaced_words": total,
            "strict_match_score": strict_match_score,
            "singular_plural_score": singular_plural_score,
            "fuzzy_match_score": fuzzy_match_score,
            "replaced_words": list(replaced_words)
        }
        scores.append(score_entry)
    
    # Calculate percentages for statistics
    total_matches = sum(facet_matches.values())
    
    facet_stats = {
        facet: {
            "count": count,
            "percentage": (count / total_matches * 100) if total_matches > 0 else 0
        }
        for facet, count in facet_matches.items()
    }
    
    state_stats = {
        state: {
            "count": count,
            "percentage": (count / total_matches * 100) if total_matches > 0 else 0
        }
        for state, count in state_matches.items()
    }
    
    return scores, len(output_data), facet_stats, state_stats

def save_scores_to_file(scores, total_sentences, facet_stats, state_stats, filepath):
    """
    Save the adaptation scores and all statistics to a file.
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
            "facet_statistics": facet_stats,
            "state_statistics": state_stats,
            "detailed_scores": scores
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        logging.info(f"Scores saved to {scores_file_path}")
        
        return avg_strict, avg_singular_plural, avg_fuzzy
    except Exception as e:
        logging.error(f"Error saving scores to file: {filepath}, Error: {e}")
        return 0, 0, 0

def main():
    # Directory paths
    output_dir = "culture-evaluation/output/new_prompt_english"
    concept_file_path = "culture-evaluation/our_csis/csis/india_cultural_features.json"
    scores_dir = "culture-evaluation/our_csis/concept_matching"
    
    # Create scores directory if it doesn't exist
    os.makedirs(scores_dir, exist_ok=True)
    
    # Process each JSON file
    for filename in os.listdir(output_dir):
        if filename.endswith(".json") and "deepseek" not in filename:
            output_file_path = os.path.join(output_dir, filename)
            scores_file_path = os.path.join(scores_dir, f"concept_matching_scores_{filename}")
            
            print(f"\nProcessing file: {output_file_path}")
            scores, total_sentences, facet_stats, state_stats = calculate_adaptation_score(
                output_file_path, concept_file_path
            )
            
            if scores:
                avg_strict, avg_singular_plural, avg_fuzzy = save_scores_to_file(
                    scores, total_sentences, facet_stats, state_stats, scores_file_path
                )
                
                print("\n=== Dataset-wide Statistics ===")
                print(f"File: {filename}")
                print(f"Total Sentences in Dataset: {total_sentences}")
                print(f"Entries with Replacements: {len(scores)}")
                print(f"Total Strict Matches: {sum(len(score['strict_matches']) for score in scores)}")
                print(f"Total Singular/Plural Matches: {sum(len(score['singular_plural_matches']) for score in scores)}")
                print(f"Total Fuzzy Matches: {sum(len(score['fuzzy_matches']) for score in scores)}")
                print(f"Average Strict Matches per Sentence: {avg_strict:.4f}")
                print(f"Average Singular/Plural Matches per Sentence: {avg_singular_plural:.4f}")
                print(f"Average Fuzzy Matches per Sentence: {avg_fuzzy:.4f}")
                
                print("\n=== Facet Statistics ===")
                for facet, stats in facet_stats.items():
                    print(f"{facet}: {stats['count']} matches ({stats['percentage']:.2f}%)")
                    
                print("\n=== State Statistics ===")
                for state, stats in state_stats.items():
                    print(f"{state}: {stats['count']} matches ({stats['percentage']:.2f}%)")
                
                print(f"\nScores saved to: {scores_file_path}")
            else:
                print(f"No adaptation scores were calculated for {filename}.")

if __name__ == "__main__":
    main()