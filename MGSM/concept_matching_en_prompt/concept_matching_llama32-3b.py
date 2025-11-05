import re
import os
import json
import logging
import inflect

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
    Read and parse the concept file.
    """
    if not os.path.exists(filepath):
        logging.error(f"Concept file not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_char = None
            while True:
                char = f.read(1)
                if not char:
                    break
                if not char.isspace():
                    first_char = char
                    break
            f.seek(0)

            facet_concepts = {}
            if first_char == '[':
                data = json.load(f)
                for entry in data:
                    facet = entry.get("facet")
                    concepts = entry.get("concepts", [])
                    concept_names = [concept.get("name") for concept in concepts if concept.get("name")]
                    if facet:
                        facet_concepts.setdefault(facet, []).extend(concept_names)
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    facet = entry.get("facet")
                    concepts = entry.get("concepts", [])
                    concept_names = [concept.get("name") for concept in concepts if concept.get("name")]
                    if facet:
                        facet_concepts.setdefault(facet, []).extend(concept_names)
            return facet_concepts
    except Exception as e:
        logging.error(f"Error reading concept file: {filepath}, Error: {e}")
        return {}

def save_scores_to_file(scores, total_sentences, filepath):
    """
    Save the adaptation scores and dataset-wide averages to a file in JSON format.
    """
    try:
        # Calculate total matches across all entries
        total_strict_matches = sum(len(score["strict_matches"]) for score in scores)
        total_singular_plural_matches = sum(len(score["singular_plural_matches"]) for score in scores)
        
        # Calculate averages based on total sentences
        avg_strict = total_strict_matches / total_sentences
        avg_singular_plural = total_singular_plural_matches / total_sentences
        
        final_output = {
            "dataset_statistics": {
                "total_sentences": total_sentences,
                "entries_with_replacements": len(scores),
                "total_strict_matches": total_strict_matches,
                "total_singular_plural_matches": total_singular_plural_matches,
                "average_strict_match_per_sentence": avg_strict,
                "average_singular_plural_match_per_sentence": avg_singular_plural
            },
            "detailed_scores": scores
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)
        logging.info(f"Scores saved to {filepath}")
        
        return avg_strict, avg_singular_plural
    except Exception as e:
        logging.error(f"Error saving scores to file: {filepath}, Error: {e}")
        return 0, 0

def calculate_adaptation_score(output_file_path, concept_file_path):
    """
    Calculate adaptation scores using only strict and singular/plural matching.
    """
    output_data = read_output_file(output_file_path)
    facet_concepts = read_concept_list(concept_file_path)
    
    all_concepts = set()
    for concept_set in facet_concepts.values():
        all_concepts.update(concept_set)
    
    scores = []
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
        
        total = len(replaced_words)
        strict_match_score = len(strict_matches) / total if total > 0 else 0
        singular_plural_score = len(singular_plural_matches) / total if total > 0 else 0
        
        score_entry = {
            "adapted_text": entry.get("cultural_adapted_text", ""),
            "strict_matches": list(strict_matches),
            "singular_plural_matches": list(singular_plural_matches),
            "total_replaced_words": total,
            "strict_match_score": strict_match_score,
            "singular_plural_score": singular_plural_score,
            "replaced_words": list(replaced_words)
        }
        scores.append(score_entry)
    
    return scores, len(output_data)

def main():
    # Update these file paths with your actual file locations
    output_file_path = "culture-evaluation/MGSM/output_en_prompt/output_llama-3.2-3b.json"
    concept_file_path = "culture-evaluation/CANDLE/concepts_filtered.jsonl"
    scores_file_path = "culture-evaluation/MGSM/concept_matching_en_prompt/llama-32-3b-adaptation_scores.txt"
    
    print("Starting adaptation score calculation...")
    scores, total_sentences = calculate_adaptation_score(output_file_path, concept_file_path)
    
    if scores:
        # Save scores and get averages
        avg_strict, avg_singular_plural = save_scores_to_file(scores, total_sentences, scores_file_path)
        
        print("\n=== Dataset-wide Statistics ===")
        print(f"Total Sentences in Dataset: {total_sentences}")
        print(f"Entries with Replacements: {len(scores)}")
        print(f"Total Strict Matches: {sum(len(score['strict_matches']) for score in scores)}")
        print(f"Total Singular/Plural Matches: {sum(len(score['singular_plural_matches']) for score in scores)}")
        print(f"Average Strict Matches per Sentence: {avg_strict:.4f}")
        print(f"Average Singular/Plural Matches per Sentence: {avg_singular_plural:.4f}")
        
        # Print detailed statistics for each entry with replacements
        print("\n=== Detailed Statistics ===")
        for i, score in enumerate(scores, 1):
            print(f"\nEntry {i}:")
            print(f"Total Replaced Words: {score['total_replaced_words']}")
            print(f"Strict Matches: {len(score['strict_matches'])}")
            print(f"Singular/Plural Matches: {len(score['singular_plural_matches'])}")
    else:
        print("No adaptation scores were calculated.")

if __name__ == "__main__":
    main()