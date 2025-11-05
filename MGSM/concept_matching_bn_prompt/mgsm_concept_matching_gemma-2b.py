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

def read_extracted_concepts_file(filepath):
    """
    Read the file containing extracted replaced concepts.
    The file is expected to have entries in the following format:
    
      Entry X-Y replaced concepts:
      {
        "key1": "value1",
        "key2": "value2",
         ...
      }
      ----------------------------------------
    
    Lines that start with "Error parsing item" will be skipped.
    Each valid JSON block is wrapped in a dict under "replaced_concepts" to be consistent
    with the rest of the code.
    """
    if not os.path.exists(filepath):
        logging.error(f"Extracted concepts file not found: {filepath}")
        return []
    
    entries = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split the file by the separator line
        blocks = content.split('----------------------------------------')
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Skip blocks that begin with an error message
            if block.startswith("Error parsing item"):
                logging.error(f"Skipping block due to error message: {block}")
                continue
            
            lines = block.splitlines()
            # If the block starts with an "Entry" header, remove that line.
            if lines and lines[0].startswith("Entry"):
                json_text = "\n".join(lines[1:]).strip()
            else:
                json_text = block
            
            if json_text:
                try:
                    data = json.loads(json_text)
                    # Wrap the JSON dictionary in our expected format
                    entries.append({"replaced_concepts": data})
                except Exception as e:
                    logging.error(f"Error parsing JSON block: {json_text[:50]}... Error: {e}")
        return entries
    except Exception as e:
        logging.error(f"Error reading extracted concepts file: {filepath}, Error: {e}")
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
        total_fuzzy_matches = sum(len(score["fuzzy_matches"]) for score in scores)
        
        # Calculate averages based on total sentences
        avg_strict = total_strict_matches / total_sentences
        avg_singular_plural = total_singular_plural_matches / total_sentences
        avg_fuzzy = total_fuzzy_matches / total_sentences
        
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
            # Set ensure_ascii=False so that Bengali characters are not escaped
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        logging.info(f"Scores saved to {filepath}")
        
        return avg_strict, avg_singular_plural, avg_fuzzy
    except Exception as e:
        logging.error(f"Error saving scores to file: {filepath}, Error: {e}")
        return 0, 0, 0

def calculate_adaptation_score(output_file_path, concept_file_path):
    """
    Calculate adaptation scores using strict, singular/plural, and fuzzy matching.
    Depending on the file extension, this function will use the appropriate reader.
    """
    # Use the new reader for the extracted_concepts.txt file (which is a .txt file)
    if output_file_path.endswith(".txt"):
        output_data = read_extracted_concepts_file(output_file_path)
    else:
        # Fallback to the original function if needed
        output_data = read_output_file(output_file_path)
    
    facet_concepts = read_concept_list(concept_file_path)
    
    all_concepts = set()
    for concept_set in facet_concepts.values():
        all_concepts.update(concept_set)
    
    scores = []
    for entry in output_data:
        # entry is expected to have the key "replaced_concepts" from our reader
        replaced_concepts = entry.get("replaced_concepts", {})
        replaced_words = {
            normalize(replaced)
            for original, replaced in replaced_concepts.items()
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
                
        remaining_words_after_sp = remaining_words - singular_plural_matches
        
        # Process fuzzy matches on remaining words
        for word in remaining_words_after_sp:
            fm = fuzzy_match(word, all_concepts)
            if fm:
                fuzzy_matches.add(fm)
        
        total = len(replaced_words)
        strict_match_score = len(strict_matches) / total if total > 0 else 0
        singular_plural_score = len(singular_plural_matches) / total if total > 0 else 0
        fuzzy_match_score = len(fuzzy_matches) / total if total > 0 else 0
        
        score_entry = {
            "adapted_text": entry.get("cultural_adapted_text", ""),  # May be empty in this file
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
    # Note: output_file_path now points to your extracted_concepts.txt file.
    output_file_path = "culture-evaluation/MGSM/output_bn_prompt/extracted_concepts.txt"
    concept_file_path = "culture-evaluation/CANDLE/concepts_bengali.jsonl"
    scores_file_path = "culture-evaluation/MGSM/concept_matching_bn_prompt/gemma-2-2badaptation_scores_bengali.txt"
    
    print("Starting adaptation score calculation...")
    scores, total_sentences = calculate_adaptation_score(output_file_path, concept_file_path)
    
    if scores:
        # Save scores and get averages
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
        
        # Print detailed statistics for each entry with replacements
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
