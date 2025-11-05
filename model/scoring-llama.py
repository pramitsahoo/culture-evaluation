

############### LLaMa with New Evaluation Criteria ###############
import csv
import json
from typing import Dict, Union
import numpy as np
from dataclasses import dataclass
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from datetime import datetime
import pathlib
import re

# Set up logging with both file and console handlers
def setup_logging():
    log_dir = pathlib.Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"indian_cultural_scorer_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Create output directories
def setup_directories():
    base_dir = pathlib.Path("output")
    dirs = {
        'base': base_dir,
        'logs': base_dir / "logs",
        'LLM_New_Scores_Llama': base_dir / "LLM_New_Scores_Llama/llama-2",
        'prompts': base_dir / "LLM_New_Scores_Llama/llama-2/prompts",
        'responses': base_dir / "LLM_New_Scores_Llama/llama-2/responses",
        'explanations': base_dir / "LLM_New_Scores_Llama/llama-2/explanations"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

DIRS = setup_directories()

# Set up Hugging Face authentication if required
os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGINGFACE_HUB_TOKEN"

@dataclass
class ScoringResult:
    score: float
    explanation: str

@dataclass
class EvaluationResults:
    cultural_relevance: ScoringResult
    language_fluency: ScoringResult
    mathematical_integrity: ScoringResult

class IndianCulturalScorer:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        logger.info(f"Successfully initialized model: {model_name}")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_scoring_prompt(self, original_text: str, adapted_text: str) -> str:
        prompt_template = f"""Consider yourself as an AI expert trained to evaluate the cultural adaptation of a given text.

Original Text:
{original_text}

Adapted Text:
{adapted_text}

Target Culture: Indian

Rate each criterion on a scale of 0-5 and provide a detailed explanation for your score:

1. Cultural Relevance (0-5):
- 0: No adaptation or complete transliterations/translations: Text maintains western concepts with no attempt at cultural adaptation
- 1: Non-sensical adaptations: Includes culturally inappropriate or nonsensical elements (e.g., paratha growing on plants, using rupees as age units)
- 2: Simple replacement of proper nouns: Only names are changed to Indian names without contextual adaptation
- 3: Proper nouns changed along with context: Names and some objects are replaced with culturally relevant alternatives
- 4: Multiple entities changed with deeper connection to Indian culture: Substantial adaptation with culturally relevant scenarios and objects
- 5: Culturally resonating with no further adaptations possible: Complete and authentic cultural adaptation that feels natural to the target audience

2. Language Fluency (0-5):
- 0: No Indian language elements: Uses completely western terminology and expressions
- 1: Very poor language adaptation: Minimal attempt with incorrect use of Indian terms
- 2: Poor language adaptation: Contains few Indian terms but used incorrectly or inappropriately
- 3: Moderate language adaptation: Uses some Indian terms correctly but lacks natural flow
- 4: Good language adaptation: Effectively incorporates Indian English and terminology with natural flow
- 5: Perfect language adaptation: Seamlessly blends Indian terminology, expressions, and natural language patterns

3. Mathematical Integrity (0-5):
- 0: Completely incorrect mathematics: Problem is mathematically nonsensical after adaptation
- 1: Incorrect mathematics: Problems are mathematically unsound or illogical
- 2: Poor mathematical clarity: Basic math is correct but poorly integrated with cultural context
- 3: Moderate mathematical structure: Problems are mathematically sound but could be better contextualized
- 4: Good mathematical adaptation: Well-structured problems with clear cultural context
- 5: Perfect mathematical integration: Mathematics seamlessly integrated with cultural elements while maintaining complete accuracy

Format your response exactly like this (make sure to use actual numeric scores instead of placeholders):

Here are some examples for your reference but you have to properly score them 0-5 based on above mentioned criterias :

Cultural Relevance: 5
Explanation: The adapted text is deeply integrated with Indian culture, accurately reflecting significant traditions and practices.

Language Fluency: 5
Explanation: The text uses natural Indian expressions and terminology, making it easy to read and culturally coherent.

Mathematical Integrity: 5
Explanation: The mathematical problem is presented correctly and is well-integrated into the cultural context.

Do not output any extra text or placeholders such as "[score]".
"""
        # Save prompt to file
        prompt_file = DIRS['prompts'] / f"prompt_{self.timestamp}_{hash(original_text)}.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_template)
            
        return prompt_template

    @staticmethod
    def extract_score(text: str) -> float:
        """
        Extracts the first occurrence of a number from the given text.
        If a placeholder "[score]" is found, or no valid number is detected, raise a ValueError.
        """
        if "[score]" in text:
            error_msg = "Placeholder [score] found in text."
            logger.error(error_msg)
            raise ValueError(error_msg)
        match = re.search(r'(\d+(\.\d+)?)', text)
        if match:
            return float(match.group(1))
        else:
            error_msg = f"Could not extract a valid score from: {text}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def parse_response(self, response: str) -> EvaluationResults:
        lines = response.split('\n')
        scores = {}
        explanations = {}
        current_criterion = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Cultural Relevance:'):
                current_criterion = 'cultural_relevance'
                score_text = line.split(':', 1)[1].strip()
                scores[current_criterion] = self.extract_score(score_text)
            elif line.startswith('Language Fluency:'):
                current_criterion = 'language_fluency'
                score_text = line.split(':', 1)[1].strip()
                scores[current_criterion] = self.extract_score(score_text)
            elif line.startswith('Mathematical Integrity:'):
                current_criterion = 'mathematical_integrity'
                score_text = line.split(':', 1)[1].strip()
                scores[current_criterion] = self.extract_score(score_text)
            elif line.startswith('Explanation:') and current_criterion:
                explanations[current_criterion] = line.split(':', 1)[1].strip()

        # Save the parsed response
        response_file = DIRS['responses'] / f"response_{self.timestamp}_{hash(response)}.txt"
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(response)

        return EvaluationResults(
            cultural_relevance=ScoringResult(
                scores.get('cultural_relevance', 0),
                explanations.get('cultural_relevance', "")
            ),
            language_fluency=ScoringResult(
                scores.get('language_fluency', 0),
                explanations.get('language_fluency', "")
            ),
            mathematical_integrity=ScoringResult(
                scores.get('mathematical_integrity', 0),
                explanations.get('mathematical_integrity', "")
            )
        )

    def save_evaluation_results(self, results: EvaluationResults, original_text: str, adapted_text: str, identifier: str):
        results_file = DIRS['explanations'] / f"evaluation_{identifier}_{self.timestamp}.txt"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("Cultural Adaptation Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Original Text:\n")
            f.write(original_text + "\n\n")
            
            f.write("Adapted Text:\n")
            f.write(adapted_text + "\n\n")
            
            f.write("Evaluation Scores and Explanations:\n")
            f.write("-" * 30 + "\n\n")
            
            f.write("1. Cultural Relevance\n")
            f.write(f"Score: {results.cultural_relevance.score}/5\n")
            f.write(f"Explanation: {results.cultural_relevance.explanation}\n\n")
            
            f.write("2. Language Fluency\n")
            f.write(f"Score: {results.language_fluency.score}/5\n")
            f.write(f"Explanation: {results.language_fluency.explanation}\n\n")
            
            f.write("3. Mathematical Integrity\n")
            f.write(f"Score: {results.mathematical_integrity.score}/5\n")
            f.write(f"Explanation: {results.mathematical_integrity.explanation}\n\n")
            
            average_score = (results.cultural_relevance.score + 
                           results.language_fluency.score + 
                           results.mathematical_integrity.score) / 3
            
            f.write("Overall Score:\n")
            f.write(f"Average Score: {average_score:.2f}/5\n")

    def evaluate_adaptation(self, original_text: str, adapted_text: str, identifier: str = None) -> EvaluationResults:
        if identifier is None:
            identifier = "eval_indian"
        
        prompt = self.create_scoring_prompt(original_text, adapted_text)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Raw model response:")
        logger.info(response)
        
        results = self.parse_response(response)
        self.save_evaluation_results(results, original_text, adapted_text, identifier)
        
        return results

def process_evaluation_files(original_filepath: str, adapted_filepath: str):
    scorer = IndianCulturalScorer()
    results_for_csv = []
    
    # Read original texts
    with open(original_filepath, 'r', encoding='utf-8') as orig_file:
        original_lines = orig_file.readlines()
    
    # Read adapted file content
    with open(adapted_filepath, 'r', encoding='utf-8') as adapt_file:
        adapted_content = adapt_file.read().strip()
    
    try:
        adapted_data = json.loads(adapted_content)
        if isinstance(adapted_data, list):
            adapted_samples = adapted_data
        else:
            adapted_samples = adapted_content.splitlines()
    except Exception as e:
        logger.error(f"Error parsing adapted file as JSON: {e}")
        adapted_samples = adapted_content.splitlines()
    
    num_orig = len(original_lines)
    num_adapt = len(adapted_samples)
    logger.info(f"Original samples: {num_orig}, Adapted samples: {num_adapt}")
    
    sample_count = min(num_orig, num_adapt)
    
    for i in range(sample_count):
        try:
            orig_obj = json.loads(original_lines[i])
            original_text = orig_obj.get("question", "").strip()
            
            sample = adapted_samples[i]
            if isinstance(sample, str):
                try:
                    adapt_array = json.loads(sample)
                except Exception as e:
                    logger.error(f"Error parsing adapted sample {i+1}: {e}")
                    continue
            else:
                adapt_array = sample
                
            if not adapt_array or not isinstance(adapt_array, list):
                logger.error(f"Sample {i+1}: Adapted JSON is not a list or is empty.")
                continue
            adapted_obj = adapt_array[0]
            adapted_text = adapted_obj.get("cultural_adapted_text", "").strip()
            
            identifier = f"eval_{i+1}"
            logger.info(f"Evaluating problem {identifier}...")
            eval_results = scorer.evaluate_adaptation(original_text, adapted_text, identifier)
            logger.info(f"Evaluation completed for {identifier}.")
            
            avg_score = (eval_results.cultural_relevance.score +
                        eval_results.language_fluency.score +
                        eval_results.mathematical_integrity.score) / 3
            
            results_for_csv.append({
                "Identifier": identifier,
                "Original Text": original_text,
                "Adapted Text": adapted_text,
                "Cultural Relevance Score": eval_results.cultural_relevance.score,
                "Language Fluency Score": eval_results.language_fluency.score,
                "Mathematical Integrity Score": eval_results.mathematical_integrity.score,
                "Average Score": f"{avg_score:.2f}"
            })
        except Exception as e:
            logger.error(f"Error processing problem {i+1}: {e}")
    
    # Save CSV in the LLM_Scores_Bengali_GSM directory
    csv_filepath = DIRS['LLM_New_Scores_Llama'] / f"llama-2-evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filepath, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ["Identifier", "Original Text", "Adapted Text", 
                     "Cultural Relevance Score", "Language Fluency Score", 
                     "Mathematical Integrity Score", "Average Score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_for_csv:
            writer.writerow(row)
    
    logger.info(f"CSV file written to {csv_filepath}")

def main():
    original_filepath = "/u/student/2023/ai23mtech14004/culture-evaluation/dataset/gsm_8k/test.jsonl"
    adapted_filepath ="/u/student/2023/ai23mtech14004/culture-evaluation/output/new_prompt_english/llama-2_gsm_8k_test.json"
    process_evaluation_files(original_filepath, adapted_filepath)
    print("\nEvaluations for 10 samples completed. Check the output/LLM_Scores_Bengali_GSM/gemma-2-9b directory for results.")

if __name__ == "__main__":
    main()