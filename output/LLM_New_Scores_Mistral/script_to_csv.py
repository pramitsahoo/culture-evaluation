def process_directory(input_dir, output_file='cultural_adaptation_results.csv'):
    """Process all text files in a directory and write results to a CSV file."""
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a valid directory.")
        return
    
    # Find all text files in the directory
    file_paths = glob.glob(os.path.join(input_dir, '*.txt'))
    
    if not file_paths:
        print(f"No text files found in '{input_dir}'.")
        return
    
    all_evaluations = []
    
    print(f"Processing {len(file_paths)} files from '{input_dir}'...")
    
    # Process each file with a progress bar
    for file_path in tqdm(file_paths, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Extract file name without extension to use as part of identifier
            file_name = os.path.basename(file_path).replace('.txt', '')
            
            # Process the content
            evaluations = extract_evaluation_data(content, identifier_prefix=file_name)
            all_evaluations.extend(evaluations)
            
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
    
    # Write all evaluations to a single CSV file
    write_to_csv(all_evaluations, output_file)
    
    print(f"Processed {len(all_evaluations)} evaluations from {len(file_paths)} files.")
    print(f"Results saved to '{output_file}'.")
import re
import csv
import sys
import os
import glob
from tqdm import tqdm

def extract_evaluation_data(text, identifier_prefix="eval"):
    """Extract evaluation data from the given text."""
    evaluations = []
    
    # Split into individual evaluations
    eval_blocks = re.split(r'={10,}|\.{10,}', text)
    
    identifier_counter = 1
    
    for block in eval_blocks:
        if not block.strip():
            continue
            
        # Extract original text
        original_match = re.search(r'Original Text:(.*?)(?=Adapted Text:|$)', block, re.DOTALL)
        original_text = original_match.group(1).strip() if original_match else ""
        
        # Extract adapted text
        adapted_match = re.search(r'Adapted Text:(.*?)(?=Evaluation Scores|$)', block, re.DOTALL)
        adapted_text = adapted_match.group(1).strip() if adapted_match else ""
        
        # Extract scores
        cultural_relevance = re.search(r'Cultural Relevance Score:\s*(\d+\.\d+)/5', block)
        cultural_score = cultural_relevance.group(1) if cultural_relevance else ""
        
        language_fluency = re.search(r'Language Fluency Score:\s*(\d+\.\d+)/5', block)
        language_score = language_fluency.group(1) if language_fluency else ""
        
        mathematical_integrity = re.search(r'Mathematical Integrity Score:\s*(\d+\.\d+)/5', block)
        math_score = mathematical_integrity.group(1) if mathematical_integrity else ""
        
        # Extract average score
        average_score = re.search(r'Average Score:\s*(\d+\.\d+)/5', block)
        if not average_score:
            average_score = re.search(r'Overall Score:.*?Average Score:\s*(\d+\.\d+)', block)
        avg_score = average_score.group(1) if average_score else ""
        
        if original_text and adapted_text:
            evaluations.append({
                'Identifier': f'{identifier_prefix}_{identifier_counter}',
                'Original Text': original_text,
                'Adapted Text': adapted_text,
                'Cultural Relevance Score': cultural_score,
                'Language Fluency Score': language_score,
                'Mathematical Integrity Score': math_score,
                'Average Score': avg_score
            })
            identifier_counter += 1
    
    return evaluations

def write_to_csv(evaluations, output_file='cultural_adaptation_results.csv'):
    """Write the extracted evaluations to a CSV file."""
    if not evaluations:
        print("No evaluations found to write to CSV.")
        return
        
    fieldnames = [
        'Identifier', 
        'Original Text', 
        'Adapted Text', 
        'Cultural Relevance Score', 
        'Language Fluency Score', 
        'Mathematical Integrity Score', 
        'Average Score'
    ]
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluations)
    
    print(f"Successfully wrote {len(evaluations)} evaluations to {output_file}")

def main():
    import argparse
    
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Extract cultural adaptation evaluation data to CSV.')
    parser.add_argument('-i', '--input', help='Input file path or directory containing evaluation files')
    parser.add_argument('-o', '--output', default='gemma-2b-cultural_adaptation_results.csv', help='Output CSV file path (default: cultural_adaptation_results.csv)')
    parser.add_argument('-b', '--batch', action='store_true', help='Process input as a directory of files')
    args = parser.parse_args()
    
    if not args.input:
        print("Error: No input specified. Please provide an input file or directory.")
        parser.print_help()
        return
    
    # Process based on whether input is a directory or single file
    if args.batch or os.path.isdir(args.input):
        process_directory(args.input, args.output)
    else:
        # Read from file
        try:
            with open(args.input, 'r', encoding='utf-8') as file:
                input_text = file.read()
        except FileNotFoundError:
            print(f"Error: Input file '{args.input}' not found.")
            return
        except Exception as e:
            print(f"Error reading input file: {e}")
            return
    
        # Process the input text
        evaluations = extract_evaluation_data(input_text)
    
        # Write to specified output file
        write_to_csv(evaluations, args.output)

if __name__ == "__main__":
    main()