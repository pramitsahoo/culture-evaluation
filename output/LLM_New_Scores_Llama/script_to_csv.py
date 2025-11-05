import os
import re
import csv

def extract_data_from_file(file_content):
    """Extract relevant data from file content"""
    data = {}
    
    # Extract the original text
    original_match = re.search(r'Original Text:(.*?)Adapted Text:', file_content, re.DOTALL)
    if original_match:
        data['Original Text'] = original_match.group(1).strip()
    
    # Extract the adapted text
    adapted_match = re.search(r'Adapted Text:(.*?)Evaluation Scores', file_content, re.DOTALL)
    if adapted_match:
        data['Adapted Text'] = adapted_match.group(1).strip()
    
    # Extract Cultural Relevance score
    cultural_match = re.search(r'1\.\s+Cultural Relevance\s+Score:\s+(\d+\.\d+)/5', file_content)
    if cultural_match:
        data['Cultural Relevance Score'] = cultural_match.group(1)
    
    # Extract Language Fluency score
    language_match = re.search(r'2\.\s+Language Fluency\s+Score:\s+(\d+\.\d+)/5', file_content)
    if language_match:
        data['Language Fluency Score'] = language_match.group(1)
    
    # Extract Mathematical Integrity score
    math_match = re.search(r'3\.\s+Mathematical Integrity\s+Score:\s+(\d+\.\d+)/5', file_content)
    if math_match:
        data['Mathematical Integrity Score'] = math_match.group(1)
    
    # Extract Average Score
    avg_match = re.search(r'Average Score:\s+(\d+\.\d+)/5', file_content)
    if avg_match:
        data['Average Score'] = avg_match.group(1)
    
    return data

def process_directory(directory_path, output_file):
    """Process all files in the directory and write results to CSV"""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Identifier', 'Original Text', 'Adapted Text', 
                     'Cultural Relevance Score', 'Language Fluency Score', 
                     'Mathematical Integrity Score', 'Average Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        file_count = 0
        for filename in os.listdir(directory_path):
            if not filename.endswith('.txt'):
                continue
                
            file_path = os.path.join(directory_path, filename)
            file_count += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                data = extract_data_from_file(content)
                
                # Create identifier from filename or use a counter
                identifier = f"eval_{file_count}"
                
                row = {
                    'Identifier': identifier,
                    'Original Text': data.get('Original Text', ''),
                    'Adapted Text': data.get('Adapted Text', ''),
                    'Cultural Relevance Score': data.get('Cultural Relevance Score', ''),
                    'Language Fluency Score': data.get('Language Fluency Score', ''),
                    'Mathematical Integrity Score': data.get('Mathematical Integrity Score', ''),
                    'Average Score': data.get('Average Score', '')
                }
                
                writer.writerow(row)
                
                if file_count % 100 == 0:
                    print(f"Processed {file_count} files...")
                    
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        
        print(f"Completed! Processed {file_count} files.")

directory_path = "/u/student/2023/ai23mtech14004/culture-evaluation/output/LLM_New_Scores_Llama/llama-3/explanations"
output_file = "/u/student/2023/ai23mtech14004/culture-evaluation/output/LLM_New_Scores_Llama/llama-3.csv"
process_directory(directory_path, output_file)