import pandas as pd
import json
import os
from pathlib import Path

def parse_excel_file(excel_file):
    """
    Reads the Excel file and converts each sheet into a dictionary.
    """
    # Define the union territories names
    union_territories = {
        "Andaman and Nicobar Islands", "Chandigarh",
        "Dadra and Nagar Haveli and Dama", "Delhi",
        "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
    }
    
    # Read all sheets from the Excel file
    sheets = pd.read_excel(excel_file, sheet_name=None)
    
    data = []
    
    for sheet_name, df in sheets.items():
        # Get the column names from the DataFrame
        columns = df.columns.tolist()
        
        # Create a list to store items from each sheet
        items = []
        
        for _, row in df.iterrows():
            item_data = {}
            for col in columns:
                # Convert the value to string if it's not null, otherwise use empty string
                value = str(row[col]).strip() if pd.notnull(row[col]) else ""
                item_data[col] = value
            items.append(item_data)
        
        # Create a dictionary for the current region
        state_data = {
            "name": sheet_name,
            "type": "Union Territory" if sheet_name in union_territories else "State",
            "items": items
        }
        data.append(state_data)
    
    return data

def write_json_file(data, filename):
    """
    Writes the provided data to a JSON file with pretty-print formatting.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_directory(directory_path):
    """
    Process all Excel files in the given directory.
    """
    # Get all Excel files in the directory
    excel_files = list(Path(directory_path).glob("*.xlsx"))
    
    for excel_file in excel_files:
        try:
            # Parse the Excel file
            json_data = parse_excel_file(excel_file)
            
            # Create JSON filename by replacing .xlsx with .json
            json_filename = excel_file.with_suffix('.json')
            
            # Write the parsed data to a JSON file
            write_json_file(json_data, json_filename)
            
            print(f"Successfully processed {excel_file.name}")
            print(f"Created {json_filename.name}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {excel_file.name}: {str(e)}")
            print("-" * 50)

if __name__ == "__main__":
    # Use the current directory
    current_dir = os.getcwd()
    process_directory(current_dir)
