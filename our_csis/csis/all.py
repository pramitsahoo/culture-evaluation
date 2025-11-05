import pandas as pd
import json
import os
from pathlib import Path
from collections import Counter

def get_feature_name(filename):
    """Convert filename to feature category"""
    feature = filename.stem.lower()
    mapping = {
        'cuisine': 'food',
        'dance_forms': 'dance',
        'languages and dialects': 'languages',
        'states and capitals': 'states_info',
        'traditional games': 'games'
    }
    return mapping.get(feature, feature)

def process_excel_file(excel_file, feature_name):
    """Process a single Excel file and return its items in standardized format"""
    sheets = pd.read_excel(excel_file, sheet_name=None)
    items = []
    
    for state_name, df in sheets.items():
        # Standardize column names
        if len(df.columns) >= 3:
            df.columns = ['name', 'description', 'source'] + list(df.columns[3:])
        
        # Process each row
        for _, row in df.iterrows():
            item = {
                'state': state_name,
                'name': str(row['name']).strip() if pd.notnull(row['name']) else "",
                'description': str(row['description']).strip() if pd.notnull(row['description']) else "",
                'source': str(row['source']).strip() if pd.notnull(row['source']) else ""
            }
            
            # Only add items that have a name
            if item['name'] and item['name'] != 'nan':
                items.append(item)
    
    return items

def generate_statistics(data):
    """Generate detailed statistics about the cultural data"""
    stats = []
    total_items = 0
    
    stats.append("INDIA CULTURAL FEATURES STATISTICS")
    stats.append("=" * 50)
    stats.append("")
    
    # Overall statistics
    for feature, items in data.items():
        total_items += len(items)
        stats.append(f"{feature.upper()}: {len(items)} items")
        
        if items:
            # State-wise distribution
            state_counts = Counter(item['state'] for item in items)
            stats.append("\nState-wise distribution:")
            for state, count in sorted(state_counts.items()):
                stats.append(f"  {state}: {count} items")
            
            # Source statistics
            sources_provided = sum(1 for item in items if item['source'].strip())
            source_percentage = (sources_provided / len(items)) * 100
            stats.append(f"\nSource statistics:")
            stats.append(f"  Items with sources: {sources_provided} ({source_percentage:.1f}%)")
            
            # Description statistics
            desc_provided = sum(1 for item in items if item['description'].strip())
            desc_percentage = (desc_provided / len(items)) * 100
            stats.append(f"  Items with descriptions: {desc_provided} ({desc_percentage:.1f}%)")
        
        stats.append("\n" + "-" * 50 + "\n")
    
    # Overall summary
    stats.append("OVERALL SUMMARY")
    stats.append("=" * 50)
    stats.append(f"Total number of cultural items: {total_items}")
    stats.append(f"Number of feature categories: {len(data)}")
    
    return "\n".join(stats)

def combine_cultural_data(directory_path):
    """Process all Excel files and combine into a single dataset"""
    india_data = {}
    
    excel_files = list(Path(directory_path).glob("*.xlsx"))
    
    for excel_file in excel_files:
        try:
            feature = get_feature_name(excel_file)
            print(f"\nProcessing {excel_file.name} as {feature}...")
            
            items = process_excel_file(excel_file, feature)
            india_data[feature] = items
            
            print(f"Processed {len(items)} items from {excel_file.name}")
            
        except Exception as e:
            print(f"Error processing {excel_file.name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    return india_data

def write_combined_json(data, output_file='india_cultural_features.json'):
    """Write the combined data to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nCreated combined file: {output_file}")

def write_statistics(stats, output_file='cultural_statistics.txt'):
    """Write statistics to a text file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(stats)
    print(f"Created statistics file: {output_file}")

def main():
    current_dir = os.getcwd()
    
    print("Starting to process Excel files...")
    combined_data = combine_cultural_data(current_dir)
    
    # Generate and write statistics
    statistics = generate_statistics(combined_data)
    write_statistics(statistics)
    
    # Write combined JSON
    write_combined_json(combined_data)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()

