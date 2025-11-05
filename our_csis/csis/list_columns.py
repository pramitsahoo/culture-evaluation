import pandas as pd
import glob

# Get a list of all Excel files in the current directory
xlsx_files = glob.glob("*.xlsx")

for file in xlsx_files:
    try:
        # Read the first sheet of the Excel file
        df = pd.read_excel(file)
        print(f"File: {file}")
        print("Columns:", df.columns.tolist())
        print("-" * 40)
    except Exception as e:
        print(f"Error reading {file}: {e}")
