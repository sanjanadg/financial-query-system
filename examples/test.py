import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import process_file, excel_query

# Process an Excel file
representation = process_file("data/Consolidated Plan 2023-2024.xlsm")
print(f"Processed {len(representation)} sheets")

# Query the data
answer = excel_query("What percent of MXDs costs are indirect? Which month had the highest percentage?", representation)
print(answer)

