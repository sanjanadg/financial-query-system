from main import process_file, excel_query

# Process an Excel file
representation = process_file("data/Consolidated Plan 2023-2024.xlsm")
print(f"Processed {len(representation)} sheets")

# Query the data
answer = excel_query("What is MXD's Gross Profit in Jan 2022?", representation)
print(answer)

