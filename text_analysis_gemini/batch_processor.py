import os
import sys

# --- Configuration ---
# 1. Set the name of the folder containing your source CSV files.
SOURCE_FOLDER_NAME = '22 sep analysis'

# 2. Set the column name that contains the text to be analyzed.
# This must match the header in your CSV files.
ARTICLE_COLUMN = 'original_input'
# ---------------------

# --- Script Logic ---
# Dynamically create the output folder name based on the source folder.
OUTPUT_FOLDER_NAME = f"output_{SOURCE_FOLDER_NAME}"

def run_batch_processing():
    """
    Finds all CSV files in the source folder and runs the pipeline on each.
    """
    # Verify that the pipeline.py script exists
    try:
        from pipeline import process_articles
    except ImportError:
        print("❌ Error: 'pipeline.py' not found.")
        print("Please ensure 'pipeline.py' is in the same directory as this script.")
        sys.exit(1)

    # Verify that the source folder exists
    if not os.path.isdir(SOURCE_FOLDER_NAME):
        print(f"❌ Error: Source folder '{SOURCE_FOLDER_NAME}' not found.")
        print("Please create it and add your CSV files.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER_NAME, exist_ok=True)
    print(f"✅ Output will be saved in the '{OUTPUT_FOLDER_NAME}' folder.")

    # Find all CSV files in the source directory
    csv_files = [f for f in os.listdir(SOURCE_FOLDER_NAME) if f.lower().endswith('.csv')]

    if not csv_files:
        print(f"⚠️ Warning: No CSV files were found in '{SOURCE_FOLDER_NAME}'.")
        return

    print(f"Found {len(csv_files)} CSV file(s) to process.\n")

    # Process each CSV file
    for i, filename in enumerate(csv_files):
        input_path = os.path.join(SOURCE_FOLDER_NAME, filename)
        output_path = os.path.join(OUTPUT_FOLDER_NAME, filename)

        print("-" * 60)
        print(f"Processing file {i + 1}/{len(csv_files)}: '{filename}'")
        print("-" * 60)
        
        # Call the main function from your original script
        process_articles(input_path, output_path, ARTICLE_COLUMN)
        
        print(f"\n✅ Finished with '{filename}'. Result saved to '{output_path}'")
        print("-" * 60 + "\n")

    print("🎉 All files have been processed!")


if __name__ == '__main__':
    run_batch_processing()