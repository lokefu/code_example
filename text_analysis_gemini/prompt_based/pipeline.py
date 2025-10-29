import pandas as pd
import json
import re
from gemini import generate # Reusing your existing Gemini API handler

def create_prompt(article_text):
    """
    Creates the detailed prompt for the Gemini API, including the tagging guidelines
    and the article text to be processed.
    """
    # Your detailed tagging guidelines are embedded directly into the prompt.
    guidelines = """
    You are an expert news analyst. Your task is to analyze the provided article and generate four types of tags based on the following guidelines.

    **CRITICAL INSTRUCTIONS:**
    1.  You **MUST** output a single, valid JSON object.
    2.  Do **NOT** include any explanatory text, markdown formatting (like ```json), or anything else outside of the main JSON object. Your entire response must be the JSON itself.
    3.  The JSON object must have four keys: "keywords", "tags", "location_keywords", "emotion_keywords". The value for each key must be an array of strings.

    ---
    **Tagging Guidelines**

    **1. Basic Principles:**
    - Format: Nouns and verbs only (nouns preferred), lowercase, connect words with underscores (_), max 3 words and 20 characters per tag. Separate tags with semicolons.
    - Example: monetary_policy; student_loans; announcement

    **2. Tagging Priority:**
    - Specificity: Use specific terms (e.g., `mas_policy` not `government_policy`).
    - Direct Impact: Prioritize direct cause/effect (e.g., `monetary_policy` for an article on an interest rate hike affecting loans).
    - Singapore Focus: Prioritize Singapore-specific issues over global trends when both are present.

    **3. Tag Type Details:**
    - **keywords (Mandatory):** Proper nouns and noun phrases explicitly mentioned.
      - Example: "The Ministry of Education in Singapore announced..." -> ["ministry_of_education_singapore"]
    - **tags (Mandatory):** The essential theme or key argument.
      - Example: "MAS to boost financial innovation by investing in fintech..." -> ["fintech_investment", "financial_innovation"]
    - **location_keywords (Optional):** Geographical names.
      - Example: "...to be built in Singapore’s Marina Bay." -> ["singapore", "marina_bay"]
    - **emotion_keywords (Optional):** Context-specific sentiments.
      - Example: "Concerns over the global recession..." -> ["concern"]

    **4. Tagging Exceptions (Do not tag these):**
    - Vague sentences, subjective/offensive language, promotional content, non-informative text, or article metadata.
    ---

    **Article to Analyze:**
    {article}

    **Your JSON Output:**
    """
    return guidelines.format(article=article_text)

def process_articles(input_csv_path, output_csv_path, article_column_name):
    """
    Reads articles from a CSV, generates tags for each using the Gemini API,
    and saves the results to a new CSV file.

    Args:
        input_csv_path (str): Path to the source CSV file.
        output_csv_path (str): Path to save the tagged CSV file.
        article_column_name (str): The name of the column containing the article text.
    """
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded '{input_csv_path}' with {len(df)} articles.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{input_csv_path}' was not found. Please make sure it exists.")
        return

    # Prepare new columns to store the results
    new_columns = ['keywords', 'tags', 'location_keywords', 'emotion_keywords']
    for col in new_columns:
        df[col] = ''

    # Process each row in the DataFrame
    for index, row in df.iterrows():
        article_text = row[article_column_name]
        
        # Skip empty rows
        if not isinstance(article_text, str) or not article_text.strip():
            print(f"Skipping row {index + 1} due to empty article content.")
            continue

        print(f"Processing article {index + 1}/{len(df)}...")

        try:
            # 1. Create the prompt for the current article
            prompt = create_prompt(article_text)
            
            # 2. Call the Gemini API
            response_text = generate(prompt)
            
            # 3. Clean and parse the JSON response
            # This logic is borrowed from your 'app.py' to handle markdown fences
            match = re.search(r"```json\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
            else:
                json_content = response_text.replace("```json", "").replace("```", "").strip()

            parsed_json = json.loads(json_content)

            # 4. Extract tags and join them into a semicolon-separated string
            keywords = "; ".join(parsed_json.get("keywords", []))
            tags = "; ".join(parsed_json.get("tags", []))
            locations = "; ".join(parsed_json.get("location_keywords", []))
            emotions = "; ".join(parsed_json.get("emotion_keywords", []))

            # 5. Update the DataFrame with the new tags
            df.loc[index, 'keywords'] = keywords
            df.loc[index, 'tags'] = tags
            df.loc[index, 'location_keywords'] = locations
            df.loc[index, 'emotion_keywords'] = emotions

        except json.JSONDecodeError:
            print(f"⚠️ Warning: Failed to decode JSON for article {index + 1}. The model's response might be malformed.")
            df.loc[index, 'Keywords'] = "ERROR: Invalid JSON response"
        except Exception as e:
            print(f"❌ An unexpected error occurred on article {index + 1}: {e}")
            df.loc[index, 'Keywords'] = f"ERROR: {e}"

    # Save the final DataFrame to a new CSV file
    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"\n✅ Success! Tagged articles saved to '{output_csv_path}'.")
    except Exception as e:
        print(f"❌ Error: Could not save the final CSV file. Details: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # Define the input and output file paths and the name of the article column.
    INPUT_CSV = 'test.csv'
    OUTPUT_CSV = 'tagged_articles.csv'
    ARTICLE_COLUMN = 'original_input' # <-- IMPORTANT: Make sure this matches your CSV header

    # --- Execution ---
    process_articles(INPUT_CSV, OUTPUT_CSV, ARTICLE_COLUMN)