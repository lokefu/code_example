import pandas as pd
import json
import re
from gemini import generate  # Reusing your existing Gemini API handler
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional

# --- Custom Type with Built-in Validation ---
# This class defines the strict formatting rules for a SINGLE valid tag.
class TagName(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core.core_schema import str_schema, with_info_before_validator_function

        return with_info_before_validator_function(
            cls.validate, 
            str_schema(max_length=30),
        )

    @classmethod
    def validate(cls, v: str, info) -> str:
        """Validates a single tag against our formatting rules."""
        if not isinstance(v, str):
            raise ValueError("Tag must be a string.")
        # --- THIS IS THE FIX ---
        # Add the length check directly into this validator.
        if len(v) > 30:
            raise ValueError("Tag cannot be longer than 20 characters.")
            
        if not v.islower():
            raise ValueError("Tag must be lowercase.")
        if ' ' in v:
            raise ValueError("Tag must use underscores (_) instead of spaces.")
        if not re.fullmatch(r'[a-z_-]+', v):
            raise ValueError("Tag can only contain lowercase letters and underscores.")
        if len(v.split('_')) > 3:
            raise ValueError("Tag cannot have more than 3 words.")
        return v

# --- Pydantic Schema Definition with Filtering Logic ---
class ArticleTags(BaseModel):
    """Defines the structured output for article analysis."""
    keywords: List[TagName] = Field(description="Proper nouns and noun phrases explicitly mentioned in the article. Example: 'The Ministry of Education in Singapore announced...' -> ['ministry_of_education_singapore']")
    tags: List[TagName] = Field(description="The essential theme or key argument of the article. Example: 'MAS to boost financial innovation by investing in fintech...' -> ['fintech_investment', 'financial_innovation']")
    location_keywords: Optional[List[TagName]] = Field(default=[], description="Geographical names mentioned. Example: '...to be built in Singapore’s Marina Bay.' -> ['singapore', 'marina_bay']")
    emotion_keywords: Optional[List[TagName]] = Field(default=[], description="Context-specific sentiments. Example: 'Concerns over the global recession...' -> ['concern']")

    # --- THIS LOGIC FILTERS INVALID TAGS INSTEAD OF RAISING AN ERROR ---
    @field_validator('keywords', 'tags', 'location_keywords', 'emotion_keywords', mode='before')
    @classmethod
    def filter_invalid_tags(cls, values: List[str]) -> List[str]:
        """
        This validator runs BEFORE Pydantic validates the list items.
        It iterates through the incoming list and filters out any invalid tags.
        """
        if not isinstance(values, list):
            return [] # Return empty list if the API provides a non-list type
            
        valid_tags = []
        for tag in values:
            try:
                # We try to validate each tag using our TagName logic
                TagName.validate(tag, None)
                valid_tags.append(tag)
            except (ValueError, TypeError):
                # If validation fails for a single tag, we simply skip it
                print(f"  ...Skipping invalid tag: '{tag}'")
                pass # Continue to the next tag in the list
        return valid_tags


def create_prompt_with_guidelines(article_text):
    """
    Creates the detailed prompt for the Gemini API.
    The prompt now focuses on the higher-level conceptual guidelines.
    """
    guidelines = """
    You are an expert news analyst. Your task is to analyze the provided article and generate tags based on the following guidelines.
    Your output MUST be a valid JSON object that strictly follows the provided schema.
    For all tags, ensure they are lowercase and use underscores to connect words (e.g., 'financial_innovation').

    ---
    **Conceptual Tagging Guidelines**

    1.  **Tagging Priority:**
        -   Specificity: Use specific terms (e.g., `mas_policy` not `government_policy`).
        -   Direct Impact: Prioritize direct cause/effect (e.g., `monetary_policy` for an article on an interest rate hike).
        -   Singapore Focus: Prioritize Singapore-specific issues over global trends when both are present.

    2.  **Tagging Exceptions (Do not tag these):**
        -   Vague sentences, subjective/offensive language, promotional content, non-informative text, or article metadata.
    ---

    **Article to Analyze:**
    {article}
    """
    return guidelines.format(article=article_text)

def process_articles(input_csv_path, output_csv_path, article_column_name):
    """
    Reads articles from a CSV, generates tags for each using the Gemini API with structured output,
    and saves the results to a new CSV file.
    """
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded '{input_csv_path}' with {len(df)} articles.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{input_csv_path}' was not found. Please make sure it exists.")
        return

    new_columns = ['keywords', 'tags', 'location_keywords', 'emotion_keywords']
    for col in new_columns:
        df[col] = ''

    for index, row in df.iterrows():
        article_text = row[article_column_name]
        
        if not isinstance(article_text, str) or not article_text.strip():
            print(f"Skipping row {index + 1} due to empty article content.")
            continue

        print(f"Processing article {index + 1}/{len(df)}...")

        try:
            prompt = create_prompt_with_guidelines(article_text)
            
            # Correctly calling the generate function with generation_config
            response_text = generate(prompt, ArticleTags)
            
            json_data = json.loads(response_text)
            
            # The filter_invalid_tags method now runs automatically, skipping bad tags
            parsed_data = ArticleTags(**json_data)

            df.loc[index, 'keywords'] = "; ".join(parsed_data.keywords)
            df.loc[index, 'tags'] = "; ".join(parsed_data.tags)
            df.loc[index, 'location_keywords'] = "; ".join(parsed_data.location_keywords or [])
            df.loc[index, 'emotion_keywords'] = "; ".join(parsed_data.emotion_keywords or [])

        except (json.JSONDecodeError, ValidationError) as e:
            # This will now only catch errors if the entire JSON is malformed
            print(f"⚠️ Warning: Failed to parse or validate the response for article {index + 1}. Error: {e}")
            for col in new_columns:
                df.loc[index, col] = "PARSING_OR_VALIDATION_ERROR"
        except Exception as e:
            print(f"❌ An unexpected error occurred on article {index + 1}: {e}")
            for col in new_columns:
                df.loc[index, col] = "PROCESSING_ERROR"

    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"\n✅ Success! Tagged articles saved to '{output_csv_path}'.")
    except Exception as e:
        print(f"❌ Error: Could not save the final CSV file. Details: {e}")

if __name__ == '__main__':
    INPUT_CSV = 'test2.csv'
    OUTPUT_CSV = 'test2_output_test.csv'
    ARTICLE_COLUMN = 'original_input'

    process_articles(INPUT_CSV, OUTPUT_CSV, ARTICLE_COLUMN)

