from google import genai
from google.genai import types
import os

def generate(input, ArticleTags):
    # Configure the API key from environment variables
    gcp_project_id = os.environ.get("GCP_PROJECT_ID")
    if not gcp_project_id:
        raise ValueError("Error: The 'GCP_PROJECT_ID' environment variable is not set.")
    gcp_location = os.getenv("GCP_LOCATION", "global") # Default location if not set

    client = genai.Client(
        vertexai=True,
        project=gcp_project_id,
        location=gcp_location,
        #project="gcp-r-d-sg-nonprod-3-svc-buax",
        #location="global",
    )

    model = "gemini-2.5-pro" # Note: Changed to a common model name
    contents = [
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=f"""{input}""")
        ]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 0.95,
        max_output_tokens = 8192, # Adjusted to a typical max
        safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_NONE" # Use standard threshold names
        ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_NONE"
        ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_NONE"
        ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_NONE",
        )],
        response_mime_type="application/json",  # Uncomment if supported by your library version
        response_schema= ArticleTags,
        # thinking_config is not a standard parameter in GenerateContentConfig for this library version
    )

    # --- Key Change is Here ---
    # Use generate_content instead of generate_content_stream
    response = client.models.generate_content(
        model = model,
        contents = contents,
        config = generate_content_config,
    )

    # The full response text is now in the 'text' attribute
    full_response_text = response.text
    
    # Now you can use the variable

    return full_response_text

