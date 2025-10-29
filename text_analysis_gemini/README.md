# Text Analysis Pipeline with Gemini

## 📖 Overview

This project provides a Python script that automates the process of analyzing news articles from a CSV file. It uses the Google Gemini to generate four distinct types of tags for each article: **keywords**, **thematic tags**, **location keywords**, and **emotion keywords**. The final output is a new CSV file containing the original articles alongside their generated tags.

---

## ⚙️ Project Workflow

The pipeline operates in a simple, sequential manner executed by a single script:

1.  **Load Data**: The `pipeline.py` script reads articles from a specified input CSV file (e.g., `test.csv`) using the pandas library. The `batch_processor.py` script do the batching processing from a specified input folder with CSV files inside.
2.  **Create Prompt**: For each article, a detailed prompt is dynamically created. This prompt includes comprehensive tagging guidelines and the article text to be analyzed by the Gemini model.
3.  **Generate Tags**: The script calls the Google Gemini via the `gemini.py` utility module, sending the generated prompt to the model.
4.  **Parse Response**: The script receives the response, which is expected in a JSON format. It cleans and parses this response to extract the four categories of tags.
5.  **Save Output**: The extracted tags are appended as new columns to the original data, and the final result is saved to a new output CSV file (e.g., `tagged_articles.csv`).

---

## 🚀 Getting Started

Follow these instructions to set up and run the project.

### Prerequisites

* Conda package manager
* Python 3.11.13
* Google Cloud SDK installed and configured
* A GCP service account attached in a GCP Project with the Vertex AI API enabled
    * permission needed (based on testing for now): "serviceusage.services.use"

### Installation and Setup

1.  **Download all files:**
    * `pipeline.py`
    * `batch_processor.py`
    * `gemini.py`
    * `requirements.txt`
    * `test.csv` (This serves as an example input file)

2.  **Create and activate a Conda environment:**
    ```bash
    # Create a new conda environment named 'gemini-tagging'
    conda create --name gemini-tagging python=3.11.13
    
    # Activate the environment
    conda activate gemini-tagging
    ```

3.  **Install the required packages:**
    ```bash
    # Install all dependencies from the requirements.txt file
    pip install -r requirements.txt
    ```

4. **Authenticate with Google Cloud: 🔑**
    This command logs you in and sets up Application Default Credentials (ADC), allowing the script to securely access the Gemini API on your behalf.
    ```bash
    gcloud auth application-default login
    ```
    This will open a web browser for you to complete the login process.
    
5. **Set Environment Variables: 🔑**
    This is a critical step. You must tell the script which Google Cloud project to use. You can 
    ```bash
    # On macOS or Linux:
    export GCP_PROJECT_ID='your-gcp-project-id-here'
    # On Windows (Command Prompt):
    set GCP_PROJECT_ID=your-gcp-project-id-here
    # On Windows (PowerShell):
    $env:GCP_PROJECT_ID="your-gcp-project-id-here"
    ```
    The script will fail to run if this environment variable is not set. You can optionally set GCP_LOCATION (Default: global).

6.  **Configure the Pipeline:**
    Before running, open `pipeline.py` and modify the configuration variables at the bottom of the script to match your file names:
    ```python
    if __name__ == '__main__':
        # --- Configuration ---
        INPUT_CSV = 'test.csv'
        OUTPUT_CSV = 'tagged_articles.csv'
        ARTICLE_COLUMN = 'original_input' # IMPORTANT: Must match the CSV header for articles
    ```
    Or for batch process, open `batch_processor.py` and modify the configuration variables at the top of the script to match your folder name:
    ```python
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
    ```


### Running the Application

1.  **Direct Execution:**
    To run the script directly from your terminal, execute the following command in the project directory:
    ```bash
    #python3 pipeline.py
    python3 batch_processor.py
    ```
    You will see the progress printed to the console.

2.  **Running as a Background Process (for large files):**
    For large datasets, it's recommended to run the script in the background and save the output to a log file.
    ```bash
    # Start the process, redirect output to test.log, and disown it
    #nohup python3 pipeline.py > test.log 2>&1 & disown
    nohup python3 batch_processor.py > test.log 2>&1 & disown
    ```
    * **Check the process ID**: `pgrep -f batch_processor.py`
    * **View the live log**: `tail -f test.log`
    * **Check if it's still running**: `ps -p <process_id>`

---

## 📂 File Descriptions

* **`pipeline.py`**: The main script that drives the entire workflow. It reads data, orchestrates API calls, processes results, and saves the output.
* **`batch_processor.py`**: The batch processing script that uses `pipeline.py`.
* **`gemini.py`**: A utility module that handles all communication with the Google Gemini API. It is configured to connect to the specified GCP project and model.
* **`requirements.txt`**: A list of all Python dependencies required for the project, including `pandas`, `google-genai`, and their dependencies.
* **`test.csv`**: An example input file containing articles under the `original_input` column that you can use to test the pipeline.
* **`tagged_articles.csv`**: An example of the output file generated by the script, containing the original articles plus the four new tag columns.
* **`prompt_based`**: The scripts which allow to use prompt to structure output instead of Gemini Schema.