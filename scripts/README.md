# MOSDAC AI Help Bot - Scripts

This directory contains utility scripts for the MOSDAC AI Help Bot project.

## Available Scripts

### 1. `add_sample_documents.py`

This script populates the vector store with sample documents for testing purposes.

**Usage:**
```bash
python scripts/add_sample_documents.py
```

**What it does:**
- Initializes a new ChromaDB collection
- Adds sample documents related to MOSDAC satellite data
- Performs a test search to verify the data was added correctly

### 2. `test_api.py`

This script tests the API endpoints to ensure they're working as expected.

**Usage:**
```bash
# Test with default settings (http://localhost:8000)
python scripts/test_api.py

# Test with a custom base URL
python scripts/test_api.py --base-url http://your-api-url:port
```

**What it tests:**
- Health check endpoint (`/health`)
- Search endpoint (`/search`) with various queries
- Filtered search functionality
- Error handling for no results

## Prerequisites

Before running these scripts, make sure you have:

1. Installed all required dependencies:
   ```bash
   pip install -r requirements-api.txt
   ```

2. Downloaded the required spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. Started the API server (for testing):
   ```bash
   python run_api.py
   ```

## Troubleshooting

### Vector Store Issues
If you encounter issues with the vector store:
- Delete the `data/vector_store` directory and let the application recreate it
- Make sure the application has write permissions to the `data` directory

### API Connection Issues
If the test script can't connect to the API:
- Verify the API server is running
- Check if the `--base-url` parameter matches the server's address
- Ensure there are no firewall rules blocking the connection

## Adding More Test Data

To add more test data, edit the `SAMPLE_DOCS` list in `add_sample_documents.py`. Each document should have:
- `text`: The main content of the document
- `metadata`: A dictionary with relevant metadata (e.g., satellite name, year, etc.)

Example:
```python
{
    "text": "Sample document text about satellite data",
    "metadata": {
        "satellite": "Oceansat-2",
        "parameter": "Sea Surface Temperature",
        "year": 2023,
        "source": "MOSDAC"
    }
}
```
