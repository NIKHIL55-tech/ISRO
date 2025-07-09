# MOSDAC AI Help Bot API

This is the API for the MOSDAC AI Help Bot, providing natural language search capabilities over MOSDAC's knowledge base.

## Features

- **Hybrid Search**: Combines TF-IDF and vector search for better results
- **Intent Classification**: Understands user intent for more relevant responses
- **Query Processing**: Handles domain-specific terminology and acronyms
- **RESTful API**: Easy to integrate with other applications

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ISRO
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements-api.txt
   ```

4. Download the required spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Configuration

Create a `.env` file in the project root with the following variables:

```env
# Server configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1
LOG_LEVEL=info
RELOAD=true

# Vector store configuration
VECTOR_STORE_PATH=./data/vector_store
```

### Running the API

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## API Endpoints

### Search

**POST** `/search`

Search the knowledge base with a natural language query.

**Request Body**:
```json
{
  "query": "sea surface temperature data for Indian Ocean",
  "top_k": 5,
  "filters": {
    "year": 2023,
    "satellite": "Oceansat-2"
  }
}
```

**Response**:
```json
{
  "query": "sea surface temperature data for Indian Ocean",
  "processed_query": "sea surface temperature sst data for indian ocean",
  "intent": "data_availability",
  "confidence": 0.92,
  "results": [
    {
      "text": "Oceansat-2 Level 2 Sea Surface Temperature (SST) data for Indian Ocean region...",
      "score": 0.95,
      "metadata": {
        "satellite": "Oceansat-2",
        "parameter": "SST",
        "year": 2023,
        "source": "MOSDAC"
      },
      "intent": "data_availability",
      "confidence": 0.92
    }
  ],
  "metadata": {
    "intent_explanation": "The query is asking about data availability for sea surface temperature in the Indian Ocean region.",
    "search_metrics": {
      "total_results": 1
    }
  }
}
```

### Health Check

**GET** `/health`

Check if the API is running.

**Response**:
```json
{
  "status": "healthy"
}
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **pylint** for code quality

Run the following commands before committing:

```bash
black .
isort .
pylint src/
```

## License

[Your License Here]

## Contact

[Your Contact Information]
