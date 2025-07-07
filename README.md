# MOSDAC AI Help Bot - 5-Day MVP Implementation Plan

## Basic Flow Confirmation ✅
**Web Scraping → Knowledge Graph Creation → NLP → RAG Pipeline with LLM → Response**

## MVP Scope
**Core Features Only:**
- Simple web scraping of MOSDAC FAQs and key documentation
- Basic knowledge graph with entities and relationships
- NLP preprocessing for query understanding
- RAG pipeline with vector search
- Simple chat interface for Q&A
- Basic response generation

**Out of Scope for MVP:**
- Advanced geospatial queries
- Multi-turn conversations
- User authentication
- Complex UI features
- Mobile optimization
- Performance optimization

---

## Team Structure & Roles( Note : I am Member A )

### **Member A - Data Pipeline Specialist**
**Focus:** Web Scraping → Knowledge Graph Creation
- Web scraping and data extraction
- Data cleaning and preprocessing
- Knowledge graph construction
- Entity and relationship extraction

### **Member B - ML/NLP Engineer** 
**Focus:** NLP → RAG Pipeline → LLM Integration
- NLP preprocessing pipeline
- Vector embeddings and similarity search
- RAG pipeline implementation
- LLM integration and prompt engineering

### **Member C - Integration & Frontend Developer**
**Focus:** System Integration → User Interface → Response Delivery
- Backend API development
- Simple chat interface
- System integration
- Response formatting and delivery

---

## Day-by-Day MVP Implementation

### **Day 1: Foundation & Data Collection**

#### Member A - Data Pipeline (Web Scraping)
**Primary Tasks:**
- Set up development environment and project structure
- Analyze MOSDAC portal structure (focus on FAQs, documentation)
- Implement web scraper for key pages using BeautifulSoup/Scrapy
- Extract and clean text from FAQs, product documentation, guides

**Deliverables:**
- Working web scraper for MOSDAC content
- Clean text dataset (FAQs, docs, product info)
- Data in structured format (JSON/CSV)

**Key Files to Target:**
- FAQ sections
- Product catalogs
- User guides
- API documentation

#### Member B - NLP Setup
**Primary Tasks:**
- Set up NLP environment (spaCy, NLTK, transformers)
- Research and select LLM (OpenAI API or Hugging Face model)
- Create basic text preprocessing pipeline
- Test embedding generation for sample documents

**Deliverables:**
- NLP preprocessing pipeline
- LLM integration setup
- Text chunking and embedding generation prototype

#### Member C - Backend Foundation
**Primary Tasks:**
- Set up FastAPI backend structure
- Create basic API endpoints structure
- Set up simple chat interface using Streamlit
- Plan system architecture and data flow

**Deliverables:**
- FastAPI backend skeleton
- Basic Streamlit chat interface
- System architecture diagram

---

### **Day 2: Knowledge Graph Creation & Vector Setup**

#### Member A - Knowledge Graph (Core Focus)
**Primary Tasks:**
- Process scraped data to extract entities (products, missions, data types)
- Create relationships between entities using NLP
- Build knowledge graph using NetworkX or simple graph structure
- Implement basic graph querying functions

**Deliverables:**
- Knowledge graph with MOSDAC entities and relationships
- Graph visualization (basic)
- Entity extraction from documents
- Graph query functions

**Key Entities to Extract:**
- Satellite missions (INSAT, IRS, etc.)
- Data products (SST, precipitation, etc.)
- File formats and specifications
- Geographic regions

#### Member B - Vector Database Setup
**Primary Tasks:**
- Create document chunks from scraped content
- Generate embeddings using sentence transformers
- Set up vector database (ChromaDB or FAISS)
- Implement similarity search functionality

**Deliverables:**
- Vector database with document embeddings
- Similarity search implementation
- Document chunking strategy
- Retrieval testing framework

#### Member C - API Development
**Primary Tasks:**
- Create API endpoints for knowledge graph queries
- Develop API endpoints for vector search
- Integrate basic query processing
- Test API endpoints with sample queries

**Deliverables:**
- REST API for graph and vector queries
- Query processing logic
- API testing suite

---

### **Day 3: NLP Pipeline & RAG Implementation**

#### Member A - Entity Enhancement
**Primary Tasks:**
- Improve entity extraction accuracy
- Add entity linking and disambiguation
- Create entity-to-document mappings
- Optimize knowledge graph structure

**Deliverables:**
- Enhanced entity extraction
- Entity-document mapping
- Optimized knowledge graph queries

#### Member B - RAG Pipeline (Core Focus)
**Primary Tasks:**
- Implement complete RAG pipeline
- Create query understanding and intent detection
- Develop context-aware retrieval
- Integrate LLM for response generation

**Deliverables:**
- Working RAG pipeline
- Query preprocessing and understanding
- Context-aware document retrieval
- LLM response generation

**RAG Pipeline Components:**
1. Query preprocessing and intent detection
2. Vector similarity search for relevant documents
3. Knowledge graph query for entities
4. Context compilation and prompt creation
5. LLM response generation

#### Member C - System Integration
**Primary Tasks:**
- Integrate knowledge graph with vector search
- Connect RAG pipeline to API endpoints
- Implement response formatting
- Create unified query processing workflow

**Deliverables:**
- Integrated query processing system
- Response formatting and delivery
- Error handling and fallbacks

---

### **Day 4: LLM Integration & Response Generation**

#### Member A - Data Quality & Optimization
**Primary Tasks:**
- Improve data quality and coverage
- Add more MOSDAC content sources
- Optimize knowledge graph performance
- Create data validation checks

**Deliverables:**
- Expanded and cleaned dataset
- Performance-optimized knowledge graph
- Data quality metrics

#### Member B - LLM Optimization (Core Focus)
**Primary Tasks:**
- Fine-tune prompts for MOSDAC domain
- Implement response quality checks
- Add fallback responses for unclear queries
- Optimize LLM parameters and settings

**Deliverables:**
- Domain-specific prompt templates
- Response quality validation
- Fallback response system
- Optimized LLM configuration

**Prompt Engineering Focus:**
- MOSDAC-specific terminology and context
- Structured response formats
- Handling of technical queries
- Citation and source attribution

#### Member C - Frontend Integration
**Primary Tasks:**
- Complete frontend-backend integration
- Implement real-time chat functionality
- Add loading states and error handling
- Create simple but functional UI

**Deliverables:**
- Complete chat interface
- Real-time response streaming
- Error handling and user feedback
- Working end-to-end system

---

### **Day 5: Testing, Deployment & Demo Preparation**

#### All Members - Collaborative Focus

**Morning Session (Individual Tasks):**

#### Member A - Data Validation
- Final data quality checks
- Knowledge graph validation
- Performance testing for data retrieval

#### Member B - Model Testing  
- End-to-end RAG pipeline testing
- Response quality evaluation
- Performance benchmarking

#### Member C - System Testing
- Full system integration testing
- UI/UX final touches
- Deployment preparation

**Afternoon Session (Collaborative):**

#### Joint Tasks:
- **System Integration Testing**: Test complete pipeline end-to-end
- **Query Testing**: Test with various MOSDAC-related queries
- **Performance Testing**: Measure response times and accuracy
- **Demo Preparation**: Create demonstration scenarios
- **Documentation**: Basic usage documentation
- **Deployment**: Deploy to accessible environment

**Deliverables:**
- Fully functional MVP chatbot
- Test results and performance metrics
- Demo presentation
- Basic documentation
- Deployed system

---

## Technical Stack (MVP)

### Data Processing
- **Python** with requests/BeautifulSoup for scraping
- **Pandas** for data manipulation
- **NetworkX** for simple knowledge graph
- **spaCy** for NLP preprocessing

### ML & AI
- **sentence-transformers** for embeddings
- **ChromaDB** for vector storage
- **OpenAI API** or **Hugging Face transformers** for LLM
- **LangChain** for RAG pipeline orchestration

### Backend & Frontend
- **FastAPI** for backend APIs
- **Streamlit** for quick chat interface
- **uvicorn** for API serving

## Success Criteria for MVP

### Functional Requirements
- ✅ Successfully scrape MOSDAC FAQs and documentation
- ✅ Create basic knowledge graph with key entities
- ✅ Process user queries through NLP pipeline
- ✅ Retrieve relevant information using RAG
- ✅ Generate coherent responses using LLM
- ✅ Provide working chat interface

### Performance Targets
- **Response Time**: < 10 seconds (MVP acceptable)
- **Query Coverage**: Handle 70%+ of FAQ-related queries
- **Response Relevance**: Subjectively relevant responses
- **System Uptime**: Stable for demo period

## Daily Checkpoints

### Daily Standup Format (15 mins):
1. **Yesterday's Progress**: What was completed
2. **Today's Goals**: Specific deliverables
3. **Blockers**: Any issues needing help
4. **Integration Points**: What needs coordination

### Key Integration Moments:
- **End of Day 2**: Test knowledge graph + vector search together
- **End of Day 3**: Test complete RAG pipeline
- **End of Day 4**: Full system integration test

## Sample Test Queries for MVP

### Basic FAQ Queries:
- "How do I download satellite data?"
- "What is SST data?"
- "Which missions provide precipitation data?"
- "How to register for MOSDAC?"

### Knowledge Graph Queries:
- "What satellites are part of INSAT series?"
- "What data products are available for ocean monitoring?"
- "Which missions provide weather data?"

## Risk Mitigation

### High-Risk Areas:
1. **Web Scraping Blocks**: Have backup static data ready
2. **LLM API Limits**: Use local model as fallback
3. **Integration Issues**: Daily integration checks
4. **Time Constraints**: Focus on core pipeline, skip nice-to-have features

### Contingency Plans:
- **Day 3 Checkpoint**: If behind, simplify knowledge graph structure
- **Day 4 Checkpoint**: If issues with LLM, use template-based responses
- **Day 5 Morning**: If major issues, focus on demo-ready subset

## Post-MVP Enhancement Path
1. **Advanced NLP**: Better intent recognition, multi-turn conversations
2. **Improved UI**: React-based interface, better UX
3. **Geospatial Features**: Location-aware queries
4. **Performance**: Caching, optimization, scalability
5. **Modularization**: Make it deployable on other portals

ISRO/
    ├── .env.example             # Example environment variables
    ├── Dockerfile               # Containerization config
    ├── docker-compose.yml       # Multi-service orchestration
    ├── README.md                # Project documentation and plan
    ├── requirements.txt         # Python dependencies
    ├── data/                    # Data storage (see below)
    │   ├── embeddings/          # Likely stores vector embeddings
    │   ├── kg.json              # Saved knowledge graph data
    │   ├── processed/           # Cleaned/processed data
    │   └── raw/                 # Raw scraped data
    ├── llm/                     # LLM-related code
    │   ├── llm_client.py        # Handles LLM API/model calls
    │   └── prompt_templates.py  # Prompt templates for LLM
    ├── notebooks/               # Jupyter notebooks for prototyping/testing
    │   ├── kg_testing.ipynb     # Knowledge graph experiments
    │   └── rag_testing.ipynb    # Retrieval-augmented generation experiments
    ├── src/                     # Core source code
    │   ├── app.py               # Main application (likely Streamlit/FastAPI)
    │   ├── build_kg.py          # Knowledge graph construction
    │   ├── generate_embeddings.py # Embedding generation
    │   ├── ingest.py            # Data ingestion (scraping/cleaning)
    │   └── query.py             # Query processing logic
    └── utils/                   # Utility modules
        ├── kg_utils.py          # Knowledge graph utilities
        └── text_utils.py        # Text/data processing utilities




        # MOSDAC AI Help Bot 🚀

An intelligent Question-Answering system for the [MOSDAC](https://www.mosdac.gov.in) portal that retrieves context-aware answers using semantic search and natural language understanding. This bot is designed to help users query satellite data, understand products, and access information efficiently using local models and document embeddings.

---

## 🔍 Key Features

- **Query Understanding**
  - Preprocessing (cleaning, spell check, abbreviation expansion)
  - Domain-specific tokenization
  - Intent classification (custom-trained using Scikit-learn)
  - Entity extraction using **TF-IDF** from document corpus

- **Document Embedding & Search**
  - Chunked documents embedded using `sentence-transformers` (`all-MiniLM-L6-v2`)
  - Vector database powered by **ChromaDB**
  - Top-K similarity search and reranking with entity-based scoring

---

## 🗂 Project Structure


