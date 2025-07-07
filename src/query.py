# src/query.py
import re
import numpy as np
from typing import Dict, List
import spacy
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import logging
from datetime import datetime

# NLP modules
from src.nlp.query_preprocessor import QueryPreprocessor
from src.nlp.intent_classifier import MOSDACIntentClassifier
from src.nlp.entity_extractor import MOSDACEntityExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        try:
            logger.info("Initializing QueryProcessor...")

            self.preprocessor = QueryPreprocessor()
            self.intent_classifier = MOSDACIntentClassifier()
            self.intent_classifier.train_classifier([
                ("how to download sst data", "data_access"),
                ("what is insat satellite", "mission_info"),
                ("compare insat and irs", "comparison"),
                ("explain precipitation data", "product_details"),
                ("how to use netcdf files", "technical_support")
            ])
            self.entity_extractor = MOSDACEntityExtractor()
            self.nlp = spacy.load("en_core_web_sm")
            self.embedder = SentenceTransformer(embedding_model)

            # Connect to ChromaDB and get documents
            self.chroma_client = PersistentClient(path="data/embeddings")
            self.collection = self.chroma_client.get_or_create_collection(
                name="mosdac_docs",
                metadata={"description": "MOSDAC document collection"}
            )

            # Build TF-IDF entity vocabulary from all documents
            try:
                all_docs = self.collection.get(include=["documents"])["documents"]
                if all_docs:
                 self.entity_extractor.build_entity_vocab(all_docs)
                 logger.info("TF-IDF vocabulary built from ChromaDB corpus.")
            except Exception as e:
                logger.warning(f"TF-IDF vocab build skipped: {e}")

            logger.info("QueryProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    def process_query(self, query: str, top_k: int = 5) -> Dict:
        try:
            logger.info(f"Processing query: {query}")

            cleaned_query = self.preprocessor.clean_query(query)
            spell_checked = self.preprocessor.spell_check_technical(cleaned_query)
            tokens = self.preprocessor.tokenize_domain_specific(spell_checked)

            doc = self.nlp(spell_checked)
            intent_result = self.intent_classifier.classify_intent(spell_checked)

            # ✅ Use TF-IDF based entity extraction (built from ChromaDB)
            entities = self.entity_extractor.extract_entities(spell_checked)

            # Generate embedding
            query_embedding = np.array(self.embedder.encode([spell_checked])).tolist()

            # Search ChromaDB
            search_results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            results = []
            if search_results["documents"] and search_results["metadatas"] and search_results["distances"]:
                for i in range(len(search_results["documents"][0])):
                    results.append({
                        "text": search_results["documents"][0][i],
                        "metadata": search_results["metadatas"][0][i],
                        "score": 1 - search_results["distances"][0][i]
                    })

            return {
                "query_info": {
                    "original": query,
                    "cleaned": spell_checked,
                    "intent": intent_result.get("intent", "unknown"),
                    "confidence": intent_result.get("confidence", 0.0),
                    "entities": entities,
                    "timestamp": datetime.now().isoformat()
                },
                "results": results
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query_info": {"error": str(e)},
                "results": []
            }

if __name__ == "__main__":
    qp = QueryProcessor()
    queries = [
        "How do I download SST data from MOSDAC?",
        "What is the resolution of INSAT-3D images?",
        "Compare Oceansat-3 and Scatsat-1.",
        "Rainfall data for Chennai last month"
    ]
    for q in queries:
        response = qp.process_query(q)
        print("\nQuery:", q)
        if "intent" in response['query_info']:
            print("Intent:", response['query_info']['intent'])
            print("Confidence:", response['query_info'].get('confidence', 0.0))
        else:
            print("Intent detection failed")
        print("Entities:", response['query_info'].get('entities', {}))
        if response['results']:
            print("Top result:", response['results'][0]['text'][:150] + "...")
        else:
            print("No results.")
