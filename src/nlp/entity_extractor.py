import re
import spacy
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import asdict
from ..models.entities import Entity, EntityType, Relationship

class MOSDACEntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict:
        """Initialize regex patterns for entity extraction"""
        return {
            EntityType.SATELLITE: [
                r'\b(INSAT-\d+[A-Z]*|SCATSAT-\d+|OCEANSAT-\d+)\b',
                r'\b(Kalpana-\d+|Megha-Tropiques|SARAL-Altika)\b',
                r'\b(INSAT-3D|INSAT-3DR|INSAT-3DS|SCATSAT-1|OCEANSAT-2|OCEANSAT-3)\b'
            ],
            EntityType.INSTRUMENT: [
                r'\b(VHRR|SCAT|DWR|SAPHIR|MADRAS|SCATTEROMETER|RADIOMETER|IMAGER)\b',
                r'\b(Synthetic Aperture Radar|Microwave Radiometer|Very High Resolution Radiometer|Doppler Weather Radar)\b',
                r'\b(Scatterometer|Altimeter|Ocean Color Monitor|Multispectral Imager)\b'
            ],
            EntityType.PARAMETER: [
                r'\b(SST|NDVI|AOD|Rainfall|Wind Speed|Soil Moisture|Sea Surface Temperature|Aerosol Optical Depth)\b',
                r'\b(Sea Surface Height|Ocean Color|Chlorophyll|Wind Vector|Humidity Profile)\b'
            ],
            EntityType.LOCATION: [
                r'\b(Indian Ocean|Bay of Bengal|Arabian Sea|North Indian Ocean|South Indian Ocean)\b',
                r'\b(Himalayas|Indo-Gangetic Plain|Western Ghats|Deccan Plateau)\b'
            ]
        }
    
    def extract_entities(self, text: str, source_url: str = "") -> List[Entity]:
        """Extract entities from text"""
        doc = self.nlp(text)
        entities = []
        
        # Extract using patterns
        for ent_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = Entity(
                        id=self._generate_entity_id(ent_type, match.group(0)),
                        type=ent_type,
                        name=match.group(0),
                        description="",
                        attributes={
                            "mention": match.group(0),
                            "start_char": match.start(),
                            "end_char": match.end(),
                            "source_text": text[match.start():match.end()],
                            "context": self._get_context(text, match.start(), match.end())
                        },
                        source_urls={source_url} if source_url else set()
                    )
                    entities.append(entity)
        
        # Extract using NER
        for ent in doc.ents:
            ent_type = self._map_ner_to_entity_type(ent.label_)
            if ent_type:
                entity = Entity(
                    id=self._generate_entity_id(ent_type, ent.text),
                    type=ent_type,
                    name=ent.text,
                    description="",
                    attributes={
                        "mention": ent.text,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "source_text": ent.text,
                        "context": self._get_context(text, ent.start_char, ent.end_char),
                        "confidence": 0.9
                    },
                    source_urls={source_url} if source_url else set()
                )
                entities.append(entity)
                
        return self._deduplicate_entities(entities)
    
    def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities in text"""
        relationships = []
        doc = self.nlp(text)
        
        # Simple co-occurrence based relationships within a sentence
        for sent in doc.sents:
            sent_entities = [e for e in entities 
                           if e.attributes['start_char'] >= sent.start_char 
                           and e.attributes['end_char'] <= sent.end_char]
            
            if len(sent_entities) > 1:
                for i in range(len(sent_entities)-1):
                    for j in range(i+1, len(sent_entities)):
                        rel_type = self._infer_relationship(
                            sent_entities[i].type, 
                            sent_entities[j].type,
                            str(sent)
                        )
                        if rel_type:
                            # Add relationship in both directions
                            relationships.extend([
                                Relationship(
                                    source_id=sent_entities[i].id,
                                    target_id=sent_entities[j].id,
                                    type=rel_type,
                                    attributes={
                                        "source": "co-occurrence",
                                        "confidence": 0.7,
                                        "context": str(sent)
                                    }
                                )
                            ])
        
        # Dependency-based relationships
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'attr', 'prep']:
                subj_ent = next((e for e in entities 
                               if e.attributes['start_char'] <= token.idx <= e.attributes['end_char']), None)
                if subj_ent:
                    head_ent = next((e for e in entities 
                                   if e.attributes['start_char'] <= token.head.idx <= e.attributes['end_char']
                                   and e.id != subj_ent.id), None)
                    if head_ent:
                        rel_type = self._get_dependency_relationship(token.dep_, token.text.lower())
                        if rel_type:
                            relationships.append(Relationship(
                                source_id=head_ent.id,
                                target_id=subj_ent.id,
                                type=rel_type,
                                attributes={
                                    "source": "dependency_parse",
                                    "confidence": 0.8,
                                    "dependency": token.dep_,
                                    "token": token.text
                                }
                            ))
        
        return relationships
    
    def _infer_relationship(self, 
                          type1: EntityType, 
                          type2: EntityType,
                          context: str = "") -> Optional[str]:
        """Infer relationship type between two entity types"""
        relationship_map = {
            (EntityType.SATELLITE, EntityType.INSTRUMENT): "carries",
            (EntityType.INSTRUMENT, EntityType.PARAMETER): "measures",
            (EntityType.SATELLITE, EntityType.MISSION): "part_of_mission",
            (EntityType.DATASET, EntityType.PARAMETER): "contains",
            (EntityType.SATELLITE, EntityType.LOCATION): "observes"
        }
        
        # Check direct mapping
        rel = relationship_map.get((type1, type2))
        if rel:
            return rel
            
        # Check reverse mapping
        rel = relationship_map.get((type2, type1))
        if rel:
            return self._get_inverse_relationship(rel)
            
        # Context-based inference
        if "measure" in context.lower() and type1 == EntityType.INSTRUMENT:
            return "measures"
            
        return None
    
    def _get_dependency_relationship(self, dep: str, token: str) -> Optional[str]:
        """Map dependency parse to relationship type"""
        dep_map = {
            'nsubj': 'has_subject',
            'dobj': 'has_object',
            'prep': 'related_to',
            'attr': 'has_attribute'
        }
        return dep_map.get(dep)
    
    def _get_inverse_relationship(self, rel_type: str) -> Optional[str]:
        """Get the inverse of a relationship type"""
        inverse_map = {
            "carries": "carried_by",
            "carried_by": "carries",
            "measures": "measured_by",
            "measured_by": "measures",
            "part_of_mission": "includes",
            "includes": "part_of_mission",
            "contains": "contained_in",
            "contained_in": "contains",
            "observes": "observed_by",
            "observed_by": "observes"
        }
        return inverse_map.get(rel_type)
    
    def _generate_entity_id(self, entity_type: EntityType, name: str) -> str:
        """Generate a unique ID for an entity"""
        base = f"{entity_type.value}:{name.strip().lower()}"
        return re.sub(r'\W+', '_', base)
    
    def _map_ner_to_entity_type(self, ner_label: str) -> Optional[EntityType]:
        """Map spaCy NER labels to our entity types"""
        mapping = {
            "ORG": EntityType.MISSION,
            "PRODUCT": EntityType.INSTRUMENT,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "EVENT": EntityType.MISSION,
            "WORK_OF_ART": EntityType.DATASET
        }
        return mapping.get(ner_label)
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around a span of text"""
        start_ctx = max(0, start - window)
        end_ctx = min(len(text), end + window)
        return text[start_ctx:end_ctx]
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping the most complete version"""
        unique_entities = {}
        for entity in entities:
            if entity.id not in unique_entities:
                unique_entities[entity.id] = entity
            else:
                # Merge attributes and source URLs
                existing = unique_entities[entity.id]
                existing.attributes.update(entity.attributes)
                existing.source_urls.update(entity.source_urls)
                # Keep the longer description
                if len(entity.description) > len(existing.description):
                    existing.description = entity.description
        return list(unique_entities.values())
    
    def process_document(self, text: str, url: str = "") -> Tuple[List[Entity], List[Relationship]]:
        """Convenience method to extract both entities and relationships"""
        entities = self.extract_entities(text, url)
        relationships = self.extract_relationships(text, entities)
        return entities, relationships