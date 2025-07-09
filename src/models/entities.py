from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

class EntityType(str, Enum):
    SATELLITE = "satellite"
    INSTRUMENT = "instrument"
    PARAMETER = "parameter"
    DATASET = "dataset"
    LOCATION = "location"
    MISSION = "mission"

@dataclass
class Entity:
    id: str
    type: EntityType
    name: str
    description: str = ""
    attributes: Dict = field(default_factory=dict)
    source_urls: Set[str] = field(default_factory=set)

@dataclass
class Relationship:
    source_id: str
    target_id: str
    type: str
    attributes: Dict = field(default_factory=dict)
