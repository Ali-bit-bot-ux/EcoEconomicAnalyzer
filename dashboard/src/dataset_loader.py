"""
Dataset loader and validator for Drought Mitigation & Soil-Plant Interaction.
Provides indexing, fuzzy query resolution and referential integrity checks.
"""

import json
import os
from typing import Dict, List, Optional, Any

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")


class DroughtDataset:
    def __init__(self, dataset_dir: str = DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.soils: Dict[str, Dict[str, Any]] = {}
        self.crops: Dict[str, Dict[str, Any]] = {}
        self.practices: Dict[str, Dict[str, Any]] = {}
        self.matrix: List[Dict[str, Any]] = []
        self.references: Dict[str, Dict[str, Any]] = {}
        self.load_all()

    def _load_json(self, filename: str) -> Any:
        filepath = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all(self):
        soils_data = self._load_json("soils.json")
        self.soils = {item["id"]: item for item in soils_data}

        crops_data = self._load_json("crops.json")
        self.crops = {item["id"]: item for item in crops_data}

        practices_data = self._load_json("drought_practices.json")
        self.practices = {item["id"]: item for item in practices_data}

        self.matrix = self._load_json("soil_crop_mitigation_matrix.json")

        refs_data = self._load_json("references.json")
        self.references = {item["id"]: item for item in refs_data}

    def validate_integrity(self) -> List[str]:
        """Validates cross-references between matrix, practices, soils and crops."""
        errors = []
        
        # Check matrix references
        for item in self.matrix:
            m_id = item.get("id", "UNKNOWN")
            c_id = item.get("crop_id")
            s_id = item.get("soil_id")

            if c_id not in self.crops:
                errors.append(f"Matrix {m_id}: crop_id '{c_id}' not found in crops.")
            if s_id not in self.soils:
                errors.append(f"Matrix {m_id}: soil_id '{s_id}' not found in soils.")

            for pid in item.get("recommended_practice_ids", []):
                if pid not in self.practices:
                    errors.append(f"Matrix {m_id}: practice '{pid}' not found in practices.")

        # Check practice references to Gale/FAO citations
        for pid, practice in self.practices.items():
            for ref_id in practice.get("gale_reference_ids", []):
                if ref_id not in self.references:
                    errors.append(f"Practice {pid}: reference '{ref_id}' not found in references.")

        return errors

    def search_crop(self, query: str) -> Optional[Dict[str, Any]]:
        """Finds crop by exact id, or fuzzy substring in Russian / Latin name."""
        q = query.strip().lower()
        if not q:
            return None

        # 1. Exact ID
        if q in self.crops:
            return self.crops[q]

        # 2. Exact or substring in name_ru or latin_name
        for crop_id, crop in self.crops.items():
            name_ru = crop.get("name_ru", "").lower()
            latin_name = crop.get("latin_name", "").lower()
            if q == name_ru or q == latin_name:
                return crop
            if q in name_ru or q in latin_name:
                return crop

        # 3. Stemming/root heuristics for common agricultural words in Russian
        aliases = {
            "пшениц": "spring_wheat",
            "пшенич": "spring_wheat",
            "озим": "winter_wheat",
            "ячмен": "spring_barley",
            "овес": "oats",
            "овс": "oats",
            "кукуруз": "maize_corn",
            "прос": "millet",
            "сорго": "sorghum",
            "подсолн": "sunflower",
            "семечк": "sunflower",
            "сафлор": "safflower",
            "рапс": "spring_rapeseed",
            "лен": "flax_linseed",
            "льн": "flax_linseed",
            "нут": "chickpea",
            "чечевиц": "lentil",
            "горох": "pea",
            "люцерн": "alfalfa",
            "хлопок": "cotton",
            "хлопчат": "cotton"
        }
        for root, cid in aliases.items():
            if root in q and cid in self.crops:
                return self.crops[cid]

        return None

    def search_soil(self, query: str) -> Optional[Dict[str, Any]]:
        """Finds soil by exact id or fuzzy substring in Russian / WRB name."""
        q = query.strip().lower()
        if not q:
            return None

        # 1. Exact ID
        if q in self.soils:
            return self.soils[q]

        # 2. Substring in name or wrb
        for soil_id, soil in self.soils.items():
            name_ru = soil.get("name_ru", "").lower()
            wrb = soil.get("wrb_classification", "").lower()
            if q in name_ru or q in wrb:
                return soil

        # 3. Aliases
        soil_aliases = {
            "чернозем": "chernozem_southern_ordinary",
            "каштан": "chestnut_dark_light",
            "серозем": "sierozem_calcareous",
            "солончак": "solonchak_solonetz",
            "солонец": "solonchak_solonetz",
            "песок": "sandy_arenosol",
            "песчан": "sandy_arenosol",
            "супес": "sandy_arenosol",
            "пойм": "alluvial_meadow",
            "аллюви": "alluvial_meadow",
            "пустын": "brown_semi_desert",
            "бурая": "brown_semi_desert"
        }
        for root, sid in soil_aliases.items():
            if root in q and sid in self.soils:
                return self.soils[sid]

        return None

    def get_matrix_entry(self, crop_id: str, soil_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves exact or best-matching rule from the matrix."""
        # 1. Exact match
        for entry in self.matrix:
            if entry.get("crop_id") == crop_id and entry.get("soil_id") == soil_id:
                return entry

        # 2. General soil rule fallback (e.g. any crop on sand or solonetz)
        for entry in self.matrix:
            if entry.get("soil_id") == soil_id:
                return entry

        return None
