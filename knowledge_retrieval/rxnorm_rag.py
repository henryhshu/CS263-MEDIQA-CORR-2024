#!/usr/bin/env python
"""
RxNorm RAG (Retrieval-Augmented Generation) Module

This module provides functionality to:
1. Extract drug/medication names from medical text
2. Query the RxNorm API to retrieve detailed drug information
3. Augment prompts with retrieved drug context before LLM inference

RxNorm API Documentation: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
"""

import re
import json
import time
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from functools import lru_cache

import requests
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

# Common drug name patterns (brand names are title case, generics are lowercase)
# This regex matches potential drug names in text
DRUG_NAME_PATTERN = re.compile(
    r'\b([A-Z][a-z]+(?:[-][A-Z][a-z]+)?)\b'  # Brand names (e.g., Cymbalta, Advil-PM)
    r'|'
    r'\b([a-z]+(?:mycin|cillin|pril|olol|statin|prazole|tidine|afil|sartan|dipine|azole|cycline|floxacin|vir|mab|nib|zole|ine|ate|ide)\b)'  # Generic patterns
)

# Common medication suffixes that indicate drug names
DRUG_SUFFIXES = {
    'mycin', 'cillin', 'pril', 'olol', 'statin', 'prazole', 'tidine',
    'afil', 'sartan', 'dipine', 'azole', 'cycline', 'floxacin', 'vir',
    'mab', 'nib', 'zole', 'pine', 'done', 'lam', 'pam', 'ine', 'ide'
}

# Words to exclude (common medical terms, lab values, organisms, anatomy - NOT drugs)
EXCLUDE_WORDS = {
    # Common words
    'patient', 'doctor', 'hospital', 'diagnosis', 'examination', 'treatment',
    'symptoms', 'syndrome', 'disease', 'condition', 'history', 'physical',
    'examination', 'laboratory', 'imaging', 'procedure', 'surgery', 'therapy',
    'the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 'been',
    'she', 'her', 'his', 'him', 'they', 'their', 'was', 'were', 'are', 'has',
    'also', 'given', 'taking', 'prescribed', 'daily', 'twice', 'once',
    'normal', 'abnormal', 'positive', 'negative', 'elevated', 'decreased',
    'blood', 'urine', 'serum', 'plasma', 'tissue', 'cell', 'infection',
    'inflammation', 'pain', 'fever', 'nausea', 'vomiting', 'diarrhea',
    'headache', 'headaches', 'occasionally', 'management', 'cholesterol',
    'hypertension', 'diabetes', 'medicine', 'medication', 'drug', 'drugs',
    # Lab values and cells
    'hemoglobin', 'hematocrit', 'platelet', 'platelets', 'leucocyte', 'leucocytes',
    'leukocyte', 'leukocytes', 'neutrophil', 'neutrophils', 'lymphocyte', 'lymphocytes',
    'monocyte', 'monocytes', 'eosinophil', 'eosinophils', 'basophil', 'basophils',
    'erythrocyte', 'erythrocytes', 'reticulocyte', 'reticulocytes',
    'albumin', 'globulin', 'bilirubin', 'creatinine', 'glucose', 'protein',
    # Organisms (not drugs)
    'mycobacterium', 'streptococcus', 'staphylococcus', 'escherichia', 'candida',
    'pseudomonas', 'enterococcus', 'klebsiella', 'salmonella', 'shigella',
    'clostridium', 'bacillus', 'listeria', 'neisseria', 'haemophilus',
    'hepatitis', 'tuberculosis', 'malaria', 'influenza', 'coronavirus',
    # Anatomy and body parts  
    'chest', 'lung', 'lungs', 'heart', 'liver', 'kidney', 'kidneys', 'spleen',
    'bone', 'bones', 'muscle', 'muscles', 'nerve', 'nerves', 'brain', 'spine',
    'skin', 'eye', 'eyes', 'ear', 'ears', 'nose', 'throat', 'mouth',
    # Medical conditions (not drugs)
    'anemia', 'arthritis', 'asthma', 'bronchitis', 'cancer', 'carcinoma',
    'cirrhosis', 'colitis', 'dermatitis', 'eczema', 'embolism', 'fibrosis',
    'gastritis', 'glaucoma', 'gout', 'hernia', 'hepatitis', 'hyperthyroid',
    'hypothyroid', 'ischemia', 'leukemia', 'lymphoma', 'melanoma', 'nephritis',
    'neuropathy', 'osteoporosis', 'pancreatitis', 'pneumonia', 'psoriasis',
    'sarcoma', 'sclerosis', 'sepsis', 'stenosis', 'stroke', 'thrombosis',
    # Common adjectives and verbs
    'limited', 'complete', 'partial', 'total', 'acute', 'chronic', 'severe',
    'mild', 'moderate', 'progressive', 'stable', 'improving', 'worsening',
    'past', 'present', 'future', 'chart', 'note', 'notes', 'report', 'reports',
    # Immune and antibody terms
    'antibody', 'antibodies', 'antigen', 'antigens', 'immunoglobulin',
    'complement', 'cytokine', 'cytokines', 'interleukin', 'interferon',
    # Other medical terms
    'temperature', 'pulse', 'pressure', 'rate', 'rhythm', 'reflex', 'response',
    'level', 'levels', 'count', 'counts', 'ratio', 'index', 'score', 'grade',
    # Ethnicities and nationalities (sometimes extracted)
    'american', 'african', 'asian', 'european', 'hispanic', 'latino', 'latina',
    'cuban', 'mexican', 'chinese', 'japanese', 'indian', 'korean',
    'cuban-american', 'african-american', 'asian-american',
    # Anti- prefixed terms that aren't complete drug names
    'anti', 'ante', 'pre', 'post', 'sub', 'super', 'hyper', 'hypo',
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DrugInfo:
    """Represents information about a drug from RxNorm."""
    rxcui: str
    name: str
    term_type: str  # e.g., 'IN' (ingredient), 'BN' (brand name), 'SCD' (clinical drug)
    synonym: Optional[str] = None
    ingredients: List[str] = field(default_factory=list)
    dose_forms: List[str] = field(default_factory=list)
    related_drugs: List[str] = field(default_factory=list)
    drug_class: Optional[str] = None
    
    def to_context_string(self) -> str:
        """Convert drug info to a string suitable for LLM context."""
        parts = [f"Drug: {self.name} (RxCUI: {self.rxcui}, Type: {self.term_type})"]
        
        if self.synonym:
            parts.append(f"  Synonym: {self.synonym}")
        if self.ingredients:
            parts.append(f"  Ingredients: {', '.join(self.ingredients)}")
        if self.dose_forms:
            parts.append(f"  Dose Forms: {', '.join(self.dose_forms[:5])}")  # Limit to 5
        if self.related_drugs:
            parts.append(f"  Related: {', '.join(self.related_drugs[:5])}")  # Limit to 5
        if self.drug_class:
            parts.append(f"  Drug Class: {self.drug_class}")
            
        return '\n'.join(parts)


@dataclass  
class RetrievalResult:
    """Result of drug information retrieval."""
    query_term: str
    found: bool
    drug_info: Optional[DrugInfo] = None
    error: Optional[str] = None
    
    
# =============================================================================
# RxNorm API Client
# =============================================================================

class RxNormClient:
    """Client for interacting with the RxNorm REST API."""
    
    def __init__(self, base_url: str = RXNORM_BASE_URL, rate_limit_delay: float = 0.1):
        """
        Initialize the RxNorm client.
        
        Args:
            base_url: Base URL for the RxNorm API
            rate_limit_delay: Delay between API calls to respect rate limits
        """
        self.base_url = base_url.rstrip('/')
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self._last_request_time = 0.0
        
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
        
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a GET request to the RxNorm API.
        
        Args:
            endpoint: API endpoint (e.g., '/drugs')
            params: Query parameters
            
        Returns:
            JSON response as dict, or None if request failed
        """
        self._rate_limit()
        url = f"{self.base_url}{endpoint}.json"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"RxNorm API request failed for {endpoint}: {e}")
            return None
            
    @lru_cache(maxsize=500)
    def get_drugs(self, name: str) -> Optional[Dict]:
        """
        Get drug products associated with a specified name.
        
        Args:
            name: Drug name (ingredient, brand name, etc.)
            
        Returns:
            Drug group data from RxNorm API
        """
        return self._get('/drugs', {'name': name})
    
    @lru_cache(maxsize=500)
    def find_rxcui_by_string(self, name: str, search_type: int = 2) -> Optional[str]:
        """
        Find RxCUI by drug name string.
        
        Args:
            name: Drug name to search
            search_type: 0=exact, 1=normalized, 2=best match
            
        Returns:
            RxCUI string or None
        """
        result = self._get('/rxcui', {'name': name, 'search': search_type})
        if result and 'idGroup' in result:
            rxnorm_ids = result['idGroup'].get('rxnormId', [])
            if rxnorm_ids:
                return rxnorm_ids[0]
        return None
    
    @lru_cache(maxsize=500)
    def get_rxcui_properties(self, rxcui: str) -> Optional[Dict]:
        """
        Get properties for an RxCUI.
        
        Args:
            rxcui: RxNorm Concept Unique Identifier
            
        Returns:
            Concept properties
        """
        return self._get(f'/rxcui/{rxcui}/properties')
    
    @lru_cache(maxsize=500)
    def get_all_related(self, rxcui: str) -> Optional[Dict]:
        """
        Get all related concepts for an RxCUI.
        
        Args:
            rxcui: RxNorm Concept Unique Identifier
            
        Returns:
            Related concept groups
        """
        return self._get(f'/rxcui/{rxcui}/allrelated')
    
    @lru_cache(maxsize=500)
    def get_spelling_suggestions(self, name: str) -> List[str]:
        """
        Get spelling suggestions for a drug name.
        
        Args:
            name: Potentially misspelled drug name
            
        Returns:
            List of spelling suggestions
        """
        result = self._get('/spellingsuggestions', {'name': name})
        if result and 'suggestionGroup' in result:
            return result['suggestionGroup'].get('suggestionList', {}).get('suggestion', [])
        return []
    
    @lru_cache(maxsize=500)
    def get_approximate_match(self, term: str, max_entries: int = 5) -> List[Dict]:
        """
        Get approximate matches for a term.
        
        Args:
            term: Search term
            max_entries: Maximum number of results
            
        Returns:
            List of approximate match candidates
        """
        result = self._get('/approximateTerm', {'term': term, 'maxEntries': max_entries})
        if result and 'approximateGroup' in result:
            candidates = result['approximateGroup'].get('candidate', [])
            return candidates if isinstance(candidates, list) else [candidates]
        return []


# =============================================================================
# Drug Name Extractor
# =============================================================================

class DrugNameExtractor:
    """Extracts potential drug names from medical text."""
    
    def __init__(self, rxnorm_client: Optional[RxNormClient] = None):
        """
        Initialize the extractor.
        
        Args:
            rxnorm_client: Optional RxNorm client for validation
        """
        self.rxnorm_client = rxnorm_client or RxNormClient()
        
        # Compile patterns for different drug name formats
        self.patterns = [
            # Brand names (capitalized)
            re.compile(r'\b([A-Z][a-z]+(?:[-][A-Z][a-z]+)?)\b'),
            # Generic drugs with common suffixes
            re.compile(r'\b([a-z]+(?:' + '|'.join(DRUG_SUFFIXES) + r'))\b', re.IGNORECASE),
            # Drugs mentioned with dosage (e.g., "metformin 500 mg")
            re.compile(r'\b([a-zA-Z]+)\s+\d+\s*(?:mg|mcg|g|ml|mL|units?)\b', re.IGNORECASE),
        ]
        
    def extract_candidates(self, text: str) -> Set[str]:
        """
        Extract potential drug name candidates from text.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            Set of candidate drug names
        """
        candidates = set()
        
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                # Get the captured group (first non-None group)
                for group in match.groups():
                    if group:
                        word = group.strip()
                        if self._is_valid_candidate(word):
                            candidates.add(word.lower())
                        break
                        
        return candidates
    
    def _is_valid_candidate(self, word: str) -> bool:
        """Check if a word is a valid drug name candidate."""
        word_lower = word.lower()
        
        # Must be at least 3 characters
        if len(word) < 3:
            return False
            
        # Exclude common non-drug words
        if word_lower in EXCLUDE_WORDS:
            return False
            
        # Should contain at least one vowel
        if not any(c in word_lower for c in 'aeiou'):
            return False
            
        return True
    
    def extract_and_validate(self, text: str, validate: bool = True) -> List[str]:
        """
        Extract drug names and optionally validate against RxNorm.
        
        Args:
            text: Medical text to analyze
            validate: Whether to validate candidates against RxNorm
            
        Returns:
            List of validated drug names
        """
        candidates = self.extract_candidates(text)
        
        if not validate:
            return list(candidates)
            
        validated = []
        for candidate in candidates:
            rxcui = self.rxnorm_client.find_rxcui_by_string(candidate)
            if rxcui:
                validated.append(candidate)
            else:
                # Try approximate matching
                matches = self.rxnorm_client.get_approximate_match(candidate, max_entries=1)
                if matches and float(matches[0].get('score', 0)) > 0:
                    validated.append(candidate)
                    
        return validated


# =============================================================================
# PubMedBERT NER Drug Extractor
# =============================================================================

# Default PubMedBERT model for chemical/drug NER
DEFAULT_NER_MODEL = "pruas/BENT-PubMedBERT-NER-Chemical"


class PubMedBERTDrugExtractor:
    """
    Extracts drug/chemical names from medical text using a PubMedBERT model
    fine-tuned for biomedical NER.
    
    Uses the pruas/BENT-PubMedBERT-NER-Chemical model by default, which was
    trained on 15 biomedical NER datasets including BC5CDR, DDI corpus,
    CHEMDNER, NLM-CHEM, and more.
    """
    
    def __init__(
        self,
        rxnorm_client: Optional['RxNormClient'] = None,
        model_name: str = DEFAULT_NER_MODEL,
        confidence_threshold: float = 0.5,
        device: Optional[int] = None,
    ):
        """
        Initialize the PubMedBERT drug extractor.
        
        Args:
            rxnorm_client: Optional RxNorm client for validation
            model_name: HuggingFace model ID for the NER model
            confidence_threshold: Minimum confidence score for entity extraction
            device: Device to run inference on (-1 for CPU, 0+ for GPU, None for auto)
        """
        self.rxnorm_client = rxnorm_client or RxNormClient()
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 0
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = -1  # CPU
        
        logger.info(f"Loading PubMedBERT NER model: {model_name}")
        logger.info(f"Using device: {device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model, 
            tokenizer=self.tokenizer,
            aggregation_strategy="first",
            device=device,
        )
        
        logger.info("PubMedBERT NER model loaded successfully")

    def extract_candidates(self, text: str) -> Set[str]:
        """
        Extract potential drug name candidates from text using PubMedBERT NER.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            Set of candidate drug names
        """
        candidates = set()
        
        # Split long texts into chunks to handle model max length (512 tokens)
        chunks = self._split_text(text, max_length=400)
        
        for chunk in chunks:
            try:
                entities = self.ner_pipeline(chunk)
            except Exception as e:
                logger.warning(f"NER inference failed on chunk: {e}")
                continue
            
            for entity in entities:
                # Accept any entity group (the model labels chemicals/drugs)
                score = entity.get("score", 0)
                word = entity.get("word", "").strip()
                
                if score >= self.confidence_threshold and self._is_valid_entity(word):
                    # Clean up tokenizer artifacts
                    cleaned = self._clean_entity(word)
                    if cleaned:
                        candidates.add(cleaned.lower())
        
        return candidates
    
    def _split_text(self, text: str, max_length: int = 400) -> List[str]:
        """
        Split text into chunks that fit within the model's token limit.
        Splits on sentence boundaries when possible.
        
        Args:
            text: Text to split
            max_length: Approximate maximum number of tokens per chunk
            
        Returns:
            List of text chunks
        """
        # Rough estimate: 1 token ≈ 4 characters for medical text
        max_chars = max_length * 4
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _is_valid_entity(self, word: str) -> bool:
        """
        Check if an extracted entity is a valid drug candidate.
        
        Args:
            word: Extracted entity text
            
        Returns:
            True if the entity is a valid drug name candidate
        """
        if not word or len(word) < 2:
            return False
        
        # Filter out purely numeric entities
        if word.replace('.', '').replace('-', '').isdigit():
            return False
        
        # Filter out single characters
        cleaned = word.strip()
        if len(cleaned) < 2:
            return False
        
        # Filter out drug class terms and generic chemical fragments
        # that the NER model sometimes extracts instead of actual drug names
        NER_EXCLUDE = {
            # Drug class terms (not specific drugs)
            'ace', 'inhibitor', 'inhibitors', 'blocker', 'blockers',
            'antagonist', 'antagonists', 'agonist', 'agonists',
            'receptor', 'receptors', 'antibiotics', 'antibiotic',
            'analgesic', 'analgesics', 'nsaid', 'nsaids',
            'anticoagulant', 'anticoagulants', 'antiplatelet',
            'antihypertensive', 'antihypertensives',
            'diuretic', 'diuretics', 'steroid', 'steroids',
            'statin', 'statins', 'opioid', 'opioids',
            # Common chemical elements/compounds (usually not drug names)
            'calcium', 'sodium', 'potassium', 'magnesium', 'iron',
            'chloride', 'phosphate', 'sulfate', 'bicarbonate',
            'gluconate', 'acetate', 'citrate', 'oxide',
            'oxygen', 'nitrogen', 'carbon', 'hydrogen',
            # Generic medical terms NER sometimes picks up
            'saline', 'dextrose', 'water', 'solution',
            'tablet', 'capsule', 'injection', 'infusion',
        }
        if cleaned.lower() in NER_EXCLUDE:
            return False
        
        return True
    
    def _clean_entity(self, word: str) -> str:
        """
        Clean up entity text from tokenizer artifacts.
        
        Args:
            word: Raw entity text from the NER pipeline
            
        Returns:
            Cleaned entity text
        """
        # Remove leading/trailing punctuation and whitespace
        cleaned = word.strip()
        cleaned = re.sub(r'^[^a-zA-Z0-9]+', '', cleaned)
        cleaned = re.sub(r'[^a-zA-Z0-9]+$', '', cleaned)
        
        # Remove tokenizer special tokens
        cleaned = cleaned.replace('[UNK]', '').replace('[SEP]', '').replace('[CLS]', '')
        cleaned = cleaned.replace('##', '')
        
        return cleaned.strip()
    
    def extract_and_validate(self, text: str, validate: bool = True) -> List[str]:
        """
        Extract drug names using PubMedBERT NER and optionally validate against RxNorm.
        
        Args:
            text: Medical text to analyze
            validate: Whether to validate candidates against RxNorm
            
        Returns:
            List of validated drug names
        """
        candidates = self.extract_candidates(text)
        logger.info(f"PubMedBERT NER extracted {len(candidates)} candidates: {candidates}")
        
        if not validate:
            return sorted(candidates)
        
        validated = []
        for candidate in candidates:
            rxcui = self.rxnorm_client.find_rxcui_by_string(candidate)
            if rxcui:
                validated.append(candidate)
                logger.debug(f"  ✓ '{candidate}' validated (RxCUI: {rxcui})")
            else:
                # Try approximate matching
                matches = self.rxnorm_client.get_approximate_match(candidate, max_entries=1)
                if matches and float(matches[0].get('score', 0)) > 0:
                    validated.append(candidate)
                    logger.debug(f"  ✓ '{candidate}' validated via approximate match")
                else:
                    logger.debug(f"  ✗ '{candidate}' not found in RxNorm")
        
        logger.info(f"Validated {len(validated)}/{len(candidates)} drug candidates")
        return sorted(validated)


# =============================================================================
# Drug Information Retriever
# =============================================================================

class DrugInfoRetriever:
    """Retrieves comprehensive drug information from RxNorm."""
    
    def __init__(self, rxnorm_client: Optional[RxNormClient] = None):
        """
        Initialize the retriever.
        
        Args:
            rxnorm_client: RxNorm API client
        """
        self.client = rxnorm_client or RxNormClient()
        
    def retrieve(self, drug_name: str) -> RetrievalResult:
        """
        Retrieve drug information for a given drug name.
        
        Args:
            drug_name: Name of the drug to look up
            
        Returns:
            RetrievalResult with drug information
        """
        try:
            # First, try to get drugs directly
            drugs_data = self.client.get_drugs(drug_name)
            
            if drugs_data and 'drugGroup' in drugs_data:
                drug_group = drugs_data['drugGroup']
                concept_groups = drug_group.get('conceptGroup', [])
                
                # Find the best concept (prefer ingredients or brand names)
                best_concept = self._find_best_concept(concept_groups)
                if best_concept:
                    drug_info = self._build_drug_info(best_concept)
                    return RetrievalResult(
                        query_term=drug_name,
                        found=True,
                        drug_info=drug_info
                    )
            
            # If direct lookup fails, try approximate matching
            matches = self.client.get_approximate_match(drug_name)
            if matches:
                best_match = matches[0]
                rxcui = best_match.get('rxcui')
                if rxcui:
                    props = self.client.get_rxcui_properties(rxcui)
                    if props and 'properties' in props:
                        drug_info = DrugInfo(
                            rxcui=rxcui,
                            name=props['properties'].get('name', drug_name),
                            term_type=props['properties'].get('tty', 'Unknown'),
                            synonym=props['properties'].get('synonym')
                        )
                        return RetrievalResult(
                            query_term=drug_name,
                            found=True,
                            drug_info=drug_info
                        )
            
            return RetrievalResult(
                query_term=drug_name,
                found=False,
                error=f"No RxNorm data found for '{drug_name}'"
            )
            
        except Exception as e:
            logger.error(f"Error retrieving drug info for '{drug_name}': {e}")
            return RetrievalResult(
                query_term=drug_name,
                found=False,
                error=str(e)
            )
    
    def _find_best_concept(self, concept_groups: List[Dict]) -> Optional[Dict]:
        """Find the best concept from a list of concept groups."""
        # Priority: IN (ingredient) > BN (brand name) > SCD (clinical drug)
        priority_order = ['IN', 'BN', 'SCD', 'SBD', 'GPCK', 'BPCK']
        
        for tty in priority_order:
            for group in concept_groups:
                if group.get('tty') == tty and 'conceptProperties' in group:
                    concepts = group['conceptProperties']
                    if concepts:
                        return concepts[0] if isinstance(concepts, list) else concepts
                        
        # Fallback: return any concept with properties
        for group in concept_groups:
            if 'conceptProperties' in group:
                concepts = group['conceptProperties']
                if concepts:
                    return concepts[0] if isinstance(concepts, list) else concepts
                    
        return None
    
    def _build_drug_info(self, concept: Dict) -> DrugInfo:
        """Build a DrugInfo object from a concept."""
        rxcui = concept.get('rxcui', '')
        
        drug_info = DrugInfo(
            rxcui=rxcui,
            name=concept.get('name', ''),
            term_type=concept.get('tty', 'Unknown'),
            synonym=concept.get('synonym')
        )
        
        # Get related information if we have an rxcui
        if rxcui:
            related = self.client.get_all_related(rxcui)
            if related and 'allRelatedGroup' in related:
                groups = related['allRelatedGroup'].get('conceptGroup', [])
                
                for group in groups:
                    tty = group.get('tty', '')
                    concepts = group.get('conceptProperties', [])
                    if not isinstance(concepts, list):
                        concepts = [concepts]
                        
                    names = [c.get('name', '') for c in concepts if c.get('name')]
                    
                    if tty == 'IN':  # Ingredients
                        drug_info.ingredients = names[:5]
                    elif tty == 'DF':  # Dose forms
                        drug_info.dose_forms = names[:5]
                    elif tty in ['BN', 'SBD']:  # Brand names
                        drug_info.related_drugs.extend(names[:3])
                        
        return drug_info
    
    def retrieve_multiple(self, drug_names: List[str]) -> List[RetrievalResult]:
        """
        Retrieve information for multiple drugs.
        
        Args:
            drug_names: List of drug names to look up
            
        Returns:
            List of RetrievalResults
        """
        return [self.retrieve(name) for name in drug_names]


# =============================================================================
# RAG Context Builder
# =============================================================================

class RxNormRAGContext:
    """Builds augmented prompts with RxNorm drug context."""
    
    def __init__(self, extractor_type: str = "pubmedbert", **extractor_kwargs):
        """
        Initialize the RAG context builder.
        
        Args:
            extractor_type: Type of drug extractor to use.
                - "pubmedbert": PubMedBERT NER model (default, higher accuracy)
                - "regex": Legacy regex-based extractor (faster, no model download)
            **extractor_kwargs: Additional kwargs passed to the extractor constructor.
                For PubMedBERT: model_name, confidence_threshold, device
                For regex: (none)
        """
        self.client = RxNormClient()
        self.extractor_type = extractor_type
        
        if extractor_type == "pubmedbert":
            self.extractor = PubMedBERTDrugExtractor(
                rxnorm_client=self.client,
                **extractor_kwargs
            )
        elif extractor_type == "regex":
            self.extractor = DrugNameExtractor(self.client)
        else:
            raise ValueError(
                f"Unknown extractor_type: '{extractor_type}'. "
                f"Use 'pubmedbert' or 'regex'."
            )
        
        self.retriever = DrugInfoRetriever(self.client)
        logger.info(f"RxNormRAGContext initialized with '{extractor_type}' extractor")
        
    def extract_drugs_from_text(self, text: str, validate: bool = True) -> List[str]:
        """
        Extract drug names from medical text.
        
        Args:
            text: Medical text to analyze
            validate: Whether to validate against RxNorm
            
        Returns:
            List of extracted drug names
        """
        return self.extractor.extract_and_validate(text, validate=validate)
    
    def get_drug_context(self, drug_names: List[str]) -> str:
        """
        Get formatted context string for a list of drugs.
        
        Args:
            drug_names: List of drug names to look up
            
        Returns:
            Formatted context string
        """
        if not drug_names:
            return ""
            
        results = self.retriever.retrieve_multiple(drug_names)
        
        context_parts = []
        for result in results:
            if result.found and result.drug_info:
                context_parts.append(result.drug_info.to_context_string())
                
        if not context_parts:
            return ""
            
        return "=== DRUG REFERENCE INFORMATION ===\n" + "\n\n".join(context_parts) + "\n=== END DRUG REFERENCE ==="
    
    def build_augmented_prompt(
        self,
        original_text: str,
        system_prompt: str,
        include_context: bool = True,
        max_drugs: int = 10
    ) -> Tuple[str, str, List[str]]:
        """
        Build an augmented prompt with drug context.
        
        Args:
            original_text: The medical text to analyze
            system_prompt: The original system prompt
            include_context: Whether to include drug context
            max_drugs: Maximum number of drugs to look up
            
        Returns:
            Tuple of (augmented_system_prompt, user_prompt, extracted_drugs)
        """
        extracted_drugs = []
        augmented_system = system_prompt
        
        if include_context:
            # Extract drug names
            extracted_drugs = self.extract_drugs_from_text(original_text, validate=True)[:max_drugs]
            
            if extracted_drugs:
                # Get drug context
                drug_context = self.get_drug_context(extracted_drugs)
                
                if drug_context:
                    # Augment system prompt with drug context
                    augmented_system = f"""{system_prompt}

You have access to the following reference information about drugs mentioned in the text:

{drug_context}

Use this reference information to help identify and correct any medical errors related to these drugs."""
        
        return augmented_system, original_text, extracted_drugs


# =============================================================================
# Convenience Functions
# =============================================================================

def lookup_drug(name: str) -> Optional[DrugInfo]:
    """
    Quick lookup of a single drug.
    
    Args:
        name: Drug name to look up
        
    Returns:
        DrugInfo or None
    """
    retriever = DrugInfoRetriever()
    result = retriever.retrieve(name)
    return result.drug_info if result.found else None


def get_drug_context_for_text(text: str, extractor_type: str = "pubmedbert") -> str:
    """
    Extract drugs from text and return context string.
    
    Args:
        text: Medical text to analyze
        extractor_type: "pubmedbert" or "regex"
        
    Returns:
        Formatted drug context string
    """
    rag = RxNormRAGContext(extractor_type=extractor_type)
    drugs = rag.extract_drugs_from_text(text)
    return rag.get_drug_context(drugs)


# =============================================================================
# Main (Demo/Test)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RxNorm RAG Module Demo")
    parser.add_argument(
        "--extractor", "-e", type=str, default="pubmedbert",
        choices=["pubmedbert", "regex"],
        help="Drug name extractor to use (default: pubmedbert)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare regex vs PubMedBERT extraction side-by-side"
    )
    args = parser.parse_args()
    
    # Demo usage
    print("=" * 60)
    print("RxNorm RAG Module Demo")
    print("=" * 60)
    
    sample_text = """
    The patient was prescribed metformin 500mg twice daily for diabetes management.
    She was also given lisinopril 10mg for hypertension and atorvastatin 20mg for 
    cholesterol. The patient reports taking ibuprofen occasionally for headaches.
    """
    
    # Test drug lookup
    print("\n1. Testing single drug lookup:")
    print("-" * 40)
    drug = lookup_drug("metformin")
    if drug:
        print(drug.to_context_string())
    else:
        print("Drug not found")
    
    if args.compare:
        # Side-by-side comparison
        print("\n2. COMPARISON: Regex vs PubMedBERT extraction")
        print("=" * 60)
        
        print("\n  [Regex extractor]")
        print("  " + "-" * 38)
        regex_rag = RxNormRAGContext(extractor_type="regex")
        regex_drugs = regex_rag.extract_drugs_from_text(sample_text)
        print(f"  Extracted drugs: {regex_drugs}")
        
        print("\n  [PubMedBERT NER extractor]")
        print("  " + "-" * 38)
        bert_rag = RxNormRAGContext(extractor_type="pubmedbert")
        bert_drugs = bert_rag.extract_drugs_from_text(sample_text)
        print(f"  Extracted drugs: {bert_drugs}")
        
        # Show differences
        regex_set = set(regex_drugs)
        bert_set = set(bert_drugs)
        print(f"\n  Only in regex:     {regex_set - bert_set or '(none)'}")
        print(f"  Only in PubMedBERT: {bert_set - regex_set or '(none)'}")
        print(f"  In both:           {regex_set & bert_set or '(none)'}")
    else:
        # Standard demo with selected extractor
        print(f"\n2. Testing drug extraction ({args.extractor}):")
        print("-" * 40)
        
        rag = RxNormRAGContext(extractor_type=args.extractor)
        drugs = rag.extract_drugs_from_text(sample_text)
        print(f"Extracted drugs: {drugs}")
        
        # Test context building
        print("\n3. Testing context building:")
        print("-" * 40)
        context = rag.get_drug_context(drugs[:3])  # Limit for demo
        print(context)
        
        # Test augmented prompt
        print("\n4. Testing augmented prompt building:")
        print("-" * 40)
        system_prompt = "You are a medical expert reviewing clinical text for errors."
        aug_system, user_prompt, extracted = rag.build_augmented_prompt(
            sample_text, 
            system_prompt
        )
        print(f"Extracted drugs: {extracted}")
        print(f"\nAugmented system prompt:\n{aug_system[:500]}...")
