# Knowledge Retrieval Module

This module provides Retrieval-Augmented Generation (RAG) capabilities for the MEDIQA-CORR medical error detection task. It uses external knowledge sources to provide context to LLMs before making decisions.

## RxNorm RAG (`rxnorm_rag.py`)

The RxNorm RAG module uses the [RxNorm REST API](https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html) from the National Library of Medicine to retrieve drug information.

### Features

1. **Drug Name Extraction**: Automatically extracts potential drug names from medical text using pattern matching
2. **RxNorm Validation**: Validates extracted names against the RxNorm database
3. **Drug Information Retrieval**: Fetches comprehensive drug information including:
   - Drug names and synonyms
   - Active ingredients
   - Dose forms
   - Related drugs (brand names, generics)
4. **Context Augmentation**: Builds augmented prompts with drug reference information

### Usage

```python
from rxnorm_rag import RxNormRAGContext, lookup_drug

# Quick single drug lookup
drug = lookup_drug("metformin")
if drug:
    print(drug.to_context_string())

# Full RAG workflow
rag = RxNormRAGContext()

# Extract drugs from medical text
medical_text = """
The patient was prescribed metformin 500mg twice daily for diabetes.
She was also given lisinopril 10mg for hypertension.
"""

# Get drug names
drugs = rag.extract_drugs_from_text(medical_text)
print(f"Extracted: {drugs}")

# Get formatted context
context = rag.get_drug_context(drugs)
print(context)

# Build augmented prompt
system_prompt = "You are a medical expert reviewing clinical text."
aug_system, user_prompt, extracted = rag.build_augmented_prompt(
    medical_text, 
    system_prompt
)
```

## RxNorm RAG Experiment (`rxnorm_rag_experiment.py`)

Compares Gemini 2.5 Flash performance with and without RxNorm RAG augmentation.

### Running the Experiment

```bash
# Run with 10 samples
python rxnorm_rag_experiment.py --num_samples 10

# Run full 50-sample comparison
python rxnorm_rag_experiment.py

# Customize parameters
python rxnorm_rag_experiment.py --num_samples 20 --rate_limit 1.0 --model gemini-2.5-flash
```

### Results (10-sample subset)

| Metric | Baseline | RAG | Delta |
|--------|----------|-----|-------|
| Error Flag Accuracy | 0.80 | 0.80 | +0.00 |
| Sentence ID Accuracy | 0.70 | 0.70 | +0.00 |
| Drugs Extracted | 0 | 8 | +8 |
| Avg Drugs/Sample | 0 | 0.80 | - |

**Key Observations:**
- On the 10-sample subset, RAG did not change any predictions
- Drug extraction successfully identified medications like "hydralazine"
- Some false positives remain (e.g., "wine", "calcium" as supplements)

### Output Files

The experiment generates three output files in `outputs/`:
- `baseline_{timestamp}.txt` - Baseline predictions
- `rag_{timestamp}.txt` - RAG-augmented predictions  
- `comparison_{timestamp}.json` - Detailed comparison with metrics

## API Endpoints Used

| Endpoint | Description |
|----------|-------------|
| `/drugs` | Get drug products by name |
| `/rxcui` | Find RxCUI by string |
| `/rxcui/{rxcui}/properties` | Get concept properties |
| `/rxcui/{rxcui}/allrelated` | Get all related concepts |
| `/approximateTerm` | Approximate string matching |
| `/spellingsuggestions` | Get spelling suggestions |

## Rate Limiting

The module includes built-in rate limiting (100ms default) to respect the RxNorm API's usage guidelines.

## Dependencies

- `requests` - HTTP client for API calls
- `google-generativeai` - For Gemini API (experiments only)
- `datasets` - For loading MEDEC dataset (experiments only)

## No API Key Required

The RxNorm API is free to use and does not require an API key. See the [Terms of Service](https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html) for usage guidelines.

## Future Improvements

1. **Better Drug Extraction**: Use NER models for more accurate drug name extraction
2. **Drug Interaction Checking**: Use RxNorm's interaction APIs
3. **Additional Knowledge Sources**:
   - SNOMED CT - For disease/condition terminology
   - ICD-10 - For diagnosis codes
   - UpToDate - For clinical guidelines (requires license)

