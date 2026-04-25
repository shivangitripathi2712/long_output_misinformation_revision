# Long-Output Misinformation Revision for Paragraph-Level Hallucination Correction

This repository contains the implementation for our research on paragraph-level misinformation correction in Large Language Models (LLMs). The project focuses on detecting and correcting hallucinated or unsupported information in long-form LLM-generated text using evidence-grounded sequential revision.

This code is directly related to our paper:

**REVISE: A Framework for Paragraph-Level Misinformation Correction in Large Language Models**

In this work, we study how long LLM-generated outputs can contain multiple factual claims, where some parts may be correct and others may be unsupported or misleading. Instead of rewriting the entire paragraph, the proposed framework revises the text at a finer sentence or claim level using retrieved evidence and agreement checking.

---

## Research Objective

Large Language Models often generate long responses that appear fluent and confident but may contain factual errors, unsupported claims, or hallucinated information. These errors are difficult to detect and correct because misinformation may occur only in specific parts of a paragraph.

The main objective of this project is to correct misinformation in long-form LLM outputs while preserving the original meaning, structure, and fluency as much as possible.

The framework is designed to:

- Process long LLM-generated passages.
- Decompose long outputs into smaller sentence-level or claim-level units.
- Generate fact-checking questions for each claim.
- Retrieve evidence from external sources.
- Check whether the evidence supports or contradicts the claim.
- Revise only the unsupported or contradictory parts.
- Preserve claims that are already supported by evidence.
- Produce a final revised paragraph with improved factual consistency.

---

## Paper Description

### REVISE: A Framework for Paragraph-Level Misinformation Correction in Large Language Models

This paper proposes a framework for correcting misinformation in paragraph-level LLM outputs. The key idea is that long LLM-generated passages should not be corrected as a single block. Instead, the paragraph should be analyzed sentence by sentence so that factual errors can be corrected locally without changing the entire response unnecessarily.

The framework follows an evidence-grounded revision process. First, the long output is decomposed into smaller claims. Then, fact-checking questions are generated for each claim. These questions are used to retrieve external evidence. An agreement gate checks whether the retrieved evidence supports or contradicts the claim. If the evidence contradicts the claim, the claim is revised. If the evidence supports the claim, the claim is preserved with minimal change.

This approach helps improve factual reliability while reducing unnecessary rewriting.

---

## Framework Pipeline

The overall pipeline contains the following stages:

### 1. Input Long-Form Response

The system takes a long LLM-generated answer or paragraph as input.

### 2. Sentence-Level Decomposition

The long paragraph is divided into smaller sentences or claim-like units. This allows the framework to verify each part independently.

### 3. Question Generation

For each sentence or claim, the framework generates fact-checking questions. These questions help identify what information needs to be verified.

### 4. Evidence Retrieval

The generated questions are used to retrieve relevant evidence from external sources.

### 5. Agreement Gate

The agreement gate checks whether the retrieved evidence supports, contradicts, or is insufficient for the current claim.

### 6. Sequential Revision

If the evidence contradicts the claim, the claim is revised using the evidence. If the claim is already supported, it is kept mostly unchanged.

### 7. Final Revised Output

The revised sentences are combined into a final corrected paragraph.

---

## Repository Structure

```text
long_output_misinformation_revision/
│
├── run_editor_sequential.py        # Main script for sequential misinformation revision
├── make_excel.py                   # Converts output files into Excel format for analysis
├── test_one.py                     # Script for testing one example
├── tet.py                          # Additional testing script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── prompts/
│   ├── hallucination_prompts.py    # Prompts for hallucination or misinformation generation
│   └── rarr_prompts.py             # Prompts for question generation, agreement, and editing
│
├── utils/
│   ├── agreement_gate.py           # Checks whether evidence agrees with the claim
│   ├── editor.py                   # Revises claims using retrieved evidence
│   ├── evidence_selection.py       # Selects useful evidence for attribution
│   ├── hallucination.py            # Hallucination/misinformation processing utilities
│   ├── question_generation.py      # Generates fact-checking questions
│   └── search.py                   # Retrieves evidence
│
└── figs/
    └── RARR.jpg                    # Framework figure
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shivangitripathi2712/long_output_misinformation_revision.git
cd long_output_misinformation_revision
```

Create a new Python environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

## API Setup

This project may require API keys for LLM-based revision and evidence retrieval.

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Set your search API key if your implementation uses Bing/Azure search:

```bash
export AZURE_SEARCH_KEY="your_azure_search_key"
```

If your `utils/search.py` uses Tavily or another search API, set the corresponding key:

```bash
export TAVILY_API_KEY="your_tavily_api_key"
```

Do not write API keys directly in the code before uploading to GitHub.

---

## Input File Format

The input file should be in JSONL format. Each line should contain one example.

Example:

```json
{"input_info": {"claim": "The Eiffel Tower is located in Berlin and was built in 1900."}}
```

For long-form paragraph correction, the claim field can contain a full paragraph:

```json
{"input_info": {"claim": "Marie Curie discovered radium and polonium. She won Nobel Prizes in physics and chemistry. She also founded NASA in 1958."}}
```

---

## How to Run the Full Pipeline

Run the sequential revision framework using:

```bash
python run_editor_sequential.py \
  --input_file statement.jsonl \
  --output_file output.jsonl \
  --model_name text-davinci-003 \
  --claim_field claim
```

If your implementation uses a newer model, replace the model name:

```bash
python run_editor_sequential.py \
  --input_file statement.jsonl \
  --output_file output.jsonl \
  --model_name gpt-3.5-turbo \
  --claim_field claim
```

Argument explanation:

```text
--input_file      Path to the input JSONL file
--output_file     Path where the revised output will be saved
--model_name      Language model used for revision
--claim_field     Field name containing the claim or paragraph
```

---

## Running One Example

To test the framework on a single example:

```bash
python test_one.py
```

This is useful for debugging before running the full dataset.

---

## Convert Results to Excel

After generating the output JSONL file, convert the results into Excel format:

```bash
python make_excel.py
```

The Excel file can be used to analyze:

- Original claim
- Revised claim
- Retrieved evidence
- Agreement gate decision
- Number of evidence passages retrieved
- Whether the framework changed or preserved the original text

---

## Expected Output

The output file may include:

```text
original claim or paragraph
generated questions
retrieved evidence
agreement gate result
revised claim
selected evidence
final revised paragraph
```

The goal is to produce a revised output that is more factually grounded while staying close to the original text when the original claim is already supported.

---

## Evaluation

The revised outputs can be evaluated using:

- Factual correctness
- Evidence support
- Hallucination reduction
- Minimal unnecessary editing
- Semantic preservation
- Paragraph coherence
- Revision quality

This framework supports research on long-form hallucination correction, misinformation revision, and trustworthy LLM generation.

---

## Related Publications

This repository is associated with the following publications:

1. **REVISE: A Framework for Paragraph-Level Misinformation Correction in Large Language Models**  
   Shivangi Tripathi, Teancy Jennifer, Henry Griffith, and Heena Rathore.  
   2025 IEEE International Conference on Artificial Intelligence Testing (AITest), pp. 160–167, 2025.

2. **Detecting and Correcting Hallucinations in Paragraph-Level Text with Ensemble-Based Evaluation**  
   Shivangi Tripathi and Heena Rathore.  
   2025 IEEE 7th International Conference on Trust, Privacy and Security in Intelligent Systems, and Applications (TPS-ISA), pp. 623–631, 2025.

3. **Paragraph-Level Hallucination Detection and Correction for Trustworthy Large Language Models in Networked Systems**  
   Shivangi Tripathi, Teancy Jennifer, Henry Griffith, and Heena Rathore.  
   2026 IEEE 23rd Consumer Communications & Networking Conference (CCNC), pp. 1–6, 2026.

---

## Citation

If you use this code or build on this project, please cite the following papers:

```bibtex
@inproceedings{tripathi2025revise,
  title={REVISE: A Framework for Paragraph-Level Misinformation Correction in Large Language Models},
  author={Tripathi, Shivangi and Jennifer, Teancy and Griffith, Henry and Rathore, Heena},
  booktitle={2025 IEEE International Conference on Artificial Intelligence Testing (AITest)},
  pages={160--167},
  year={2025},
  organization={IEEE},
  doi={10.1109/AITest66680.2025.00027}
}

@inproceedings{tripathi2025detecting,
  title={Detecting and Correcting Hallucinations in Paragraph-Level Text with Ensemble-Based Evaluation},
  author={Tripathi, Shivangi and Rathore, Heena},
  booktitle={2025 IEEE 7th International Conference on Trust, Privacy and Security in Intelligent Systems, and Applications (TPS-ISA)},
  pages={623--631},
  year={2025},
  organization={IEEE},
  doi={10.1109/TPS-ISA67132.2025.00077}
}

@inproceedings{tripathi2026paragraph,
  title={Paragraph-Level Hallucination Detection and Correction for Trustworthy Large Language Models in Networked Systems},
  author={Tripathi, Shivangi and Jennifer, Teancy and Griffith, Henry and Rathore, Heena},
  booktitle={2026 IEEE 23rd Consumer Communications \& Networking Conference (CCNC)},
  pages={1--6},
  year={2026},
  organization={IEEE},
  doi={10.1109/CCNC65079.2026.11366495}
}
```

---

## Ethical Use

This repository is intended for academic and research purposes. The goal is to study and reduce hallucination and misinformation in LLM-generated text.

Do not use this project to generate, spread, or automate misinformation.

---

## Author

**Shivangi Tripathi**  
Ph.D. Student, Computer Science  
Texas State University  

Research interests: Large Language Models, hallucination detection and correction, misinformation revision, adversarial robustness, and trustworthy AI.
EOF
