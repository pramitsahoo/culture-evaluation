# DIWALI

[![Paper](https://img.shields.io/badge/Paper-EMNLP%202025-blue)](https://aclanthology.org/2025.emnlp-main.1706/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/nlip/DIWALI)
[![Project](https://img.shields.io/badge/Project-Page-green)](https://nlip-lab.github.io/nlip/publications/diwali/)

This repository contains the evaluation code for the DIWALI (Diversity and Inclusivity aWare cuLture specific Items for India) project, which assesses Large Language Models (LLMs) for cultural text adaptation in the Indian context.

## Citation

```bibtex
@inproceedings{sahoo-etal-2025-diwali,
    title = "{DIWALI} - Diversity and Inclusivity a{W}are cu{L}ture specific Items for {I}ndia: Dataset and Assessment of {LLM}s for Cultural Text Adaptation in {I}ndian Context",
    author = "Sahoo, Pramit  and
      Brahma, Maharaj  and
      Desarkar, Maunendra Sankar",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1706/",
    pages = "33587--33614",
    ISBN = "979-8-89176-332-6",
    abstract = "Large language models (LLMs) are widely used in various tasks and applications. However, despite their wide capabilities, they are shown to lack cultural alignment (CITATION) and produce biased generations (CITATION) due to a lack of cultural knowledge and competence. Evaluation of LLMs for cultural awareness and alignment is particularly challenging due to the lack of proper evaluation metrics and unavailability of culturally grounded datasets representing the vast complexity of cultures at the regional and sub-regional levels. Existing datasets for culture specific items (CSIs) focus primarily on concepts at the regional level and may contain false positives. To address this issue, we introduce a novel CSI dataset for Indian culture, belonging to 17 cultural facets. The dataset comprises {\textasciitilde}8k cultural concepts from 36 sub-regions. To measure the cultural competence of LLMs on a cultural text adaptation task, we evaluate the adaptations using the CSIs created, LLM as Judge, and human evaluations from diverse socio-demographic region. Furthermore, we perform quantitative analysis demonstrating selective sub-regional coverage and surface-level adaptations across all considered LLMs. Our dataset is available here: https://huggingface.co/datasets/nlip/DIWALI, project webpage, and our codebase with model outputs can be found here: https://github.com/pramitsahoo/culture-evaluation."
}
```

## Overview

This codebase evaluates LLMs on their ability to perform culturally-aware text adaptation for the Indian context. It includes:

- Culturally adapted versions of GSM-8K math word problems
- Concept extraction and matching for cultural items
- Evaluation across multiple state-of-the-art LLMs
- Support for both English and Bengali language adaptations
- Integration with the CANDLE dataset for cultural concept analysis

## Repository Structure

```
culture-evaluation/
├── adaptations/          # Culturally adapted test sets
│   └── gsm-8k/          # GSM-8K adaptations for various models
│       ├── bengali_prompt/
│       └── *_gsm_8k_test.json
├── CANDLE/              # CANDLE dataset processing
│   ├── concept_extraction.py
│   ├── concepts.jsonl
│   └── concepts_bengali.jsonl
├── concept_matching/    # Concept matching evaluation scripts
│   └── concept_matching_*.py  # Scripts for different models
├── concept_matching_bengali/ # Bengali concept matching
├── data/                # Data loading utilities
│   └── data.py
├── dataset/             # Base datasets
│   └── gsm-8k/
│       ├── train.jsonl
│       └── test.jsonl
├── MGSM/                # Multilingual GSM-8K (Bengali)
│   ├── mgsm_bn.tsv
│   ├── concept_matching_bn_prompt/
│   └── output_bn_prompt/
├── model/               # Model inference scripts
│   ├── offline_copy_replaced_words.py
│   ├── mgsm_eng_propmpt.py
│   ├── mgsm_ben_propmpt.py
│   └── scoring-*.py     # Scoring scripts for different models
├── our_csis/            # Custom culture-specific items for India
│   ├── concept_matching/
│   ├── csis/
│   └── india_map/
├── output/              # Evaluation results
├── run.sh               # Main evaluation script (English)
└── run_bengali.sh       # Bengali evaluation script
```

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Required Python packages (see installation section)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pramitsahoo/culture-evaluation.git
cd culture-evaluation
```

2. Install dependencies:
```bash
pip install torch transformers pandas datasets
```

3. Download the DIWALI dataset from HuggingFace:
```bash
# The dataset will be automatically downloaded when running evaluation scripts
# Or manually download from: https://huggingface.co/datasets/nlip/DIWALI
```

4. (Optional) Download the CANDLE dataset:
```bash
# Download from: https://www.mpi-inf.mpg.de/fileadmin/inf/d5/research/candle/candle_dataset_v1.jsonl.zip
# Extract to CANDLE/ directory
```

## Usage

### Running Evaluations

#### English Prompt Evaluation

Evaluate a model on culturally adapted GSM-8K problems with English prompts:

```bash
./run.sh <DEVICE_ID> <MODEL_NAME>
```

Example:
```bash
./run.sh 0 llama-3
```

#### Bengali Prompt Evaluation

Evaluate a model with Bengali prompts:

```bash
./run_bengali.sh <DEVICE_ID> <MODEL_NAME>
```

Example:
```bash
./run_bengali.sh 0 llama-3
```

#### Supported Models

The framework supports evaluation of the following models:
- `llama-2` (LLaMA-2 7B)
- `llama-3` (LLaMA-3 8B)
- `llama-3.2-1b` (LLaMA-3.2 1B)
- `llama-3.2-3b` (LLaMA-3.2 3B)
- `mistral` (Mistral 7B)
- `gemma-2-2b` (Gemma-2 2B)
- `gemma-2-9b` (Gemma-2 9B)

### Concept Matching

To evaluate concept matching performance:

```bash
cd concept_matching
python concept_matching_<model_name>.py
```

Example:
```bash
python concept_matching_llama2.py
```

### Custom Evaluations

To run custom evaluations, modify the parameters in the run scripts or directly call the Python scripts:

```python
python model/offline_copy_replaced_words.py \
    --dataset gsm_8k \
    --split test \
    --model llama-3 \
    --output_dir ./output/custom_evaluation/
```

## Output Format

Evaluation results are saved in the `output/` directory with the following structure:

```
output/
├── new_prompt_english/     # English prompt results
│   └── <model_name>/
│       ├── predictions.json
│       └── scores.txt
├── bengali/                # Bengali prompt results
└── LLM_Scores_*/          # Detailed scoring results
```

Each output file contains:
- Model predictions
- Cultural adaptation scores
- Concept matching accuracy
- Detailed explanations (if enabled)

## Dataset Information

### DIWALI Dataset

The DIWALI dataset is available on HuggingFace: [nlip/DIWALI](https://huggingface.co/datasets/nlip/DIWALI)

It contains:
- Culturally adapted math word problems
- Culture-specific items (CSIs) for India
- Mappings between source and target cultural concepts
- Human annotations for quality assessment

### GSM-8K Integration

This evaluation uses culturally adapted versions of the GSM-8K dataset, where original cultural references are replaced with Indian cultural equivalents while maintaining mathematical complexity and reasoning requirements.

## Key Features

1. **Cultural Adaptation**: Systematic replacement of culture-specific items with Indian equivalents
2. **Multilingual Support**: Evaluation in both English and Bengali
3. **Multiple LLMs**: Support for various state-of-the-art language models
4. **Concept Matching**: Evaluation of models' understanding of cultural concepts
5. **CANDLE Integration**: Leverages the CANDLE dataset for cultural concept extraction
6. **Comprehensive Metrics**: Detailed scoring and explanation generation

## Results

Detailed results and analysis can be found in the paper. The evaluation shows varying performance across different LLMs in handling culturally adapted content, with implications for developing more inclusive AI systems.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors through the project page

## Acknowledgments

This work builds upon the GSM-8K and CANDLE datasets. We thank the creators of these resources for making them publicly available.

---

For more details, please refer to the [paper](https://huggingface.co/papers/2509.17399) and [project page](https://nlip-lab.github.io/nlip/publications/diwali/).
