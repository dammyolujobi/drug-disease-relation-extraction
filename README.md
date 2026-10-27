# Drug–Disease Relation Extraction with Fine-Tuned BioBERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dammyolujobi/drug-disease-relation-extraction/blob/main/Drug_Disease_Relation_Extraction_Training.ipynb)

Training and evaluation code for the final-year project *A Deep Learning-Based Drug-Disease Relation Extraction Model Using scispaCy and Textual Data from Biomedical Literature* (Department of Computer and Information Sciences, Covenant University, 2026).

Two drug–disease relation types are extracted from biomedical text as separate binary classification tasks:

| Task | Corpus | Positive label |
|---|---|---|
| causes | [BC5CDR](https://github.com/JHnlp/BioCreative-V-CDR-Corpus) (1,500 PubMed abstracts) | chemical-induced disease (CID) relation |
| treats | [BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) (600 PubMed abstracts) | chemical–disease `Negative_Correlation` (proxy for treats) |

Examples are built at sentence level by distant supervision: a chemical and a disease that occur in the same sentence form a candidate pair, labelled from the document-level gold relations, with the mentions wrapped in `[E1] … [/E1]` and `[E2] … [/E2]` marker tokens.

## Contents

| File | Description |
|---|---|
| `Drug_Disease_Relation_Extraction_Training.ipynb` | Data loading, example construction, the six baselines, BioBERT fine-tuning and test-set evaluation for both tasks, with the outputs of the reported run |

## Results

Seven models per task were compared on the same held-out test set. Precision, recall and F1 are for the positive class.

**Causes (BC5CDR, 4,888 test examples)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.665 | 0.542 | 0.733 | 0.623 |
| TF-IDF + linear SVM | 0.671 | 0.546 | 0.760 | 0.635 |
| TF-IDF + LightGBM | 0.609 | 0.488 | 0.732 | 0.585 |
| Frozen BioBERT + logistic regression | 0.675 | 0.554 | 0.708 | 0.622 |
| Frozen BioBERT + linear SVM | 0.657 | 0.535 | 0.695 | 0.604 |
| Frozen BioBERT + LightGBM | 0.645 | 0.521 | 0.747 | 0.614 |
| **Fine-tuned BioBERT** | **0.746** | **0.628** | **0.801** | **0.704** |

**Treats (BioRED, 505 test examples)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.800 | 0.427 | 0.356 | 0.388 |
| TF-IDF + linear SVM | 0.818 | 0.486 | 0.389 | 0.432 |
| TF-IDF + LightGBM | 0.764 | 0.344 | 0.356 | 0.350 |
| Frozen BioBERT + logistic regression | 0.800 | 0.459 | 0.689 | 0.551 |
| Frozen BioBERT + linear SVM | 0.838 | 0.534 | 0.700 | 0.606 |
| Frozen BioBERT + LightGBM | 0.824 | 0.511 | 0.267 | 0.350 |
| **Fine-tuned BioBERT** | **0.865** | **0.617** | **0.644** | **0.630** |

## Trained models

The trained models are available on Google Drive:
**[chapter4_models](https://drive.google.com/drive/folders/1Akw2CoHHtN_0EwKuQndqgro4ChQVWu4Q?usp=sharing)**

| Folder / file | Contents |
|---|---|
| `biobert_causes_final/` | fine-tuned BioBERT, causes task (weights, config, tokenizer with marker tokens) |
| `biobert_treats_final/` | fine-tuned BioBERT, treats task |
| `*_tfidf_vectorizer.joblib`, `*_tfidf_*.joblib`, `*_frozen_*.joblib` | TF-IDF vectorisers and baseline classifiers for each task |

After downloading a model folder:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path = "biobert_causes_final"            # or "biobert_treats_final"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path).eval()

text = "[E1] Carbamazepine [/E1] can cause [E2] Stevens-Johnson syndrome [/E2] in carriers of HLA-B*1502."
with torch.no_grad():
    logits = model(**tok(text, return_tensors="pt", truncation=True, max_length=256)).logits
print("probability of the relation:", torch.softmax(logits, dim=-1)[0, 1].item())
```

## Training setup

Google Colab with an NVIDIA Tesla T4 GPU; `dmis-lab/biobert-base-cased-v1.1`; Transformers 4.49.0; class-weighted cross-entropy; 3 epochs, learning rate 2 × 10⁻⁵, batch size 16, maximum sequence length 256 tokens, mixed precision. The causes training set is capped at 3,000 examples (all 1,710 positives and 1,290 sampled negatives); validation and test sets are used in full.

## Limitations

Labels come from document-level relations by sentence-level distant supervision, so some sentence labels are noisy. BioRED has no explicit treats relation, so the treats results are results on the `Negative_Correlation` proxy.

## Author

Olujobi Damilola David (dolujobi.2302010@stu.cu.edu.ng), supervised by Mr. Oluwadamilola Odunayo Emmanuel Osofuye, Department of Computer and Information Sciences, Covenant University, Ota, Nigeria.
