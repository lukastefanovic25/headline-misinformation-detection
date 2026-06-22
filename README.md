# headline-misinformation-detection
# You Can't Judge an Article by its Headline, or Can You?

Headline-only misinformation detection across three news datasets, benchmarking classical ML models against a fine-tuned DistilBERT and using SHAP to check whether the most predictive features hold up out-of-distribution.

Course project for DS & AI II: Data and Algorithmic Governance.
Authors: Luka Stefanovic & Elisabeth Retegan 

## Overview

Most misinformation research scores models on full article text, but most readers only ever see the headline before sharing or reacting. This project asks whether that headline alone is enough signal to flag misinformation, and if so, whether the features driving that signal are genuine markers of misleading content or just artifacts of how each dataset was collected.

Main research question: Can supervised ML models reliably detect misinformation from headline text alone, and which features, model families, and cross-dataset conditions govern that performance?

Sub-questions:

1. Which linguistic, semantic, and stylistic features of headlines are most predictive of misinformation on an out-of-distribution test set?
2. How do different model families compare on headline-only classification, and how does performance change on a dataset they were never trained on?
3. Do SHAP's most important features stay consistent across datasets, or do they reveal source-style overfitting?

The motivation for this project is threefold. First, headlines are the true point of exposure. Namely, most misinformation spreads and 
lodges in memory through the title alone, never the body (Sundar, 2024; Ecker et al., 2014). This is the reason why headline-only 
detection is worth studying. Second, headline-level accuracy alone can be deceptive because a model may post strong numbers while keying 
on publisher names or collection artifacts rather than any genuine marker of misleading content. Therefore, we pair every classifier with 
SHAP explainability, making the result be generalisable. Third, EU regulation is moving the same way, with the 
Digital Services Act (Art. 34–35) requiring large platforms to assess systemic risks widely understood to include disinformation, and 
the AI Act (Regulation EU 2024/1689) pushing content-facing AI toward transparency about how outputs are produced.

## Notebook structure

`Final_Headline_Project.ipynb` runs top to bottom as a single pipeline. Each numbered section depends on the one before it.

| Section | Content |
|---|---|
| 0 | Environment setup and package imports |
| 1 | Problem framing and research questions |
| 2 | Loading WELFake, ISOT, and EU_mix, with label harmonization to a common 0=Real / 1=Fake convention |
| 3 | Cleaning (nulls, duplicates, leakage-prone columns such as ISOT's `subject` field) |
| 4 | Leakage audit: exact-match and near-duplicate (MinHash) overlap checks within and across datasets |
| 5 | Exploratory data analysis: class balance, headline length, top unigrams, vocabulary overlap, publisher/style markers, topic skew via LDA |
| 6 | Train / validation / test split on WELFake (stratified 70/15/15) |
| 7 | Feature engineering: lexical/surface features, lexicon overlap, sentiment (VADER), named entities and POS tags (spaCy), TF-IDF (word and character n-grams), and final feature assembly |
| 8 | Modeling: naive baselines, logistic regression, multinomial naive Bayes, linear SVM, random forest, and a DistilBERT fine-tune across 3 seeds, followed by threshold tuning, McNemar tests, ROC/PR curves, confusion matrices, and error analysis |
| 9 | SHAP explainability for random forest and logistic regression, cross-dataset feature stability (Spearman correlation), and a correctness/completeness check of SHAP on DistilBERT |
| 10 | Conclusion and synthesis answering the three research questions |

WELFake is the only dataset used for training and with a 70/15/15 split also used for in-distribution testing. ISOT and EU_mix are held out 
entirely as external, out-of-distribution test sets, never seen during training or feature fitting.

## Datasets

The notebook expects the following layout relative to the notebook file. Raw data files are not included in this repository because of their size; those need to be placed manually before running.

```
Datasets/
├── WELFake_Dataset.csv
├── News-_dataset/
│   ├── True.csv
│   └── Fake.csv
└── thrid dataset/
    └── EU_mix_3_EMNAD_final.txt
```

(The `thrid dataset` folder name is a typo carried over from the original data drop. The notebook's `Path` references use that exact spelling, so keep it as is rather than correcting it.)

- **WELFake**: 72,134 labeled headlines, near-balanced (51.4% fake / 48.6% real). Used for training, validation, and the in-distribution test split.
- **ISOT**: University of Victoria fake/real news corpus, split across `True.csv` and `Fake.csv`. Label is derived from which file a row came from. The `subject` column is dropped because it perfectly predicts the label and would leak the answer.
- **EU_mix**: a balanced 50/50 set of European headlines, used as a second external test set.

All three datasets are remapped to the same label convention (0 = Real, 1 = Fake) in Section 2 before any cleaning happens.

Here are the links to the three used datasets for this project: 
Ahmed, H., Traore, I., & Saad, S. (2017). ISOT Fake News Dataset… University of Victoria. 

https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

Gruensteidl, M. N., & Kirrane, S. (2025). EU_mix evaluation dataset (EMNAD subset) [Data set]. XAI-IDD-Resources, GitHub. 
https://github.com/mangrue/XAI-IDD-Resources/blob/main/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_3_EMNAD_final.csv

Verma, P. K., Agrawal, P., & Prodan, R. (2021). WELFake dataset for fake news detection in text data [Data set]. Zenodo. 
https://doi.org/10.5281/zenodo.4561253

## Prerequisites

- Python 3.9 (the notebook was developed and pinned against 3.9.6)
- A Jupyter installation that can register a custom kernel
- macOS with Apple Silicon is what this was built and run on; CPU-only is sufficient, the DistilBERT training loop in Section 8.6 runs on CPU regardless of MPS availability

### Required packages

```
numpy<2
pandas
matplotlib
seaborn
scikit-learn
shap
spacy
en_core_web_sm     # spaCy model, installed separately via spacy download
vaderSentiment
textstat
datasketch
transformers
torch==2.2.2
statsmodels
scipy
ipykernel
```

`numpy<2` and `torch==2.2.2` are hard requirements, not suggestions. Section 0 asserts both versions explicitly and raises an error if they don't match, since several downstream calls (in particular the HuggingFace/torch interop) break across that version boundary.

## Setup

1. Create and activate a virtual environment:

   ```
   python3 -m venv ~/ds2_env
   source ~/ds2_env/bin/activate
   ```

2. Install the packages listed above, for example:

   ```
   pip install "numpy<2" pandas matplotlib seaborn scikit-learn shap spacy \
       vaderSentiment textstat datasketch transformers torch==2.2.2 \
       statsmodels scipy ipykernel
   python -m spacy download en_core_web_sm
   ```

3. Register the environment as a Jupyter kernel:

   ```
   ~/ds2_env/bin/python -m ipykernel install --user --name ds2_env --display-name "Python (ds2_env)"
   ```

4. Open the notebook and select the "Python (ds2_env)" kernel before running anything. Section 0 will fail loudly with a clear message if the wrong kernel or package versions are active, rather than letting a version mismatch cause confusing errors later in the notebook.

## Running the notebook

Run cells top to bottom. The pipeline is sequential: feature matrices built in Section 7 feed directly into Section 8, and Section 8's trained models feed directly into Section 9's SHAP analysis. Skipping cells or running out of order will raise `NameError`s on variables that haven't been defined yet.

Two directories are created automatically on first run, relative to the notebook's working directory:

- `outputs/` — figures, result tables, and CSV exports generated throughout the notebook
- `models/` — pickled vectorizers and sklearn models, plus one DistilBERT checkpoint subfolder per random seed

Random seeds are fixed at `[42, 123, 456]` and set globally at the start of Section 0, and again per seed inside the DistilBERT loop in Section 8.6, so results should be reproducible run to run.

Be aware that Section 8 (DistilBERT fine-tuning across 3 seeds) and Section 9 (SHAP value computation, especially for DistilBERT in 9.4) are the most computationally expensive parts of the notebook. On CPU, expect this to take a substantial amount of time; budget accordingly if running the whole notebook in one sitting.

## Results summary

Every model beats the naive baselines on WELFake (macro-F1: DistilBERT 0.922, linear SVM 0.890, logistic regression 0.870, naive Bayes 0.835, random forest 0.770), so headline-only detection works in-distribution. That ranking flips out-of-distribution: the strongest in-distribution models lose 53-64 macro-F1 points on EU_mix, with DistilBERT dropping to 0.278, below the weakest baseline. Random forest is the exception, improving on ISOT (0.876) and degrading the least on EU_mix (0.514).

SHAP explains why. Random forest's top features (`uppercase_ratio`, `pos_propn_ratio`, `ner_org`, `titlecase_ratio`) describe writing style, not topic, and stay stable across datasets (Spearman ρ = 0.96 and 0.95 against ISOT and EU_mix, both p < 0.0001). Logistic regression's top TF-IDF features, by contrast, are dominated by publisher fingerprints like `breitbart` and fragments of `new york times`, which is a property of how WELFake was assembled rather than a property of misinformation itself. The full reasoning is in Section 10.

## Repository contents

- `Final_Headline_Project.ipynb` — the notebook described above
- `Datasets/` — input data (not version controlled; see Datasets section)
- `outputs/` — generated figures and result tables
- `models/` — generated model artifacts and checkpoints


Alongside the main notebook, Classical_Models_No_Normalization.ipynb reruns the classical modeling pipeline (Sections 2-8, excluding DistilBERT and SHAP) with one change: the title normalization step (lowercasing, URL/handle stripping, source attribution removal) is skipped, so models train directly on raw headline text. We ran this as a sanity check on our own preprocessing choices: TF-IDF based models (logistic regression, naive Bayes, linear SVM) barely moved (under half a point of macro-F1), suggesting the genuine discriminative vocabulary dominates regardless of casing or stray URLs. Random forest was more sensitive, losing about 2.5 points without normalization, since its engineered features (like uppercase ratio) pick up noise that normalization would otherwise have absorbed.
