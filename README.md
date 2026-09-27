# LLM Hallucination Mitigation Pipeline for Extractive QA

*Developed as the final project for the Natural Language Processing course (B.Sc. Computer Science, 2026).*

An end-to-end post-processing and validation pipeline around **Meta's Llama-3.2-3B-Instruct** designed to detect unanswerable queries and mitigate hallucinations on the **SQuAD 2.0** benchmark.

## Overview & Motivation
Instruction-tuned LLMs exhibit a strong "urge to answer" even when the provided context lacks the necessary information, resulting in high false-positive hallucination rates on adversarial datasets like SQuAD 2.0. 

While 3-shot prompt engineering alone achieved strong recall on answerable questions (**85.41 F1**), it performed poorly on unanswerable queries (**19.22 F1**) due to lexical overlap traps, numeric mismatches, and entity confusion. This project introduces a **two-stage neuro-symbolic validation pipeline** that boosts unanswerable query F1 to **69.53%** without relying on a secondary LLM judge.

---

## Pipeline Architecture

### 1. Hardware-Aware Inference & Prompting
* **Dynamic Resource Allocation:** Automatically detects hardware availability, running `float16` on CUDA GPUs or optimizing CPU thread allocation (`float32`) to prevent thread thrashing.
* **3-Shot Constrained Prompting:** Uses strict system rules and 3-shot demonstrations (standard extraction, unanswerable trap, and near-miss lexical trap) with greedy decoding (`do_sample=False`) for deterministic span extraction.
* **Intrinsic Confidence Calibration:** Extracts token transition log-probabilities (`compute_transition_scores`) and computes the joint probability across generated tokens:
  $$P(\text{answer}) = \exp\left(\sum_{i} \log P(t_i \mid t_{<i})\right)$$

### 2. Stage 1: Deterministic Veto Layer (Fast Rejection)
Before running heavier neural checks, candidate answers must pass five strict gates:
1. **Confidence Gate:** Rejects outputs falling below the base transition confidence threshold (`< 0.6`).
2. **Exact Substring Verification:** Ensures the extracted span exists verbatim in the source passage.
3. **Adversarial Unit Consistency (`check_unit_consistency`):** Prevents unit-mismatch hallucinations (e.g., asking for *millions* when the text only mentions *billions*, or *meters* vs. *kilometers*).
4. **Exact Quantity Verification (`check_exact_numbers`):** Uses Regex to verify that multi-digit quantities (`3+` digits, excluding years) in the question appear in the context.
5. **Temporal Consistency (`check_year_consistency`):** Extracts 4-digit years (`1000–2099`) from the question and generated answer to block answers referencing conflicting time periods.

### 3. Stage 2: Grid-Searched Weighted Voting Ensemble
Answers surviving the veto layer are scored across four linguistic and semantic validators (requiring a threshold of `>= 4` to be accepted):
* **SBERT Semantic Similarity (`+3` / `-3`):** Encodes the concatenated `Question + Answer` against individual context sentences using `all-MiniLM-L6-v2`, requiring a maximum cosine similarity `> 0.2`.
* **Native LLM Confidence Tiering (`+3` for `>0.8`, `+1` for `>0.5`):** Rewards high token-level certainty.
* **Information Gain / Anti-Parroting (`+1` / `-2`):** Uses SpaCy lemmatization and stopword removal to verify that the answer's content lemmas are not merely a subset of the question's lemmas ($\text{Lemmas}_{A} \not\subseteq \text{Lemmas}_{Q}$).
* **NER Type Consistency (`+1` / `-1`):** Maps interrogative framing (`Who`, `Where`, `When`, `How much`, `How many`) to expected SpaCy Named Entity tags (`PERSON/ORG`, `GPE/LOC`, `DATE/TIME`, `MONEY/QUANTITY/CARDINAL`).

---

## Evaluation Results (1,000 SQuAD 2.0 Dev Samples)

| Pipeline Configuration | Overall Exact | Overall F1 | Answerable (`HasAns`) F1 | Unanswerable (`NoAns`) F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Prompting Only (No Validation Tests)** | - | - | **85.41%** | 19.22% |
| **Full Validation Pipeline (Final)** | **66.00%** | **67.29%** | 65.15% | **69.53%** |

* **Runtime Performance (50-sample benchmark):** `19.22s` on CUDA GPU (`float16`) | `422.27s` on CPU (`float32`).

---

## Key Error Analysis Insights
* **Mitigating "Near-Miss" Lexical Traps:** The model frequently hallucinated answers when unanswerable questions shared high lexical overlap with the passage (e.g., guessing *"Pacific"* for *"What ocean has interior valleys?"* from *"spans from Pacific Ocean islands to interior valleys"*). Combining 3-shot negative examples with SBERT semantic scoring substantially reduced these false positives.
* **Overcoming Numeric & Magnitude Blindness:** Small parameter LLMs struggle with magnitude discrepancies (e.g., `$20 million` vs. `$20 billion`). Deterministic Regex pre-filters proved far faster and more reliable than neural scoring for catching numeric traps.
* **The Precision-Recall Trade-off:** Enforcing strict confidence and semantic thresholds increased unanswerable F1 by **+50.3%**, at the cost of occasionally rejecting valid, multi-clause extractions where token transition probabilities naturally dip over longer spans.

---

## Getting Started

### Prerequisites
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. Authenticate with Hugging Face (requires access to `meta-llama/Llama-3.2-3B-Instruct`):
   ```bash
   huggingface-cli login
   ```

### Running the Pipeline
Configure the sample size in `config.json` and run:
```bash
python main.py
```
