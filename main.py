import json
import pandas as pd
import time
import spacy
import torch
import multiprocessing
import re

# Set the default device to CPU initially before hardware checks
torch.set_default_device('cpu')

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.evaluate_results import NO_ANSWER_MARKER, evaluate_results
from sentence_transformers import SentenceTransformer, util

def setup_resources():
    """
    This function dynamically sets up the hardware resources for PyTorch.
    It detects available CPU cores to prevent thread thrashing and checks for GPU availability.
    input: None
    return: tuple(device (str), dtype (torch.dtype)) - The optimal hardware device and data precision.
    """
    print("--- Dynamic Resource Setup ---")

    num_cores = min(8, multiprocessing.cpu_count() // 2)
    if num_cores < 1: num_cores = 1
    
    print(f"Setting PyTorch threads to: {num_cores}")
    torch.set_num_threads(num_cores)

    # Default settings for CPU
    device = "cpu"
    dtype = torch.float32

    # Check if a CUDA enabled GPU is available for hardware acceleration
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16 # Use 16 bit precision to save VRAM on GPU
        print("GPU detected! Using CUDA.")
    else:
        # Fallback to standard 32 bit precision to prevent slow software emulation on CPU
        print("Using CPU with float32.")

    print(f"Final Configuration -> Device: {device}, Dtype: {dtype}")
    return device, dtype

# Initialize global resources
device, model_dtype = setup_resources()
torch.set_default_device(device)

model_name = 'meta-llama/Llama-3.2-3B-Instruct'

print(f"Loading Tokenizer: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

# Map Llama model to the EOS token to prevent warnings.
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Loading Model: {model_name} with {model_dtype}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=model_dtype,
    token=True,
    low_cpu_mem_usage=True # Optimizes memory allocation during weight loading
)
model.config.pad_token_id = tokenizer.pad_token_id

# Transfer the model to the GPU if one is available
if device == "cuda":
    model.to(device)

print("Loading SpaCy and SBERT...")
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def check_exact_numbers(question, context):
    """
    This function checks if specific large numbers requested in the question exist in the context.
    It explicitly ignores numbers that represent historical years (1000-2099).
    input: question (str), context (str)
    return: bool - True if the numbers match or aren't present, False if there is a hallucinated mismatch.
    """
    # Find all numbers with 3 or more digits (optionally containing commas)
    q_nums = re.findall(r'\b\d{3,}(?:,\d{3})*\b', question)
    for num in q_nums:
        # Skip checking if the number looks like a year between 1000 and 2099
        if not re.match(r'^(?:1[0-9]{3}|20[0-9]{2})$', num):
            # If a specific quantity is asked about, it MUST be in the context
            if num not in context:
                return False
    return True

def check_year_consistency(question, answer):
    """
    This function verifies that the generated answer does not contain a year conflicting with the question.
    input: question (str), answer (str)
    return: bool - True if years are consistent, False if a temporal mismatch is detected.
    """
    # Extract years (1000-2099) from both the question and the generated answer
    q_years = re.findall(r'\b(?:1[0-9]{3}|20[0-9]{2})\b', question)
    a_years = re.findall(r'\b(?:1[0-9]{3}|20[0-9]{2})\b', answer)
    
    if q_years and a_years:
        # If both contain years, ensure the answer doesn't introduce a year absent from the question
        if not any(y in q_years for y in a_years):
            return False
    return True

def check_unit_consistency(question, context):
    """
    This function prevents unit mismatch hallucinations (e.g., answering a "millions" question using "billions" data).
    input: question (str), context (str)
    return: bool - True if units are consistent, False if an adversarial unit trap is detected.
    """
    q_lower = question.lower()
    c_lower = context.lower()
    
    # Define common adversarial unit pairs
    units = [
        ("million", "billion"),
        ("billion", "million"),
        ("thousand", "million"),
        ("meter", "kilometer")
    ]
    
    for u1, u2 in units:
        # If the question asks for u1, but the context only contains u2, the premise is false
        if u1 in q_lower:
            if u1 not in c_lower and u2 in c_lower:
                return False 
    return True

def check_SBERT_cos_sim(context, question, answer):
    """
    This function calculates the semantic similarity between the Q&A pair and the context.
    input: context (str), question (str), answer (str)
    return: bool - True if the highest cosine similarity score exceeds the threshold, False otherwise.
    """
    # Concatenate Question and Answer for semantic encoding
    quest_ans = f"{question} {answer}"

    # Segment context into individual sentences using SpaCy
    context_sentences = [sent.text for sent in nlp(context).sents]

    # Generate dense vector embeddings
    question_answer_emb = sbert_model.encode(quest_ans, convert_to_tensor=True)
    ctx_embs = sbert_model.encode(context_sentences, convert_to_tensor=True)

    # Calculate cosine similarity between the Q&A vector and all context sentence vectors
    scores = util.cos_sim(question_answer_emb, ctx_embs)[0]
    best_score = scores.max().item()

    return best_score > 0.2

def check_NER_type(question, answer):
    """
    This function validates the generated answer against expected Named Entity Recognition (NER) types based on the question.
    input: question (str), answer (str)
    return: bool - True if the entity type matches the question's interrogative word, False otherwise.
    """
    lower_question = question.lower()
    doc_answer = nlp(answer)
    expected_tags = []

    # Map interrogative words to expected SpaCy NER tags
    if len(doc_answer.ents) > 0:
        if "who " in lower_question:
            expected_tags = ["PERSON", "ORG", "NORP"]
        elif "where " in lower_question:
            expected_tags = ["GPE", "LOC", "FAC"]
        elif "how much" in lower_question:
            expected_tags = ["MONEY", "QUANTITY", "PERCENT"]
        elif "how many" in lower_question:
            expected_tags = ["CARDINAL", "QUANTITY"]
        elif "when" in lower_question:
            # Permit answers that are numerical (like years) even without explicit DATE tags
            if any(t.like_num for t in doc_answer):
                return True
            expected_tags = ["DATE", "TIME"]

        # If we have an expectation, ensure the answer contains at least one matching entity type
        if expected_tags:
            answer_types = [ent.label_ for ent in doc_answer.ents]
            if not any(t in answer_types for t in expected_tags):
                return False

    return True

def check_information_gain(question, answer):
    """
    This function prevents the model from parroting by ensuring the answer adds new information beyond the question.
    input: question (str), answer (str)
    return: bool - True if new information is provided, False if the answer is just a subset of the question.
    """
    q_doc = nlp(question)
    a_doc = nlp(answer)
    
    # Extract lemmas, filtering out stopwords and punctuation
    q_lemmas = {t.lemma_.lower() for t in q_doc if not t.is_stop and not t.is_punct}
    a_lemmas = {t.lemma_.lower() for t in a_doc if not t.is_stop and not t.is_punct}
    
    # If the answer has no content lemmas (e.g. short functional words), give benefit of the doubt
    if len(a_lemmas) == 0:
        return True
        
    # If the answer lemmas are entirely contained within the question lemmas, it is parroting
    if a_lemmas.issubset(q_lemmas):
        return False
        
    return True

def squad_qa(data_filename):
    """
    This is the main QA pipeline function. It formats prompts, generates answers via LLM, and passes them through a voting filter.
    input: data_filename (str) - Path to the input CSV file containing SQuAD data.
    return: out_filename (str) - Path to the output CSV file containing final predictions written by this function.
    """
    print(f"Loading data from {data_filename}...")
    df = pd.read_csv(data_filename)

    final_answers = []

    # Base threshold for accepting the model's inherent confidence
    CONFIDENCE_THRESHOLD = 0.6

    print(f"Starting prediction on {len(df)} examples using {device}...")
    start_loop = time.time()

    for index, row in df.iterrows():
        context = row['context']
        question = row['question']

        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a precise extraction system.\n"
                    "Task: Extract the exact text span from the context that answers the question.\n"
                    "Rules:\n"
                    "1. Output ONLY the answer text. Do not frame it as a sentence.\n"
                    "2. The answer must contain only an EXACT SUBSTRING from the context.\n"
                    "3. Do not add punctuation (like periods) or filler words (like 'The answer is').\n"
                    "4. Verify that all numbers, units, and dates in the question match the context EXACTLY.\n"
                    "5. If the answer has a date or number, do NOT convert its format. Extract the exact string as it appears in the context."
                    f"6. If the answer is not in the context, output exactly '{NO_ANSWER_MARKER}'."
                )
            },

            # Example 1: Standard Extraction
            {
                "role": "user",
                "content": "Context: The sky is blue.\nQuestion: What color is the sky?\nAnswer:"
            },
            {"role": "assistant", "content": "blue"},

            # Example 2: Unanswerable trap
            {
                "role": "user",
                "content": "Context: The sky is blue.\nQuestion: When is the party?\nAnswer:"
            },
            {"role": "assistant", "content": "NO ANSWER"},
            
            # Example 3: Near miss hallucination trap
            {
                "role": "user",
                "content": "Context: The region spans from Pacific Ocean islands to interior valleys.\nQuestion: What ocean has interior valleys?\nAnswer:"
            },
            {"role": "assistant", "content": "NO ANSWER"},

            {
                "role": "user", 
                "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            }
        ]

        # Tokenize the chat template into model readable inputs
        encoded_output = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )

        # Handle different tokenization object return types gracefully
        if isinstance(encoded_output, dict) or hasattr(encoded_output, 'keys'):
            input_ids = encoded_output['input_ids'].to(model.device)
            attention_mask = encoded_output['attention_mask'].to(model.device)
        else:
            input_ids = encoded_output.to(model.device)
            attention_mask = torch.ones_like(input_ids).to(model.device)

        # LLM generation phase
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=30, # Cap generation length for concise SQuAD answers
                min_new_tokens=1,
                do_sample=False, # Use greedy decoding (deterministic) instead of sampling
                repetition_penalty=1.1,
                return_dict_in_generate=True,
                output_scores=True, # Required for confidence calculation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode model output tokens back into strings
        generated_tokens = outputs.sequences[0][input_ids.shape[1]:]
        answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Calculate model confidence score from transition probabilities
        try:
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            generated_log_probs = transition_scores[0]
            confidence_score = torch.exp(generated_log_probs.sum()).item()
        except:
            confidence_score = 1.0

        final_decision = answer_text

        # Deterministic veto phase - Immediate rejection if underlying confidence is low, or if strict Regex checks fail
        if (confidence_score < CONFIDENCE_THRESHOLD) or \
            (final_decision.lower() not in context.lower()) or \
            (not check_unit_consistency(question, context)) or \
                (not check_exact_numbers(question, context)) or \
                    (not check_year_consistency(question, final_decision)):
            final_decision = NO_ANSWER_MARKER
        
        # Weighted voting phase - Only run deeper semantic checks if the answer survived the veto
        if final_decision != NO_ANSWER_MARKER:
            score = 0
            
            # Factor 1: LLM Native Confidence
            if confidence_score > 0.8:
                score += 3
            elif confidence_score > 0.5:
                score += 1
                
            # Factor 2: Parroting Check
            if check_information_gain(question, final_decision):
                score += 1
            else:
                score -= 2
                
            # Factor 3: Deep Semantic Similarity (Highest Weight)
            if check_SBERT_cos_sim(context, question, final_decision):
                score += 3
            else:
                score -= 3

            # Factor 4: Linguistic Type Consistency
            if check_NER_type(question, final_decision):
                score += 1
            else:
                score -= 1
                    
            # Final Evaluation: Must meet score threshold to avoid being marked as a hallucination
            if score < 4:
                final_decision = NO_ANSWER_MARKER

        final_answers.append(final_decision)

        # Console logging for progress tracking
        if index > 0 and index % 10 == 0:
            elapsed = time.time() - start_loop
            avg_per_q = elapsed / (index + 1)
            remaining_time = (len(df) - index) * avg_per_q
            print(
                f"Processed {index}/{len(df)}... Avg time per Q: {avg_per_q:.2f}s. Est. remaining: {remaining_time / 60:.1f} min")

    # Save the results
    df['final answer'] = final_answers
    out_filename = data_filename.replace('.csv', '-results.csv')
    df.to_csv(out_filename, index=False)

    print(f'Final answers recorded into {out_filename}')
    return out_filename

if __name__ == '__main__':
    start_time = time.time()

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    data = pd.read_csv(config['data'])
    sample = data.sample(n=config['sample_for_solution'])  # for grading will be replaced with 'sample_for_grading'
    sample_filename = config['data'].replace('.csv', '-sample.csv')
    sample.to_csv(sample_filename, index=False)

    out_filename = squad_qa(sample_filename)  # todo: the function you implement

    eval_out = evaluate_results(out_filename, final_answer_column='final answer')
    eval_out_list = [str((k, round(v, 3))) for (k, v) in eval_out.items()]
    print('\n'.join(eval_out_list))

    elapsed_time = time.time() - start_time
    print(f"time: {elapsed_time: .2f} sec")
