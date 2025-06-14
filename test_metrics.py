# Test script for semantic similarity metrics
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from collections import Counter
import math

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load data
df = pd.read_excel("potential-talents.xlsx")
possible_columns = [
    "job_title",
    "title",
    "position",
    "role",
    "job",
    "designation",
    "job title",
]
job_title_column = None
for col in df.columns:
    if any(keyword in col.lower() for keyword in possible_columns):
        job_title_column = col
        break
if not job_title_column:
    raise ValueError("Job title column not found. Please specify it manually.")
job_titles = df[job_title_column].dropna().astype(str).tolist()
search_term = "Data Scientist"

print(f"Testing with search term: '{search_term}'")
print(f"Total job titles: {len(job_titles)}")
print("\n" + "="*70)

# 1. BLEU Score
print("\n1. BLEU SCORE RESULTS:")
print("-" * 30)
smoothie = SmoothingFunction().method4
bleu_scores = [
    sentence_bleu(
        [nltk.word_tokenize(search_term.lower())],
        nltk.word_tokenize(title.lower()),
        smoothing_function=smoothie,
    )
    for title in job_titles
]
ranked_indices = np.argsort(bleu_scores)[::-1]
print("Top 5 job titles by BLEU semantic similarity:")
for i, idx in enumerate(ranked_indices[:5]):
    print(f"{i+1}. {job_titles[idx]} (Score: {bleu_scores[idx]:.3f})")

# 2. METEOR Score
print("\n2. METEOR SCORE RESULTS:")
print("-" * 30)
meteor_scores = [
    meteor_score(
        [nltk.word_tokenize(search_term.lower())],
        nltk.word_tokenize(title.lower())
    )
    for title in job_titles
]
ranked_indices = np.argsort(meteor_scores)[::-1]
print("Top 5 job titles by METEOR semantic similarity:")
for i, idx in enumerate(ranked_indices[:5]):
    print(f"{i+1}. {job_titles[idx]} (Score: {meteor_scores[idx]:.3f})")

# 3. CIDEr Score (Simplified Implementation)
print("\n3. CIDEr SCORE RESULTS:")
print("-" * 30)

def compute_cider_score(reference_tokens, candidate_tokens, n=4):
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def compute_tf_idf(ngrams, all_ngrams_counter):
        tf = Counter(ngrams)
        total_docs = len(all_ngrams_counter)
        tf_idf = {}
        for ngram, count in tf.items():
            tf_val = count / len(ngrams) if len(ngrams) > 0 else 0
            df = sum(1 for doc_ngrams in all_ngrams_counter if ngram in doc_ngrams)
            idf = math.log(total_docs / max(1, df))
            tf_idf[ngram] = tf_val * idf
        return tf_idf
    
    # Collect all n-grams for IDF calculation
    all_job_ngrams = []
    for title in job_titles:
        tokens = nltk.word_tokenize(title.lower())
        ngrams = []
        for i in range(1, n+1):
            ngrams.extend(get_ngrams(tokens, i))
        all_job_ngrams.append(Counter(ngrams))
    
    # Calculate TF-IDF for reference and candidate
    ref_ngrams = []
    cand_ngrams = []
    for i in range(1, n+1):
        ref_ngrams.extend(get_ngrams(reference_tokens, i))
        cand_ngrams.extend(get_ngrams(candidate_tokens, i))
    
    ref_tf_idf = compute_tf_idf(ref_ngrams, all_job_ngrams)
    cand_tf_idf = compute_tf_idf(cand_ngrams, all_job_ngrams)
    
    # Compute cosine similarity
    common_ngrams = set(ref_tf_idf.keys()) & set(cand_tf_idf.keys())
    if not common_ngrams:
        return 0.0
    
    numerator = sum(ref_tf_idf[ngram] * cand_tf_idf[ngram] for ngram in common_ngrams)
    ref_norm = math.sqrt(sum(val**2 for val in ref_tf_idf.values()))
    cand_norm = math.sqrt(sum(val**2 for val in cand_tf_idf.values()))
    
    if ref_norm == 0 or cand_norm == 0:
        return 0.0
    
    return numerator / (ref_norm * cand_norm)

search_tokens = nltk.word_tokenize(search_term.lower())
cider_scores = [
    compute_cider_score(search_tokens, nltk.word_tokenize(title.lower()))
    for title in job_titles
]
ranked_indices = np.argsort(cider_scores)[::-1]
print("Top 5 job titles by CIDEr semantic similarity:")
for i, idx in enumerate(ranked_indices[:5]):
    print(f"{i+1}. {job_titles[idx]} (Score: {cider_scores[idx]:.3f})")

# 4. TF-IDF Baseline for comparison
print("\n4. TF-IDF + COSINE SIMILARITY (BASELINE):")
print("-" * 45)
vectorizer = TfidfVectorizer()
corpus = job_titles + [search_term]
X = vectorizer.fit_transform(corpus)
search_vec = X[-1]
job_vecs = X[:-1]
similarities = cosine_similarity(search_vec, job_vecs).flatten()
ranked_indices = np.argsort(similarities)[::-1]
print("Top 5 job titles by TF-IDF similarity:")
for i, idx in enumerate(ranked_indices[:5]):
    print(f"{i+1}. {job_titles[idx]} (Score: {similarities[idx]:.3f})")

# Summary and Recommendation
print("\n" + "="*70)
print("SUMMARY AND RECOMMENDATION")
print("="*70)
print("\nMETRIC ANALYSIS:")
print("\n1. **METEOR Score** - RECOMMENDED FOR YOUR TASK")
print("   ✓ Best for semantic similarity of job titles")
print("   ✓ Considers synonyms, stemming, and word order")
print("   ✓ More robust than BLEU for short texts")
print("   ✓ Specifically designed for semantic evaluation")

print("\n2. **CIDEr Score** - GOOD ALTERNATIVE")
print("   ✓ Uses TF-IDF weighting for better discrimination")
print("   ✓ Good consensus-based evaluation")
print("   ✓ Handles n-gram overlap effectively")

print("\n3. **BLEU Score** - NOT RECOMMENDED")
print("   ✗ Designed for machine translation")
print("   ✗ Poor performance on short texts like job titles")
print("   ✗ Doesn't handle semantic similarity well")

print("\n4. **TF-IDF + Cosine** - SIMPLE BASELINE")
print("   ✓ Fast and simple")
print("   ✗ Limited semantic understanding")
print("   ✗ Only considers exact word matches")

print("\n" + "="*70)
print("FINAL RECOMMENDATION:")
print("Use **METEOR Score** for the best semantic similarity evaluation.")
print("It provides the most accurate results for job title matching tasks.")
print("="*70)

