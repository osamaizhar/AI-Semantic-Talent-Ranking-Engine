from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import math
from collections import defaultdict, Counter

smoothie = SmoothingFunction().method4

def bleu_score(reference, candidate):
    return sentence_bleu([word_tokenize(reference.lower())],
                         word_tokenize(candidate.lower()),
                         smoothing_function=smoothie)

def meteor(reference, candidate):
    return meteor_score([reference.lower()], candidate.lower())

class CiderScorer:
    def __init__(self, corpus, n=4):
        self.n = n
        self.idf = self._build_idf(corpus)

    def _ngrams(self, tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def _build_idf(self, corpus):
        """Build IDF scores for n-grams in the corpus"""
        # Count document frequency for each n-gram
        df = defaultdict(int)
        total_docs = len(corpus)
        
        for doc in corpus:
            tokens = word_tokenize(doc.lower())
            # Get unique n-grams for this document
            doc_ngrams = set()
            for n in range(1, self.n + 1):
                doc_ngrams.update(self._ngrams(tokens, n))
            
            # Count document frequency
            for ngram in doc_ngrams:
                df[ngram] += 1
        
        # Calculate IDF scores
        idf = {}
        for ngram, freq in df.items():
            idf[ngram] = math.log(total_docs / freq)
        
        return idf

    def score(self, reference, candidate):
        """Calculate CIDEr score using pre-computed IDF and cosine similarity"""
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        
        # Get n-gram vectors for reference and candidate
        ref_vector = self._get_ngram_vector(ref_tokens)
        cand_vector = self._get_ngram_vector(cand_tokens)
        
        # Calculate cosine similarity
        return self._cosine_similarity(ref_vector, cand_vector)
    
    def _get_ngram_vector(self, tokens):
        """Get TF-IDF weighted n-gram vector"""
        vector = defaultdict(float)
        
        for n in range(1, self.n + 1):
            ngrams = self._ngrams(tokens, n)
            ngram_counts = Counter(ngrams)
            
            for ngram, count in ngram_counts.items():
                if ngram in self.idf:
                    # TF-IDF weighting
                    tf = count / len(ngrams) if len(ngrams) > 0 else 0
                    vector[ngram] = tf * self.idf[ngram]
        
        return vector
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        # Get all unique keys
        all_keys = set(vec1.keys()) | set(vec2.keys())
        
        if not all_keys:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vec1[key] * vec2[key] for key in all_keys)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(vec1[key] ** 2 for key in all_keys))
        mag2 = math.sqrt(sum(vec2[key] ** 2 for key in all_keys))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)

