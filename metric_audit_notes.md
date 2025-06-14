# NLP_OPS.ipynb Metric Code Audit - Sections 7-9

## SECTION 7: BLEU Score Calculation (Lines 334-377)

### Execution Status: ✅ EXECUTED
- Cell has execution_count: 8
- Shows output with calculated BLEU scores

### Variable Name Issues:
- **CRITICAL**: `bleu_scores` variable is created but `similarities` variable is NOT overwritten in this section
- No variable naming conflicts detected in this section

### Logic Bugs:
- **CRITICAL LOGIC BUG**: BLEU score implementation is INCORRECT
  - Lines 366-367: `sentence_bleu([nltk.word_tokenize(search_term.lower())], nltk.word_tokenize(title.lower()))`
  - **BUG**: BLEU expects the reference to be a LIST OF TOKEN LISTS, but here it's passed as a single list of tokens
  - **CORRECT FORMAT**: Should be `sentence_bleu([[tokens]], candidate_tokens)` where reference is wrapped in double brackets
  - Current implementation passes `[tokens]` instead of `[[tokens]]`

### Other Issues:
- BLEU scores are extremely low (mostly 0.000, max 0.019) indicating the metric is not working effectively
- This is likely due to the logic bug above

---

## SECTION 8: METEOR Score Calculation (Lines 384-408)

### Execution Status: ❌ NEVER EXECUTED
- Cell has execution_count: null
- No output shown - this section was never run

### Variable Name Issues:
- **POTENTIAL ISSUE**: If executed, would create `meteor_scores` variable
- No variable naming conflicts, but section not executed so impact unknown

### Logic Bugs:
- **CRITICAL LOGIC BUG**: METEOR score implementation is INCORRECT
  - Lines 397-399: `meteor_score([nltk.word_tokenize(search_term.lower())], nltk.word_tokenize(title.lower()))`
  - **BUG**: Same issue as BLEU - METEOR expects reference to be a LIST OF TOKEN LISTS
  - **CORRECT FORMAT**: Should be `meteor_score([[tokens]], candidate_tokens)`
  - Current implementation passes `[tokens]` instead of `[[tokens]]`

### Other Issues:
- Section never executed, so no practical impact on results yet
- Would fail if executed due to incorrect parameter format

---

## SECTION 9: CIDEr Score Calculation (Lines 415-488)

### Execution Status: ❌ NEVER EXECUTED
- Cell has execution_count: null
- No output shown - this section was never run

### Variable Name Issues:
- **POTENTIAL ISSUE**: If executed, would create `cider_scores` variable
- No variable naming conflicts, but section not executed so impact unknown

### Logic Bugs:
- **MAJOR PERFORMANCE BUG**: CIDEr implementation recomputes IDF each call
  - Lines 444-451: IDF calculation is performed inside the `compute_cider_score` function
  - **BUG**: For each job title comparison, the function recalculates IDF for ALL job titles
  - **PERFORMANCE IMPACT**: O(n²) complexity instead of O(n)
  - **SOLUTION**: IDF should be precomputed once and reused

### Detailed CIDEr Issues:
1. **IDF Recomputation Bug**:
   - Lines 444-451: `all_job_ngrams` is computed fresh for every single job title
   - This means for 100 job titles, IDF is calculated 100 times identically
   - Should compute once and pass as parameter

2. **Inefficient N-gram Processing**:
   - Lines 447-450: N-grams are recalculated for all job titles on each call
   - Should be precomputed and cached

### Other Issues:
- Section never executed, so no practical impact on results yet
- Algorithm is correct in principle but very inefficient

---

## SECTION 10: Comprehensive Comparison (Lines 495-561)

### Execution Status: ❌ NEVER EXECUTED
- Cell has execution_count: null
- This section references variables from sections 8-9 that were never executed

### Variable Name Issues:
- **CRITICAL VARIABLE OVERWRITING BUG FOUND**:
  - Lines 511-514: Multiple assignments to `similarities` variable!
  - `'TF-IDF + Cosine': (np.argsort(similarities)[::-1], similarities)`
  - `'Word2Vec + Cosine': (np.argsort(similarities)[::-1], similarities)`
  - `'GloVe + Cosine': (np.argsort(similarities)[::-1], similarities)`
  - `'FastText + Cosine': (np.argsort(similarities)[::-1], similarities)`
  - **BUG**: All four methods reference the SAME `similarities` variable
  - **IMPACT**: Only the last computed similarities (FastText) would be used for all four methods
  - **SOLUTION**: Should use method-specific variable names (e.g., `tfidf_similarities`, `w2v_similarities`, etc.)

### Logic Bugs:
- References to `meteor_scores` and `cider_scores` that don't exist (lines 515-517)
- Would fail if executed due to undefined variables

---

## SUMMARY OF CRITICAL ISSUES:

### 1. Variable Overwriting Issues:
- ✅ **CONFIRMED**: `similarities` variable overwritten in section 10
- Multiple similarity methods all reference the same variable
- Only FastText similarities would be used for all methods

### 2. Never Executed Cells:
- ✅ **CONFIRMED**: METEOR section (8) never executed
- ✅ **CONFIRMED**: CIDEr section (9) never executed
- ✅ **CONFIRMED**: Comparison section (10) never executed

### 3. Logic Bugs:
- ✅ **CONFIRMED**: BLEU fed token list instead of list of token lists
- ✅ **CONFIRMED**: METEOR fed token list instead of list of token lists
- ✅ **CONFIRMED**: CIDEr recomputes IDF on each call (O(n²) performance)

### 4. Additional Issues Found:
- Section 10 references undefined variables (`meteor_scores`, `cider_scores`)
- Poor separation of concerns (IDF calculation mixed with scoring)
- Inefficient caching strategies

## REFACTOR PRIORITIES:
1. Fix BLEU/METEOR parameter format (wrap reference in double brackets)
2. Fix variable overwriting in similarities comparison
3. Optimize CIDEr IDF calculation (compute once, reuse)
4. Execute and test sections 8-10
5. Implement proper variable naming conventions for different similarity methods

