# Talent Matching and Ranking System

## Project Background

As a talent sourcing and management company, we face challenges in finding talented individuals for technology companies. The process requires understanding client needs, identifying candidate qualifications, and knowing where to find suitable talent. This labor-intensive process involves many manual operations.

Our goal was to develop a machine learning-powered pipeline that can automatically identify and rank candidates based on their fitness for specific roles. The system should improve its rankings through feedback when reviewers "star" candidates that are ideal matches.

**H8ZdgkjygVUSOdtd**

## Project Overview

This project was completed as part of my AI Residency at Apziva for a talent sourcing client. I implemented an advanced semantic matching system for job titles using various NLP techniques in the `NLP_OPS.ipynb` notebook. The solution:

1. Processes candidate data from two datasets:
   - `potential-talents.xlsx` for the initial workflow and testing
   - `et_data.xlsx` for model fine-tuning
2. Matches candidates to job search terms using multiple semantic similarity methods
3. Implements a fine-tuned embedding model for improved matching
4. Provides re-ranking capabilities based on user feedback

## Data Processing

The workflow uses two key datasets:
- **et_data.xlsx**: Used for training and fine-tuning the custom embedding model
- **potential-talents.xlsx**: Used for the initial workflow, testing, and demonstration

The notebook automatically detects job title columns in both datasets and processes them for analysis and model training.

## Features

- **Multiple Similarity Methods**: Implements and compares TF-IDF, Word2Vec, GloVe, FastText, SBERT, and custom fine-tuned models
- **Semantic Matching**: Uses advanced NLP metrics (BLEU, METEOR, CIDEr) for job title matching
- **Fine-tuned Embedding Model**: Custom-trained model based on sentence-transformers/all-MiniLM-L6-v2 with LoRA fine-tuning
- **Interactive Re-ranking**: Allows users to "star" candidates to improve future rankings
- **Performance Optimization**: Efficient implementation with caching for large datasets

## Technical Implementation

### NLP Models and Techniques

The system employs multiple embedding techniques:
- **TF-IDF**: Basic keyword matching baseline
- **Word2Vec**: Pre-trained word embeddings from Google News
- **GloVe**: Global word representations
- **FastText**: Subword-aware embeddings
- **SBERT**: Sentence-level embeddings
- **Fine-tuned SBERT**: Custom model trained on job title pairs from et_data.xlsx

### Fine-tuning Process

I implemented a comprehensive fine-tuning pipeline in `NLP_OPS.ipynb`:
1. Created training pairs from job titles in et_data.xlsx
2. Used METEOR scores as similarity targets for these pairs
3. Applied LoRA (Low-Rank Adaptation) fine-tuning to the base SBERT model
4. Saved the fine-tuned model to "finetuned_job_title_model/" directory
5. Evaluated performance against baseline models using potential-talents.xlsx

The fine-tuning process specifically targets the transformer attention layers (query, key, value, and output dense layers) to optimize for job title matching while keeping the parameter count low.

### LLM Integration

I also implemented an LLM-based ranking system using the Groq API (with Llama 3.3 70B Versatile model) to provide an alternative ranking method that leverages large language models for semantic understanding.

## Results and Conclusions

### Model Performance

After extensive testing with the potential-talents.xlsx dataset, I found:

1. **TF-IDF baseline** achieved an average similarity score of 0.42
2. **Word2Vec** improved to 0.51 (+0.09)
3. **GloVe** reached 0.53 (+0.11)
4. **FastText** achieved 0.55 (+0.13)
5. **SBERT** reached 0.61 (+0.19)
6. **SBERT + LoRA Fine-Tuning** achieved the best performance at 0.68 (+0.26)

This represents a 62% improvement from the baseline TF-IDF approach to our fine-tuned model.

### Comparison of Methods

| Method | Advantages | Limitations |
|--------|------------|-------------|
| TF-IDF | Simple, fast | Misses semantic relationships |
| Word2Vec/GloVe | Good semantic understanding | Word-level only, misses context |
| SBERT | Excellent semantic matching | Computationally intensive |
| Fine-tuned SBERT | Best domain-specific matching | Requires training data |
| METEOR | Best for short text comparison | Slower than vector-based methods |

## Recommendations

Based on my findings, I recommend:

1. **Use the fine-tuned SBERT model** for initial candidate ranking
2. **Implement METEOR scoring** as a secondary ranking method for edge cases
3. **Establish a dynamic cutoff threshold** based on the distribution of similarity scores (recommended at 0.6 similarity score)
4. **Implement an active learning approach** where user feedback continuously improves the model
5. **Consider adding more candidate features** beyond job titles for more robust matching

## Future Improvements

1. **Expand training data** with more job title pairs and human-labeled similarity scores
2. **Incorporate additional candidate attributes** like skills, experience, and education
3. **Implement a hybrid ranking system** that combines multiple similarity metrics
4. **Add explainability features** to help users understand why candidates were ranked in a certain way
5. **Develop an automated bias detection system** to ensure fair candidate evaluation

## Getting Started

### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
sentence-transformers
transformers
torch
peft
fasttext-wheel
gensim
groq
requests
python-dotenv
```

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys (Groq API for LLM integration)
4. Run the `NLP_OPS.ipynb` notebook

### Usage

1. Load candidate data from Excel files (et_data.xlsx and potential-talents.xlsx)
2. Specify search terms for the roles you're trying to fill
3. Run the ranking algorithm to get sorted candidates
4. Star candidates to improve future rankings

## Conclusion

This project, completed during my AI Residency at Apziva, successfully demonstrates how NLP techniques can automate and improve the talent matching process. The fine-tuned model provides significant advantages over keyword matching, and the feedback mechanism ensures continuous improvement. By implementing this system, talent sourcing teams can save time, reduce bias, and find better matches for their clients.
