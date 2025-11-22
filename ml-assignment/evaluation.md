# Evaluation of Trigram Language Model Design

## 1️. Introduction  
This project implements a Trigram Language Model (N = 3) from scratch.  
The model learns word transition patterns from text and generates new sentences using probabilistic sampling techniques.

## 2️. Text Cleaning & Tokenization
- Text is converted to lowercase for normalization.
- Sentences are separated using punctuation markers (`.`, `!`, `?`).
- Tokenization uses a regex that preserves contractions: [a-z]+(?:'[a-z]+)?
- Words like “don’t” and “I’m” remain meaningful instead of splitting into broken tokens.

## 3️. Handling Unknown Words 
To reduce sparsity and improve robustness:

- Word frequencies are counted initially.
- Words appearing fewer than 2 times are replaced by `<UNK>`. 
- Allows model to generalize better

## 4️. Sentence Padding
Each sentence is padded to support context prediction at the beginning and end: <s> <s> w1 w2 ... wn </s>

- Ensures trigram prediction works from the first real word  
- Allows `<END>` token to stop generation cleanly

## 5️. N-Gram Count Storage  
Efficient Python collections are used:

| N-Gram | Structure |
|--------|-----------|
| Unigram | `Counter()` | 
| Bigram | `defaultdict(Counter)` | 
| Trigram | `defaultdict(lambda: defaultdict(Counter))` | 

## 6️. Generation Strategy (Backoff Model)
To generate each next token:

- Trigram: P(wᵢ | wᵢ₋₂, wᵢ₋₁)
- If missing → Bigram: P(wᵢ | wᵢ₋₁)
- If still missing → Unigram: P(wᵢ)
- Ensures the model never gets stuck  
- Allows generation even from unseen contexts

## 7️. Probabilistic Sampling + Temperature
Sampling is probability-driven, not greedy:

1. Convert counts → probabilities  
2. Apply temperature scaling: pᵢ' = (pᵢ)^(1/T)
3. normalize → sample
4. The default temperature value is 1.5 which ensures the generated output is creative. 

## 8️. Edge Case Handling
- If input text is empty or contains no valid tokens:
  - `fit()` marks model as untrained
  - `generate()` returns an empty string
- Ensures unit tests pass  
- Avoids runtime errors

## Conclusion

The final implementation meets project requirements while producing coherent, meaningful generated text that reflects the style of its training data.
