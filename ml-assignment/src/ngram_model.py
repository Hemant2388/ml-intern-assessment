import random
from collections import Counter, defaultdict
import re

class TrigramModel:
    START = "<s>"
    END = "</s>"
    UNK = "<UNK>"
    def __init__(self, min_count = 2):
        """
        Initializes the TrigramModel.
        """
        self.min_count = min_count

        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(lambda: defaultdict(Counter))

        self.vocab = set()
        self.trained = False
    
    def _clean_and_tokenize(self, text):
        text = text.lower()

        sentences = re.split(r'(?<=[.!?])\s+', text)

        word_re = re.compile(r"[a-z]+(?:'[a-z]+)?")
        tokenized = []

        for sent in sentences:
            words = word_re.findall(sent)
            if words:
                tokenized.append(words)

        return tokenized
    
    def _replace_rare_words(self, sentences, freq):
        def unk_if_rare(w):
            return w if freq[w] >= self.min_count else self.UNK

        new_sentences = []
        for s in sentences:
            new_sentences.append([unk_if_rare(w) for w in s])
        return new_sentences

    def _pad_sentences(self, sentences):
        padded = []
        for s in sentences:
            padded.append([self.START, self.START] + s + [self.END])
        return padded

    def _count_ngrams(self, padded_sentences):
        for sent in padded_sentences:
            # unigrams
            for w in sent:
                self.unigram_counts[w] += 1

            # bigrams
            for i in range(len(sent) - 1):
                w1, w2 = sent[i], sent[i+1]
                self.bigram_counts[w1][w2] += 1

            # trigrams
            for i in range(len(sent) - 2):
                w1, w2, w3 = sent[i], sent[i+1], sent[i+2]
                self.trigram_counts[w1][w2][w3] += 1

    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        sentences = self._clean_and_tokenize(text)

        if not sentences:
            self.trained = False
            return

        freq = Counter()
        for s in sentences:
            freq.update(s)

        sentences = self._replace_rare_words(sentences, freq)
        padded = self._pad_sentences(sentences)

        self._count_ngrams(padded)
        self.vocab = set(self.unigram_counts.keys())
        self.trained = True

    def _sample(self, counter, temperature=1.0):
        """Sample a word using probabilities with optional temperature scaling."""
        if not counter:
            raise ValueError("Counter is empty, cannot sample.")

        words = list(counter.keys())
        counts = list(counter.values())
        total = sum(counts)
        probs = [c / total for c in counts]

        # Apply temperature scaling
        if temperature is not None and temperature > 0 and temperature != 1.0:
            inv_temp = 1.0 / temperature
            probs = [p ** inv_temp for p in probs]
            s = sum(probs)
            probs = [p / s for p in probs]
            weights = probs
        else:
            weights = counts  # default behavior is normal sampling

        return random.choices(words, weights=weights, k=1)[0]
    def generate(self, max_length=50, temperature=1.0):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        if not self.trained or not self.unigram_counts:
            return ""  # No knowledge -> return empty string as required by tests

        output = []
        w1, w2 = self.START, self.START

        for _ in range(max_length):
            next_word = None

            # 1) Try trigram first
            if w1 in self.trigram_counts and w2 in self.trigram_counts[w1]:
                cnt = self.trigram_counts[w1][w2]
                if cnt:
                    next_word = self._sample(cnt, temperature)

            # 2) Backoff to bigram
            if next_word is None:
                if w2 in self.bigram_counts:
                    cnt = self.bigram_counts[w2]
                    if cnt:
                        next_word = self._sample(cnt, temperature)

            # 3) Backoff to unigram (avoid <s> inside sentence)
            if next_word is None:
                uni = {w: c for w, c in self.unigram_counts.items() if w != self.START}
                next_word = self._sample(uni, temperature)

            # End token stops generation
            if next_word == self.END:
                break

            output.append(next_word)
            w1, w2 = w2, next_word  # shift context

        return " ".join(output)
