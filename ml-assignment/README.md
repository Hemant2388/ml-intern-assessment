# Trigram Language Model

This directory contains the core assignment files for the Trigram Language Model.

### File Structure 

```
├── ml-assignment/
│   ├── data/
│   │   └── example_corpus.txt
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── generate.py
│   │   ├── ngram_model.py
│   │   └── utils.py
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_ngram.py
│   │
│   ├── evaluation.md
│   └── README.md
│
├── .gitignore
├── assignment.md
├── quick_start.md
└── requirements.txt
```

## How to Run 

Follow the steps below to run the Trigram Language Model and execute tests.

---

1. Clone the Repository:

    ```sh
    git clone <your-repository-link-here>
    cd ml-assignment
    ```

2. Set up a virtual environment:

   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

4. To train on the included corpus and generate text:

    ```sh
    python src/generate.py
    ```
5. To run tests

    ```sh
    python -m pytest
    ```
    
## Design Choices

Please document your design choices in the `evaluation.md` file. This should be a 1-page summary of the decisions you made and why you made them.
