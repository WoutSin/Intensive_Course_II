# Runic Inscription Emendation

This Python script is designed to reconstruct incomplete runic inscriptions from the Medieval and Viking Age periods using n-gram probabilities and a modified Minimum Edit Distance algorithm. The script processes a dataset of runic inscriptions, extracts n-gram probabilities and generates potential candidates for missing or incomplete tokens in the inscriptions.

## Features

- Reads runic inscriptions from a text file and metadata from an Excel file.
- Filters the dataset to include only inscriptions from the Medieval and Viking Age periods.
- Tokenizes the inscriptions and tags tokens as complete (`<com>`), incomplete (`<inc>`), or missing (`<mis>`).
- Extracts unigram, bigram and trigram probabilities from the training data.
- Generates potential candidates for incomplete tokens based on the n-gram probabilities and context.
- Filters out the most likely candidates using a modified MED algorithm
- Evaluates the performance of the reconstruction using prediction coverage, accuracy and Mean Reciprocal Rank metrics.

## Usage

1. Ensure that you have the required dependencies installed (e.g., pandas, numpy, tqdm, sklearn).
2. Place the runic inscription text file (RUNTEXTX.txt) and the metadata Excel file (RUNDATA.xls) in the same directory as the script.
3. Run the script: `python Sinnaeve_Wout_IC2_Code.py`
4. The script will process the data, extract n-gram probabilities, generate candidates for a synthetic test set and evaluate the performance for different values of k (maximum number of candidates considered).
5. The results will be saved in an Excel file (results_non_0.xlsx) in the same directory.

## Dependencies

- pandas
- pickle
- numpy
- tqdm
- sklearn

## Hyperparameters

The script includes the following hyperparameters that can be adjusted:

- `number_predictions`: The maximum number of predictions that the `get_best_candidates` function may return for each incomplete token. A higher value will increase accuracy, but also the amount of work required to manually evaluate the candidates.
- `unigram_candidates_dict`: If set to `None`, unigram candidates will not be considered. If set to `unigram_candidates_dict`, unigram candidates will be considered (this is the recommended setting).
- `maximum_score`: The maximum allowed modified Minimum Edit Distance between the actual token and the candidate token. A higher value will increase the recall but may decrease the accuracy.

## Notes

- The script generates a synthetic test set by altering complete inscriptions from the original dataset. The test set is used for evaluation purposes.
- The script saves the extracted n-gram probabilities to a text file (n-gram_probabilities.txt) and the n-gram tokens to pickle files (unigram_tokens.pkl, bigram_tokens.pkl, trigram_tokens.pkl).
