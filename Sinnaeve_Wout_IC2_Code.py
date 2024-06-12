import pandas as pd
import pickle
import numpy as np
import random
import copy
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def read_runestones(file_path):
    """
    Input:
        file_path: path to a .txt file containing runic inscriptions
    Output:
        lines: list of runic inscriptions
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


def read_excel_to_dataframe(file_path):
    """
    Input:
        file_path: path to an Excel file containing metadata on runic inscriptions
    Output:
        df: dataframe containing metadata on runic inscriptions
    """
    df = pd.read_excel(file_path)
    return df


def filter_dataframe(df):
    """
    Input:
        df: dataframe containing metadata on the runic inscriptions
    Output:
        df: dataframe containing metadata on runic inscriptions from the Medieval and Viking Age period
    """
    df["Period/Datering"] = df["Period/Datering"].fillna("NaN")
    df = df[
        df["Period/Datering"].str.startswith("M")
        | df["Period/Datering"].str.startswith("V")
    ]
    return df


def filter_inscriptions(df, inscriptions):
    """
    Input:
        df: dataframe containing metadata on the runic inscriptions
        inscriptions: list of runic inscriptions
    Output:
        filtered_inscriptions: selection of Medieval and Viking Age runic inscriptions
    """
    filtered_inscriptions = []
    for signum in df["Signum"]:
        for inscription in inscriptions:
            if inscription.startswith(signum):
                filtered_inscription = inscription[len(signum) :].strip()
                filtered_inscriptions.append(filtered_inscription)

    filtered_inscription = list(set(filtered_inscriptions))

    return filtered_inscriptions


def remove_punctuation(input_string, punctuation):
    """
    Input:
        input_string: runic sequence (string)
        punctuation: string of non-alpabetical characters to be removed
    Output:
        no_punct: runic sequence without unnecessary punctuation (string)
    """
    translator = str.maketrans("", "", punctuation)
    no_punct = input_string.translate(translator)

    return no_punct


def print_non_alpha(runic_list):
    """
    Input:
        runic_list: list of runic inscriptions
    Output:
        non_alpha_chars: string of non-alpabetical characters to be removed
    """
    non_alpha_chars = set()
    for item in runic_list:
        non_alpha_chars.update(
            [
                char
                for char in item
                if not char.isalnum() and char not in ["-", "…", " "]
            ]
        )

    return "".join(non_alpha_chars)


def tokenize_text(input_string):
    """
    Input:
       input_string: runic sequence (string)
    Output:
       list of tokens in the runic sequence
    """
    return [token for token in input_string.split(" ") if token]


def get_tags(runic_list):
    """
    Input:
        runic_list: tokenized runic sequence
    Output:
        output_list: tuple containing tokens and their respective tags
    Tags:
        <com>: complete tokens, no characters are missing
        <inc>: incomplete tokens, one or multiple characters are missing
        <mis>: missing tokens, all characters are missing
    Characters:
        - denotes a one missing character
        … denotes one or more missing characters
        Example: "stein", "st--n", "st…n"
    """
    special_chars = ["-", "…"]
    output_list = []

    for item in runic_list:
        if any(char in item for char in special_chars) and any(
            char not in special_chars for char in item
        ):
            output_list.append((item, "<inc>"))
        elif all(char in special_chars for char in item):
            output_list.append((item, "<mis>"))
        else:
            output_list.append((item, "<com>"))

    return output_list


def split_data(tagged_sequences):
    """
    Input:
        tagged_sequences: list containing runic sequences as (token, tag) tuples
    Output:
        train: selection of train data with size sequences - 100
        test: selection of test data with size 100
    """
    train, test = train_test_split(tagged_sequences, test_size=0.05, random_state=27)

    return train, test


def extract_ngram_probabilities(tagged_sequences):
    """
    Input:
        tagged_sequences: list containing runic sequences as (token, tag) tuples
    Output:
        unigrams: dictionary with key = one token and value = probability of token
        bigrams: dictionary with key = two tokens and value = probability of token 2 given token 1
        trigrams: dictionary with key = three tokens and value = probability of token 3 given token 1 and token 2
    """
    unigrams = defaultdict(int)
    bigrams = defaultdict(Counter)
    trigrams = defaultdict(Counter)

    for tagged_sequence in tagged_sequences:
        com_items = []

        for i in range(len(tagged_sequence)):
            if tagged_sequence[i][1] == "<com>":
                com_items.append(tagged_sequence[i][0])
                unigrams[tagged_sequence[i][0]] += 1
            else:
                com_items.append(None)

        for i in range(len(com_items)):
            if (
                i < len(com_items) - 1
                and com_items[i] is not None
                and com_items[i + 1] is not None
            ):
                bigrams[com_items[i]][com_items[i + 1]] += 1
            if (
                i < len(com_items) - 2
                and com_items[i] is not None
                and com_items[i + 1] is not None
                and com_items[i + 2] is not None
            ):
                trigrams[(com_items[i], com_items[i + 1])][com_items[i + 2]] += 1

    total_unigrams = sum(unigrams.values())
    for word in unigrams:
        unigrams[word] /= total_unigrams

    for word in bigrams:
        total_count = sum(bigrams[word].values())
        for next_word in bigrams[word]:
            bigrams[word][next_word] /= total_count

    for words in trigrams:
        total_count = sum(trigrams[words].values())
        for next_word in trigrams[words]:
            trigrams[words][next_word] /= total_count

    return unigrams, bigrams, trigrams


def save_probabilities_to_file(unigrams, bigrams, trigrams, filename):
    """
    Input:
        unigrams: dictionary with key = one token and value = probability of token
        bigrams: dictionary with key = two tokens and value = probability of token 2 given token 1
        trigrams: dictionary with key = three tokens and value = probability of token 3 given token 1 and token 2
        filename: name of the file to which the dictionaries should be saved
    Output:
        .txt file containing the n-gram probabilities
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Unigram Probabilities:\n")
        for word, probability in unigrams.items():
            f.write(f"{word}: {probability}\n")

        f.write("\nBigram Probabilities:\n")
        for word, next_words in bigrams.items():
            for next_word, probability in next_words.items():
                f.write(f"{word} {next_word}: {probability}\n")

        f.write("\nTrigram Probabilities:\n")
        for words, next_words in trigrams.items():
            for next_word, probability in next_words.items():
                f.write(f"{words[0]} {words[1]} {next_word}: {probability}\n")


def extract_unigram_tokens(unigrams_dictionary):
    """
    Input:
        unigrams_dictionary: dictionary with key = one token and value = probability of token
    Output:
        unigram_words: list containing (unigram, probability) tuples
    """
    unigram_words = []

    for word, probability in unigrams_dictionary.items():
        unigram_words.append((word, probability))

    return list(set(unigram_words))


def extract_bigram_tokens(bigrams_dictionary):
    """
    Input:
        bigrams: dictionary with key = two tokens and value = probability of token 2 given token 1
    Output:
        bigram_words: list containing (bigram, probability) tuples
    """
    bigram_words = []

    for word, next_words in bigrams_dictionary.items():
        for next_word, probability in next_words.items():
            bigram_words.append(((word, next_word), probability))

    return list(set(bigram_words))


def extract_trigram_tokens(trigrams_dictionary):
    """
    Input:
        trigrams: dictionary with key = three tokens and value = probability of token 3 given token 1 and token 2
    Output:
        trigram_words: list containing (trigram, probability) tuples
    """
    trigram_words = []

    for words, next_words in trigrams_dictionary.items():
        for next_word, probability in next_words.items():
            trigram_words.append(((words[0], words[1], next_word), probability))

    return list(set(trigram_words))


def extract_potential_tokens(tagged_sequences, unigrams, bigrams, trigrams):
    """
    Input:
        tagged_sequences: list containing runic sequences as (token, tag) tuples
        unigrams: list containing (unigram, probability) tuples
        bigrams: list containing (bigram, probability) tuples
        trigrams: list containing (trigram, probability) tuples
    Output:
        potential_tokens_from_unigrams: dictionary with key = position of the token saved as [sequence_index, token_index]
            and value = all unigrams and their probabilites as (unigram, probability)
        potential_tokens_from_bigrams: dictionary with key = position of the token saved as [sequence_index, token_index]
            and value = all possible bigrams, given the context of the token, and their probabilities as (bigram, probability)
        potential_tokens_from_trigrams: dictionary with key = position of the token saved as [sequence_index, token_index]
            and value = all possible trigrams, given the context of the token, and their probabilities as (trigram, probability)
    Example:
        Input:
            trigram: (ias satr aiftir, 0.0256)
            sequence 5, position 7: ('ias', '<com>')
            sequence 5, position 8: ('s--r', '<inc>')
            sequence 5, position 9: ('aiftir', '<com>')
        Output:
            potential_tokens_from_trigrams[5, 8] = [(satr, 0.0256)]
    """
    potential_tokens_from_unigrams = defaultdict(list)
    potential_tokens_from_bigrams = defaultdict(list)
    potential_tokens_from_trigrams = defaultdict(list)

    total_length = len(unigrams) + len(bigrams) + len(trigrams)
    pbar = tqdm(total=total_length, desc="Extracting potential tokens", ncols=80)

    # For each trigram, the code slides over the sequences with a window of size three tokens
    for trigram, probability in trigrams:
        pbar.update()

        for i, sequence in enumerate(tagged_sequences):
            for j in range(len(sequence) - 2):
                # The code checks whether the sequence contains two <com> tags (tokens used as reference) and one <inc> tag (token to predict)
                # Additionally, the code checks whether all tokens with a <com> tag in the window with position j also occur in the trigram at position j
                if (
                    all(
                        (
                            trigram[k] == sequence[j + k][0]
                            if sequence[j + k][1] == "<com>"
                            else True
                        )
                        for k in range(3)
                    )
                    and sum(sequence[j + k][1] == "<com>" for k in range(3)) == 2
                ):
                    for k in range(3):
                        if sequence[j + k][1] == "<inc>":
                            # If both conditions are met, the potential_token dictionary with key = position of an <inc> token is appended with the relevant
                            # candidate from the trigram
                            potential_tokens_from_trigrams[(i, j + k)].append(
                                (trigram[k], probability)
                            )

    # For each bigram, the code slides over the sequences with a window of size two tokens
    for bigram, probability in bigrams:
        pbar.update()

        for i, sequence in enumerate(tagged_sequences):
            for j in range(len(sequence) - 1):
                # The code checks whether the sequence contains one <com> tag (token used as reference) and one <inc> tag (token to predict)
                # Additionally, the code checks whether all tokens with a <com> tag in the window with position j also occur in the bigram at position j
                if (
                    all(
                        (
                            bigram[k] == sequence[j + k][0]
                            if sequence[j + k][1] == "<com>"
                            else True
                        )
                        for k in range(2)
                    )
                    and sum(sequence[j + k][1] == "<com>" for k in range(2)) == 1
                ):
                    for k in range(2):
                        if sequence[j + k][1] == "<inc>":
                            # If both conditions are met, the potential_token dictionary with key = position of an <inc> token is appended with
                            # the relevant candidate from the bigram
                            potential_tokens_from_bigrams[(i, j + k)].append(
                                (bigram[k], probability)
                            )

    # For unigrams, all unigrams and their probabilities are appended to potential_tokens_from_unigrams with key = position of an <inc> token
    for unigram, probability in unigrams:
        pbar.update()

        for i, sequence in enumerate(tagged_sequences):
            for j in range(len(sequence)):
                if sequence[j][1] == "<inc>":
                    potential_tokens_from_unigrams[(i, j)].append(
                        (unigram, probability)
                    )

    pbar.close()

    return (
        dict(potential_tokens_from_unigrams),
        dict(potential_tokens_from_bigrams),
        dict(potential_tokens_from_trigrams),
    )


def get_best_candidates(
    sequences,
    bigram_candidates_dict,
    trigram_candidates_dict,
    unigram_candidates_dict=None,
    maximum_score=0,
    number_predictions=float("inf"),
):
    """
    Input:
        sequences: list containing runic sequences as (token, tag) tuples
        bigram_candidates_dict: dictionary with key = position of the token and value = all possible bigrams in that position
        trigram_candidates_dict: dictionary with key = position of the token and value = all possible trigrams in that position
        unigram_candidates_dict: (optional variable) dictionary with key = position of the token and value = all possible unigrams
        maximum_score: the maximum allowed modified Minimum Edit Distance between the actual token and the candidate token
            e.g. token = r-sa, risa (MED = 0), raisa (MED = 1)
            0 = characters can only be added in positions where characters are known to be missing
            higher MED = higher recall, lower accuracy
        number_predictions: the maximum number of predictions that the function may return
    Output: best_candidates: dictionary with key = position of the token and value = top candidates for that position, ordered as follows:
            1. Trigram candidates > Bigram candidates (> Unigram candidates)
            2. Highest to lowest n-gram probability
    """
    best_candidates = {}

    total_length = sum(
        sum(1 for token_tag in sequence if token_tag[1] == "<inc>")
        for sequence in sequences
    )
    pbar = tqdm(total=total_length, desc="Extracting best candidates", ncols=80)

    for i, sequence in enumerate(sequences):
        for j, token_tag in enumerate(sequence):
            if sequence[j][1] == "<inc>":
                pbar.update()
                max_score_unigrams = []
                max_score_bigrams = []
                max_score_trigrams = []

                if unigram_candidates_dict and (i, j) in unigram_candidates_dict.keys():
                    for candidate, probability in unigram_candidates_dict[i, j]:
                        MED_score = min_edit_distance(token_tag[0], candidate)
                        if MED_score <= maximum_score:
                            max_score_unigrams.append(
                                (candidate, MED_score, probability)
                            )

                if (i, j) in bigram_candidates_dict.keys():
                    for candidate, probability in bigram_candidates_dict[i, j]:
                        MED_score = min_edit_distance(token_tag[0], candidate)
                        if MED_score <= maximum_score:
                            max_score_bigrams.append(
                                (candidate, MED_score, probability)
                            )

                if (i, j) in trigram_candidates_dict.keys():
                    for candidate, probability in trigram_candidates_dict[i, j]:
                        MED_score = min_edit_distance(token_tag[0], candidate)
                        if MED_score <= maximum_score:
                            max_score_trigrams.append(
                                (candidate, MED_score, probability)
                            )

                max_score_unigrams = list(set(max_score_unigrams))
                max_score_unigrams = sorted(
                    max_score_unigrams, key=lambda x: (x[1], -x[2])
                )
                max_score_bigrams = list(set(max_score_bigrams))
                max_score_bigrams = sorted(
                    max_score_bigrams, key=lambda x: (x[1], -x[2])
                )
                max_score_trigrams = list(set(max_score_trigrams))
                max_score_trigrams = sorted(
                    max_score_trigrams, key=lambda x: (x[1], -x[2])
                )

                potential_candidates_trigrams = [c[0] for c in max_score_trigrams]
                potential_candidates_bigrams = [c[0] for c in max_score_bigrams]
                potential_candidates_unigrams = [c[0] for c in max_score_unigrams]

                merged_candidates = (
                    potential_candidates_trigrams
                    + [
                        candidate
                        for candidate in potential_candidates_bigrams
                        if candidate not in potential_candidates_trigrams
                    ]
                    + [
                        candidate
                        for candidate in potential_candidates_unigrams
                        if candidate not in potential_candidates_bigrams
                        and candidate not in potential_candidates_trigrams
                    ]
                )
                best_candidates[i, j] = merged_candidates[:number_predictions]

    pbar.close()

    return best_candidates


def min_edit_distance(source, target):
    """
    Input:
        source: source token containing missing characters
            '-' single missing character
            '…' multiple missing characters
        target: potential candidate token
    Output:
        distance_matrix[-1][-1]: the bottem right value in the distance_matrix representing the final Minimum Edit Distance
    Modifications:
        for each '-' symbol, the function allows one free substitutions
        for each '…' symbol, the function allows multiple free substitutions/insertions
    """
    distance_matrix = np.zeros((len(source) + 1, len(target) + 1))

    for i in range(len(source) + 1):
        distance_matrix[i][0] = i
    for j in range(len(target) + 1):
        distance_matrix[0][j] = j

    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            if source[i - 1] == target[j - 1]:
                distance_matrix[i][j] = distance_matrix[i - 1][j - 1]

            elif source[i - 1] == "-":
                distance_matrix[i][j] = distance_matrix[i - 1][j - 1]

            # By allowing the model to also choose from the value on its left, it can make infinite free insertions for the '…' token
            elif source[i - 1] == "…":
                distance_matrix[i][j] = min(
                    distance_matrix[i - 1][j - 1], distance_matrix[i][j - 1]
                )

            else:
                distance_matrix[i][j] = min(
                    distance_matrix[i - 1][j - 1] + 2,
                    distance_matrix[i - 1][j] + 1,
                    distance_matrix[i][j - 1] + 1,
                )

    return distance_matrix[-1][-1]


def integrate_candidates(sequences, best_candidates, k):
    """
    Input:
        sequences: list containing runic sequences with (incomplete token, <inc> tag) tuples
        best_candidates: dictionary with key = position of the token and value = top candidates for that position
        k: number of candidates are considered from the best_candidates
            (this option is incorporated for efficient testing of model as it only requires the candidates to be calculated once)
    Output:
        sequences: list containing runic sequences with (list of complete candidates, <mod> tag) tuples
    """
    for i, sequence in enumerate(sequences):
        for j, _ in enumerate(sequence):
            if (i, j) in best_candidates.keys():
                sequence[j] = (best_candidates[(i, j)][:k], "<mod>")

    return sequences


def extract_test_samples(test_set):
    """
    Input:
        test_set: list containing the test sequences as (token, tag) tuples
    Output:
        test_set: list with selection of viable sequences as (token, tag) tuples
    """
    for i, sequence in enumerate(test_set):
        longest_sequence = []
        current_sequence = []
        for token, tag in sequence:
            if tag == "<com>":
                current_sequence.append(token)
                if len(current_sequence) > len(longest_sequence):
                    longest_sequence = current_sequence
        else:
            current_sequence = []

        if len(longest_sequence) >= 4:
            test_set[i] = longest_sequence

        else:
            test_set[i] = []

    test_set = [sequence for sequence in test_set if sequence]

    return test_set


def alter_token(token, seed):
    """
    Input:
        token: a token from which a number of characters need to be replaced by '-' and/or '…'
        seed: an integer between 0 and 4
    Output:
        Modified token with one or multiple characters replaced by '-' and/or '…'
    """
    random.seed(seed)
    token = list(token)
    max_changes = len(token) - 1
    number_of_changes = 0

    replacement = random.choice(["-", "…"])

    if replacement == "-":
        while number_of_changes != max_changes:
            pos = random.randint(0, len(token) - 1)
            token[pos] = "-"
            number_of_changes += 1
            add_changes = random.choice(["yes", "no"])
            if add_changes == "no":
                break
        return "".join(token)

    if replacement == "…":
        start = random.randint(0, len(token) - 2)
        end = random.randint(start + 1, len(token) - 1)
        return "".join(token[:start]) + replacement + "".join(token[end:])


def alter_sequence(sequence, seed):
    """
    Input:
        sequence: list containing a single test sequence as (token, <com> tag) tuples
        seed: an integer between 0 and 4
    Output:
        sequence: list containing a single test sequence with (modified_token, <inc> tag) tuples
    """
    random.seed(seed)
    viable_tokens = []
    # Create a copy of the sequence
    sequence = sequence[:]
    for index, token in enumerate(sequence):
        if len(token) > 1:
            viable_tokens.append(index)

    if len(sequence) in [4]:
        num_tokens_to_alter = 1
    elif len(sequence) in [5, 6]:
        num_tokens_to_alter = 2
    else:
        num_tokens_to_alter = 3

    tokens_to_alter = random.sample(
        viable_tokens, min(num_tokens_to_alter, len(viable_tokens))
    )

    for i in tokens_to_alter:
        sequence[i] = alter_token(sequence[i], seed)

    return sequence


def alter_sequences(sequences):
    """
    Input:
        sequences: list containing all test sequences as (token, <com> tag) tuples
    Output:
        altered_sequences: list containing all test sequences with (modified_token, <inc> tag) tuples
    """
    altered_sequences = []
    for sequence in sequences:
        for seed in range(5):
            altered_sequence = alter_sequence(sequence, seed)
            altered_sequences.append(altered_sequence)
    return altered_sequences


def calculate_metrics(gold_pred_list):
    """
    Input:
        gold_pred_list: list containing tuples as (gold standard, [list of predictions])
    Output:
        Metrics: prediction_coverage, accuracy, MRR
        Raw data: correct_predictions, incorrect_predictions and non_predictions
    """
    correct_predictions = 0
    incorrect_predictions = 0
    non_predictions = 0
    reciprocal_ranks = []

    for item in gold_pred_list:
        gold, pred = item
        for i, predictions in enumerate(pred):
            if predictions[1] == "<mod>":
                if len(predictions[0]) == 0:
                    non_predictions += 1
                else:
                    if gold[i] in predictions[0]:
                        correct_predictions += 1
                        rank = predictions[0].index(gold[i]) + 1
                        reciprocal_ranks.append(1.0 / rank)
                    else:
                        incorrect_predictions += 1

    predicted = correct_predictions + incorrect_predictions
    to_predict = correct_predictions + incorrect_predictions + non_predictions

    prediction_coverage = (to_predict - non_predictions) / to_predict
    accuracy = correct_predictions / predicted
    MRR = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

    return (
        prediction_coverage,
        accuracy,
        MRR,
        correct_predictions,
        incorrect_predictions,
        non_predictions,
    )


def main():

    # Extract data and run pre-processing steps
    runestones = read_runestones("RUNTEXTX.txt")
    df = read_excel_to_dataframe("RUNDATA.xls")
    df = filter_dataframe(df)
    runestones = filter_inscriptions(df, runestones)

    df_runestones = pd.DataFrame(runestones)

    punct = print_non_alpha(runestones)
    df_runestones.iloc[:, 0] = df_runestones.iloc[:, 0].apply(
        lambda x: remove_punctuation(x, punct)
    )
    df_runestones.iloc[:, 0] = df_runestones.iloc[:, 0].apply(tokenize_text)
    df_runestones.iloc[:, 0] = df_runestones.iloc[:, 0].apply(get_tags)
    df_runestones = df_runestones[df_runestones.iloc[:, 0].apply(lambda x: len(x) >= 3)]

    df_runestones.to_csv("Processed_Runestones.csv", index=False)
    runestones = df_runestones.iloc[:, 0].tolist()

    train, test = split_data(runestones)

    print(f"Test set size: {len(test)}")

    runic_unigrams, runic_bigrams, runic_trigrams = extract_ngram_probabilities(train)
    save_probabilities_to_file(
        runic_unigrams, runic_bigrams, runic_trigrams, "n-gram_probabilities.txt"
    )

    extracted_unigram_tokens = extract_unigram_tokens(runic_unigrams)
    extracted_bigram_tokens = extract_bigram_tokens(runic_bigrams)
    extracted_trigram_tokens = extract_trigram_tokens(runic_trigrams)

    # Save the n-grams to a plk file
    with open("unigram_tokens.pkl", "wb") as f:
        pickle.dump(extracted_unigram_tokens, f)

    with open("bigram_tokens.pkl", "wb") as f:
        pickle.dump(extracted_bigram_tokens, f)

    with open("trigram_tokens.pkl", "wb") as f:
        pickle.dump(extracted_trigram_tokens, f)

    # Extract appropriate sequences to generate the test set
    test_keys = extract_test_samples(test)

    print(f"Appropriate test sequences: {len(test_keys)}")

    # Create a deepcopy of the test set to use as gold standard
    test_keys_copy = copy.deepcopy(test_keys)
    test_keys_copy = [seq for seq in test_keys_copy for _ in range(5)]

    # Generate the test set
    test_exercise = alter_sequences(test_keys)

    print(f"Synthetic dataset size: {len(test_exercise)}")

    # Provide <com> and <inc> tags for the test set
    test_exercise_tagged = [get_tags(seq) for seq in test_exercise]

    unigram_candidates_dict, bigram_candidates_dict, trigram_candidates_dict = (
        extract_potential_tokens(
            test_exercise_tagged,
            extracted_unigram_tokens,
            extracted_bigram_tokens,
            extracted_trigram_tokens,
        )
    )

    best_candidates = get_best_candidates(
        test_exercise_tagged,
        bigram_candidates_dict,
        trigram_candidates_dict,
        unigram_candidates_dict=unigram_candidates_dict,
        maximum_score=0,
        number_predictions=100,
    )

    # Create an empty DataFrame to store the results
    results = pd.DataFrame(columns=["k", "prediction_coverage", "accuracy", "MRR"])

    # Set ranges of k max candidates
    for k in range(10, 101, 10):

        # Integrate the candidates into their respective sequences
        test_predictions = integrate_candidates(test_exercise_tagged, best_candidates, k)

        # Zip the gold standard and predictions together
        zipped = zip(test_keys_copy, test_predictions)

        # Modify zipped file into a tuple for ease of use
        gold_pred = [(test_key, test_prediction) for test_key, test_prediction in zipped]

        # Calculate metrics
        (
        prediction_coverage,
        accuracy,
        MRR,
        correct_predictions,
        incorrect_predictions,
        non_predictions,
        ) = calculate_metrics(gold_pred)

        # Concatenate the results to the DataFrame
        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    [{"k": k, "prediction_coverage": prediction_coverage , "accuracy": accuracy, "MRR": MRR}]
                ),
            ],
            ignore_index=True,
        )

    # Write the DataFrame to an Excel file
    results.to_excel("results.xlsx", index=False)

if __name__ == "__main__":
    main()