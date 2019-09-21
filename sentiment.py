import pprint
import re
import nltk
import numpy
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
numpy.set_printoptions(threshold=sys.maxsize)

'''
For each regex below:
The first parentheses is a non-capturing group.
The second parentheses is a capturing group.
The third group is a positive lookahead, i.e., it looks ahead but does not consume the matches.
'''

re_quotes_front = r"(?:\s+|^)(')(\w+)(?=\s+|$)"
re_quotes_end = r"(?:\s+|^)(\w+)(')(?=\s+|$)"
re_quotes_front_end = r"(?:\s+|^)(')(\w+)(')(?=\s+|$)"
generic_quotes = r"%s|%s|%s" % (re_quotes_front, re_quotes_end, re_quotes_front_end)

#  List of quotes word not to be replaced.
exception_list = ["'em", "'tis", "'70s"]

#  Regex for start negation tagging.
re_neg_start_tag = r"\bnot\b|\bno\b|\bnever\b|\bcannot\b|\b\w+n't\b"

#  Set to stop negation tagging.
neg_stop_tag = {"but", "however", "nevertheless", ".", "!", "?"}

#  Initializing train and test vocabulary
train_vocab, test_vocab = dict(), dict()


def load_corpus(corpus_path):
    """
    Load the corpus and returns a list of tuples with sentence
    as the first element and classification as the second element.
    :param corpus_path: Path of the corpus.
    :return: List of Tuples [(sen1, classification1), (sen2, classification2), ...]
    """
    snippet_label_list = []
    with open(corpus_path, encoding='latin-1') as file:
        for line in file:
            temp = line.split("\t")
            snippet_label_list.append((temp[0], temp[1].rstrip("\n")))
    return snippet_label_list


def add_spaces(match_obj):
    match_str = match_obj.group(0)
    result = match_str.replace("'", " ' ")
    return " " + result.strip()


def tokenize(snippet):
    """
    Remove quotes at the begin/end from the input sentence.
    :param snippet: A String input for the sentence.
    :return: Return a list with quotes removed.
    """
    temp = ""

    # Replace the exception_list word temporarily to not change them.
    for word in snippet.split(" "):
        if word in exception_list:
            temp = temp + word.replace("'", "$$") + " "
        else:
            temp = temp + word + " "
    temp = temp.rstrip(" ")

    temp = temp.replace("’", "'")
    result = re.sub(generic_quotes, add_spaces, temp)
    result = result.replace("$$", "'")
    return result.split(" ")


def tag_edits(tokenized_snippet):

    #  Initializing helper variables.
    i, result = 0, []
    start, end = False, True
    edit_ = "EDIT_"

    #  Iterating over the snippet to add the EDIT_ tags wherever applicable.
    for word in tokenized_snippet:

        if word == "[]":
            continue

        elif word.startswith("[") and word.endswith("]") and not start and end:
            word = word.replace("[", "")
            word = word.replace("]", "")
            result.append(edit_ + word)

        elif word.startswith("[") and not start:
            start = True
            end = False
            temp = edit_ + word.replace("[", "")
            if temp == edit_:
                continue
            result.append(temp)

        elif word.endswith("]") and not end:
            end = True
            start = False
            temp = edit_ + word.replace("]", "")
            if temp == edit_:
                continue
            result.append(temp)

        elif start and not end:
            result.append(edit_ + word)

        elif not start and end:
            result.append(word)

        else:
            raise ValueError("Something gone seriously wrong while tagging EDIT_!!!")

    return result


def tag_negation(tokenized_snippet):

    #  Initializing helper variables.
    i, result = 0, []
    edit_, not_ = "EDIT_", "NOT_"
    neg_start, neg_end = False, True

    #  Keeping a list of EDIT_ tags to add them back later.
    edit_tagged_list = tokenized_snippet.split(" ")

    #  Remove EDIT_ tags and call the nltk library to get POS tags.
    temp = tokenized_snippet.replace(edit_, "").strip()
    word_pos_list = nltk.pos_tag(temp.split(" "))

    #  Iterating over the nltk POS tagged list and adding the NOT_ tag.
    while i < len(word_pos_list):

        word, pos = word_pos_list[i]
        result_word = ""

        if edit_tagged_list[i].startswith(edit_):
            result_word = result_word + edit_

        if (word in neg_stop_tag or pos in ["JJR", "RBR"]) and neg_start:
            neg_start = False
            neg_end = True
            result.append((result_word + word, pos))
            i = i + 1

        elif neg_start and not neg_end:
            result.append((not_ + result_word + word, pos))
            i = i + 1

        elif re.search(re_neg_start_tag, word) and not neg_start and neg_end:
            if i + 1 < len(word_pos_list) and word_pos_list[i + 1][0].lower() != "only":
                neg_start = True
                neg_end = False
                result.append((result_word + word, pos))
                i = i + 1
                word, pos = word_pos_list[i]
                result.append((result_word + not_ + word, pos))
                i = i + 1
            else:
                result.append((result_word + word, pos))
                i = i + 1

        elif not neg_start and neg_end:
            result.append((result_word + word, pos))
            i = i + 1

        else:
            raise ValueError("Something gone seriously wrong while tagging NOT_!!!")

    return result


def get_features(preprocessed_snippet):

    feature_vector = numpy.zeros(len(train_vocab), dtype=int)

    for word, pos in preprocessed_snippet:
        if "EDIT_" in word or word not in train_vocab:
            continue
        # print(str(train_vocab))
        # print(word)
        index = train_vocab[word]
        val = feature_vector[index]
        feature_vector[index] = val + 1

    return feature_vector


def build_train_vocab(corpus_path):
    indices = 0
    list_snippet_label_tuple = load_corpus(corpus_path)
    for snippet, label in list_snippet_label_tuple:
        list_quotes_separated = tokenize(snippet)
        list_edit_tags = tag_edits(list_quotes_separated)
        list_negation_pos_tuple = tag_negation(" ".join(list_edit_tags))
        # print(str(list_negation_pos_tuple))
        for word, pos in list_negation_pos_tuple:
            if "EDIT_" in word:
                continue
            if word in train_vocab:
                continue
            else:
                train_vocab[word] = indices
                indices = indices + 1


def normalize(X):
    row, col = X.shape
    i, min_max_list = 0, []

    #  Get min, max for each column.
    while i < col:
        j, min, max = 0, 0, 0
        while j < row:
            if X[j, i] > max:
                max = X[j, i]
            if X[j, i] < min:
                min = X[j, i]
            j = j + 1
        min_max_list.append((min, max))
        i = i + 1

    i, j = 0, 0

    #  Normalize the value of each elem in X_train.
    while i < row:
        while j < col:
            f = X[i, j]
            min, max = min_max_list[j]
            # print(str(f) + ", " + str(min) + ", " + str(max))
            normalized_val = (f - min) / (max - min) if max - min != 0 else 0
            # if 0 < normalized_val < 1:
            #     print(normalized_val)
            X[i, j] = normalized_val
            j = j + 1
        i = i + 1

    return X


def evaluate_predictions(Y_pred, Y_true):
    i, tp, fp, fn = 0, 0, 0, 0

    while i < len(Y_pred):
        if Y_true[i] == 1 and Y_pred[i] == 1:
            tp = tp + 1
        elif Y_true[i] == 0 and Y_pred[i] == 1:
            fp = fp + 1
        elif Y_true[i] == 1 and Y_pred[i] == 0:
            fn = fn + 1
        i = i + 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (2 * precision * recall) / (precision + recall)

    return precision, recall, f_measure


def top_features(logreg_model, k):

    np_array = logreg_model.coef_[0]
    i, updated_coef_ = 0, []

    while i < len(np_array):
        updated_coef_.append((i, np_array[i]))
        i = i + 1

    updated_coef_.sort(key=lambda x: abs(x[1]), reverse=True)
    top_k = updated_coef_[:k]
    topk_word_weight = []

    for index, weight in top_k:
        for word, feature_position in train_vocab.items():
            if index == feature_position:
                topk_word_weight.append((word, weight))

    return topk_word_weight


if __name__ == "__main__":

    '''
    Training the model.
    '''
    #  Initialize train_vocabulary.
    corpus_path = "train.txt"
    build_train_vocab(corpus_path)

    #  Initialize X_train and Y_train arrays.
    list_snippet_label_tuple = load_corpus(corpus_path)
    m = len(list_snippet_label_tuple)
    V = len(train_vocab)
    X_train = numpy.empty([m, V], dtype=int)
    Y_train = numpy.empty(m, dtype=int)

    #  Build X_train and Y_train arrays.
    i = 0
    while i < len(list_snippet_label_tuple):
        snippet, label = list_snippet_label_tuple[i]
        list_quotes_separated = tokenize(snippet)
        list_edit_tags = tag_edits(list_quotes_separated)
        list_negation_pos_tuple = tag_negation(" ".join(list_edit_tags))
        feature_vector = get_features(list_negation_pos_tuple)
        X_train[i] = feature_vector
        Y_train[i] = label
        i = i + 1

    # Normalizing the values for the trained model.
    X_normalized = normalize(X_train)

    # Fitting the Gaussian Dist on normalized data.
    obj_nb = GaussianNB()
    obj_nb.fit(X_normalized, Y_train)
    print("Training Completed")

    '''
    Testing the model.
    '''
    test_corpus = "test.txt"
    list_snippet_label_tuple = load_corpus(test_corpus)

    m = len(list_snippet_label_tuple)
    X_test = numpy.empty([m, V], dtype=int)
    Y_true = numpy.empty(m, dtype=int)

    #  Build X_test and Y_true arrays.
    i = 0
    while i < len(list_snippet_label_tuple):
        snippet, label = list_snippet_label_tuple[i]
        list_quotes_separated = tokenize(snippet)
        list_edit_tags = tag_edits(list_quotes_separated)
        list_negation_pos_tuple = tag_negation(" ".join(list_edit_tags))
        feature_vector = get_features(list_negation_pos_tuple)
        X_test[i] = feature_vector
        Y_true[i] = label
        i = i + 1

    # Normalizing the values for the trained model.
    X_test_normalized = normalize(X_test)
    print("Testing Completed")

    # Predicting the test classification on trained data.
    Y_pred = obj_nb.predict(X_test_normalized)
    p, r, f = evaluate_predictions(Y_pred, Y_true)
    print("For Bayes: ")
    print("Precision:" + str(p))
    print("Recall:" + str(r))
    print("F-measure:" + str(f))

    '''
    Logistic Regression Model.
    '''
    lr_obj = LogisticRegression()
    lr_obj.fit(X_normalized, Y_train)
    Y_lr_pred = lr_obj.predict(X_test_normalized)
    p, r, f = evaluate_predictions(Y_lr_pred, Y_true)
    print("For Logistic: ")
    print("Precision:" + str(p))
    print("Recall:" + str(r))
    print("F-measure:" + str(f))

    '''
    Top k features from the Logistic Regression.
    '''
    l = top_features(lr_obj, 10)
    print(str(l))



















