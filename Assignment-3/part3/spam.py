#!/usr/local/bin/python3
# coding: utf-8

import re
import sys
import os
import csv
import math


# Everything after subject line is considered for corpus.
def generate_spam_corpus(spam_path):
    spam_corpus = {}
    token_spam = 0
    for filename in os.listdir(spam_path):
        with open(spam_path + "/" + filename, "rb") as f:
            mail = str(f.read().lower())
            text = mail[mail.find("subject:"):]
            spam = re.findall(r"[\w']+", text)
            for i in spam:
                token_spam += 1
                spam_corpus[i] = spam_corpus.get(i, 0) + 1
    return spam_corpus, token_spam


# Everything after subject line is considered for corpus.
def generate_not_spam_corpus(not_spam_path):
    not_spam_corpus = {}
    token_not_spam = 0
    for filename in os.listdir(not_spam_path):
        with open(not_spam_path + "/" + filename, "rb") as f:
            mail = str(f.read().lower())
            text = mail[mail.find("subject:"):]
            notspam = re.findall(r"[\w']+", text)
            for i in notspam:
                token_not_spam += 1
                not_spam_corpus[i] = not_spam_corpus.get(i, 0) + 1
    return not_spam_corpus, token_not_spam


# We have used a Derichlet prior with alpha value of 1.5 which adds a value of 0.5 in all elements of vocabulary
# we have tried a number a alpha values and alpha = 1.5 gives the best result.
# Calculates probability of words in each class.
def calculate_probability_words_per_class(spam_corpus, token_spam, not_spam_corpus, token_not_spam):
    vocab = {**spam_corpus, **not_spam_corpus}
    vocab_len = len(vocab.keys())
    probability = {}
    for word in vocab:
        sp = (spam_corpus.get(word, 0) + 0.5) / (token_spam + vocab_len)
        nsp = (not_spam_corpus.get(word, 0) + 0.5) / (token_not_spam + vocab_len)
        probability[word] = (sp, nsp)
    return probability


# Calculating probability of spam and not-spam classes
def calculate_probability_of_classes(spam_path, not_spam_path):
    spam_count = len(os.listdir(spam_path))
    not_spam_count = len(os.listdir(not_spam_path))
    spam_class_prob = spam_count / (spam_count + not_spam_count)
    not_spam_class_prob = not_spam_count / (spam_count + not_spam_count)
    return spam_class_prob, not_spam_class_prob


# Predicting classes of test emails
def prediction(spam_class_prob, not_spam_class_prob, probability, test_mail):
    spam_pred = math.log(spam_class_prob)
    notspam_pred = math.log(not_spam_class_prob)
    for word in test_mail:
        try:
            spam_pred += math.log(probability[word][0])
            notspam_pred += math.log(probability[word][1])
        except:
            KeyError

    if spam_pred > notspam_pred:
        return "spam"
    else:
        return "notspam"


def evaluate(test_path, spam_class_prob, not_spam_class_prob, probability):
    result = {}
    for filename in os.listdir(test_path):
        with open(test_path + "/" + filename, "rb") as f:
            mail = str(f.read().lower())
            text = mail[mail.find("subject:"):]
            test_mail = re.findall(r"[\w']+", text)
            result[filename] = prediction(spam_class_prob, not_spam_class_prob, probability, test_mail)
    return result


def write_file(output_file, result):
    with open(output_file, 'w') as outfile:
        for key, val in result.items():
            outfile.write("%s %s\n" % (key, val))
    outfile.close()
    if os.path.exists(output_file):
        return True
    else:
        return False


def test_ground_truth(result):
    with open("test-groundtruth.txt", "r") as f:
        grd = str(f.read().lower()).split("\n")
    ground_test = {}
    for x in grd:
        try:
            ground_test[x.split()[0]] = x.split()[1]
        except:
            IndexError

    # result comparison
    correct = 0
    for key, val in result.items():
        if ground_test[key] == val:
            correct += 1

    return correct / len(ground_test)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        raise Exception("usage: ./spam.py training-directory testing-directory output-file")

    training_directory, testing_directory, output_file = sys.argv[1:]

    spam_path = training_directory + "/spam"
    not_spam_path = training_directory + "/notspam"

    # generating training set: spam emails data
    spam_corpus, token_spam = generate_spam_corpus(spam_path)

    # generating training set: notspam emails data
    not_spam_corpus, token_not_spam = generate_not_spam_corpus(not_spam_path)

    # calculating probability of words in respective classes
    probability = calculate_probability_words_per_class(spam_corpus, token_spam, not_spam_corpus, token_not_spam)

    # calculating probability of spam email and notspam email
    spam_class_prob, not_spam_class_prob = calculate_probability_of_classes(spam_path, not_spam_path)

    # predicting classes of test directory
    result = evaluate(testing_directory, spam_class_prob, not_spam_class_prob, probability)

    # writing to file
    status = write_file(output_file, result)
    if status:
        print('results successfully saved!')
    else:
        print('something went wrong!')

    print('Ground-truth-accuracy: %s' % test_ground_truth(result))
