#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: PLEASE PUT YOUR NAMES AND USERIDS HERE
#
# based on skeleton code by D. Crandall, 11/2019
#
# ./break_code.py : attack encryption
#


import random
import math
import copy
import sys
import time

import encode
import string as stt
import numpy as np
# put your code here!
#generate random alphabets
def generate_random_alpha(random_alphabets):
    rand_alpha1, rand_alpha2 = random.sample(stt.ascii_lowercase, k=2)
    return rand_alpha1, rand_alpha2

#generate random numbers
def generate_random_number(random_numbers):
    rand_number1, rand_number2 = random.sample([0, 1, 2, 3], k=2)
    return rand_number1, rand_number2

#caclulate probability
def calculate_probability(decode_string, transition_dict, dictionary_alphabet):
    decodec_tuple = tuple(decode_string.split())
    probab = 0
    for word in decodec_tuple:
        letter_combo = zip(word, word[1:])
        probab += sum([math.log(transition_dict.get(letters, 0.00001)) for letters in letter_combo]) + math.log(
            dictionary_alphabet.get(word[0])[1])
    return probab

# function to decrypt the doc using replacement table
def decode_replacement(string, replace_table1):
    random_alpha1, random_alpha2 = generate_random_alpha(None)
    replace_table1[random_alpha1], replace_table1[random_alpha2] = replace_table1[random_alpha2], \
                                                                   replace_table1[random_alpha1]

    d_decode = string.translate({ord(i): ord(replace_table1[i]) for i in replace_table1})
    return d_decode, random_alpha1, random_alpha2

# function to decode the doc using rearrangment table
def decode_rearrangement(string, rearrange_table):
    random_number1, random_number2 = random.sample([0, 1, 2, 3], k=2)
    rearrange_table[random_number1], rearrange_table[random_number2] = rearrange_table[random_number2], \
                                                                       rearrange_table[random_number1]
    rearrange_decode = "".join(
        ["".join([string[rearrange_table[j] + i] for j in range(0, len(rearrange_table))]) for i in
         range(0, len(string), len(rearrange_table))])

    return rearrange_decode, random_number1, random_number2

# function to decrypt and calculate the ptobability
def break_code(string, corpus):
    if len(string) < 4000:
        n = 10
        iterations = 25000
    else:
        n = 6
        iterations = 10000
    start_time = time.time()
    dictionary_alphabet = dict.fromkeys(stt.ascii_lowercase, (0, 0))
    corpus_tuple = tuple(corpus.split())
    string = string[0:len(string) - 1]
    # W0 probabilities for each alphabet
    for i in range(len(dictionary_alphabet.keys())):
        list_words = [w for w in corpus_tuple if w.startswith(stt.ascii_lowercase[i])]
        dictionary_alphabet[stt.ascii_lowercase[i]] = (len(list_words), len(list_words) / (len(corpus_tuple)))

    # transition probabilities
    transition_dict = {}
    for word in corpus_tuple:
        transition_val = zip(word, word[1:])
        for trans_val in transition_val:
            transition_dict[trans_val] = transition_dict.get(trans_val, 0) + 1
    total_sum = sum(transition_dict.values())
    transition_dict = {x: transition_dict[x] / total_sum for x in transition_dict}

    #replacement and rearrangement tables
    letters = list(range(ord('a'), ord('z') + 1))
    random.shuffle(letters)
    replace_table_original = dict(zip(map(chr, range(ord('a'), ord('z') + 1)), map(chr, letters)))
    # replace_table_original ={'n': 'e','b': 'a','y': 'r','x': 'i','k': 'o','l': 't','m': 'n','s': 's','h': 'l','t': 'c','r': 'u','a': 'd',
    #                          'u': 'p','q': 'm','w': 'h', 'i': 'g', 'j': 'b', 'e': 'f','v': 'y','d': 'w','c': 'k','z': 'v','o': 'x','p': 'z','f': 'j','g': 'q'}

    #rearrangement table
    rearrange_table_original = list(range(0, 4))
    random.shuffle(rearrange_table_original)

    # decode for the first time to obtain p_d
    # decode by traversing in reverse first decrypt using the rearrangement code and then decrypt using the replace code
    rearrange_decode = "".join(
        ["".join([string[rearrange_table_original[j] + i] for j in range(0, len(rearrange_table_original))]) for i in
         range(0, len(string), len(rearrange_table_original))])

    #decode using the replace table
    d_decode = rearrange_decode.translate({ord(i): ord(replace_table_original[i]) for i in replace_table_original})
    #calculate the initial probability
    p_d = calculate_probability(d_decode, transition_dict, dictionary_alphabet)

    #loop to run the algorightms multiple times
    for loop in range(0, n):
        # each loop must run these many iterations
        for ran in range(iterations):
            rand_num = np.random.randint(0, 2)
            if rand_num == 0:
                decode_T_dash = "".join(
                    ["".join([string[rearrange_table_original[j] + i] for j in range(0, len(rearrange_table_original))])
                     for i in
                     range(0, len(string), len(rearrange_table_original))])

                decode_T_dash1, random_alpha1, random_alpha2 = decode_replacement(decode_T_dash,
                                                                                  dict(replace_table_original))
                p_d_dash = calculate_probability(decode_T_dash1, transition_dict, dictionary_alphabet)
            else:
                rearrange_decode1, random_number1, random_number2 = decode_rearrangement(string,
                                                                                         list(rearrange_table_original))
                decode_T_dash1 = rearrange_decode1.translate(
                    {ord(i): ord(replace_table_original[i]) for i in replace_table_original})
                p_d_dash = calculate_probability(decode_T_dash1, transition_dict, dictionary_alphabet)

            if p_d_dash > p_d:
                if rand_num == 0:
                    replace_table_original[random_alpha1], replace_table_original[random_alpha2] = \
                    replace_table_original[random_alpha2], \
                    replace_table_original[random_alpha1]
                else:
                    #             print(rearrange_table_original)
                    rearrange_table_original[random_number1], rearrange_table_original[random_number2] = \
                    rearrange_table_original[random_number2], \
                    rearrange_table_original[random_number1]
                #             print(rearrange_table_original)

                p_d = p_d_dash
                best_decoded = decode_T_dash1
            else:
                rand_nums = np.random.binomial(1, np.exp(p_d_dash - p_d))
                if rand_nums == 1:
                    if rand_num == 0:
                        replace_table_original[random_alpha1], replace_table_original[random_alpha2] = \
                        replace_table_original[random_alpha2], \
                        replace_table_original[random_alpha1]
                    else:
                        rearrange_table_original[random_number1], rearrange_table_original[random_number2] = \
                        rearrange_table_original[random_number2], \
                        rearrange_table_original[random_number1]
                    p_d = p_d_dash
                    best_decoded = decode_T_dash1

    return best_decoded


if __name__== "__main__":
    if(len(sys.argv) != 4):
        raise Exception("usage: ./break_code.py coded-file corpus output-file")

    encoded = encode.read_clean_file(sys.argv[1])
    corpus = encode.read_clean_file(sys.argv[2])
    decoded = break_code(encoded, corpus)

    with open(sys.argv[3], "w") as file:
        print(decoded, file=file)