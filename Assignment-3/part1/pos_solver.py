###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import numpy as np
import random
from collections import Counter


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    def __init__(self):
        self.initial_word_dict = {}
        self.first_pos_prob = {}
        self.emission_dict  = {}
        self.transition_dict = {}
        self.last_pos_prob = {}
        self.rev_transition_dict = {}
        self.last_first_key_table = {}
        self.last_first_transition = {}
        self.double_transition = {}
        self.key_table = {}
        self.word_table = {}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        probab = 1
        # Simple Case - We calculate the posterior probabilities in this case by multiplying all the emission probabilities for each pos in the sentence
        if model == "Simple":
            for i in range(len(sentence)):
                probab *= self.emission_dict.get((sentence[i], label[i]), 0.00001)
            return np.log(probab)
        # Complex- Case -
        # Case 1: POS of the First Word -  We multiply the prior probability for the first state, with its emission probability
        # Case 2: POS of the words from second to Second-last : We multiply the transition and emission probabilities for the subsequesnt states
        # Case 3: POS for the last state: Here we multiply the probability of the POS being at the first, second last and last locations.

        #We multiply the probabilities obtained in the above three cases and that becomes our posterior probability.

        elif model == "Complex":
            for i in range(len(sentence)):
                if i == 0 :
                    probab *= self.first_pos_prob.get(label[0], 0.00001) * self.emission_dict.get(
                        (sentence[0], label[0]), 0.000001)
                elif i == len(sentence)-1:
                    probab *= self.double_transition.get((label[len(sentence) -1], label[len(sentence) - 2], label[0]), 0.000001)
                else:
                    probab *=  self.emission_dict.get((sentence[i], label[i]), 0.000001)* self.transition_dict.get((label[i-1], label[i]), 0.00001)


            return np.log(probab)
        #HMM - Case -
        #   Case 1: POS of the First Word -  We multiply the prior probability for the first state, with its emission probability
        #    Case 2: POS of the words from second to last : We multiply the transition and emission probabilities for the subsequesnt states
        elif model == "HMM":
            for i in range(len(sentence)):
                if i > 0:
                    probab *=  self.emission_dict.get((sentence[i], label[i]), 0.000001)* self.transition_dict.get((label[i-1], label[i]), 0.00001)
            probab *= self.first_pos_prob.get(label[0], 0.00001) * self.emission_dict.get((sentence[0], label[0]), 0.00001)
            return np.log(probab)
        else:
            print("Unknown algo!")

    # Do the training!
    # We find the conditional probability tables for all the transition, emission and prior probabilities
    def train(self, data):
        # simple model
        final = []
        for sentence, pos in data:

            # reverse the statement to find the probabilities for transition from state_i+1 to state_i
            # this is used in the case of complex MCMC
            rev_sentence, rev_pos = sentence[::-1], pos[::-1]
            # CPT for the POS in the last and the first positions- Used in MCMC
            self.last_first_key_table[pos[len(pos) - 1]] = self.last_first_key_table.get(pos[len(pos) - 1], 0) + 1
            self.last_first_transition[(pos[len(pos) - 1], pos[0])] = self.last_first_transition.get((pos[len(pos) - 1], pos[0]),
                                                                                          0) + 1
            #CPT for the POS tagged to the First word of the sentence
            self.first_pos_prob[pos[0]] = self.first_pos_prob.get(pos[0], 0) + 1
            self.last_first_key_table = {x: 10000 if self.last_first_key_table[x] < 50 else self.last_first_key_table[x] for x in
                                    self.last_first_key_table}

            # Case 2:
            # reverse_transition
            # Here, we calculate the transition probabilities from pos at state i+1 to i.
            combi = list(zip(rev_pos, rev_pos[1:]))
            for trans_val in combi:
                self.rev_transition_dict[trans_val] = self.rev_transition_dict.get(trans_val, 0) + 1

            # case 3:
            # double_transition
            # sn, sn-1, s1
            # Here we calculate the CPT for transition of P(sn/s1, sn-1)
            self.double_transition[(pos[len(pos) - 1], pos[len(pos) - 2], pos[0])] = self.double_transition.get(
                (pos[len(pos) - 1], pos[len(pos) - 2], pos[0]), 0) + 1

            #cCalculate the total number of words in a corpus
            for word_val in sentence:
                self.word_table[word_val] = self.word_table.get(word_val, 0) + 1
            for pos_val in pos:
                self.key_table[pos_val] = self.key_table.get(pos_val, 0) + 1

            # probability that the sentence will start with this pos
            self.first_pos_prob[pos[0]] = self.first_pos_prob.get(pos[0], 0) + 1

            # emission probabilities of a word and pos
            combo = tuple(zip(sentence, pos))
            for com in combo:
                self.emission_dict[com] = self.emission_dict.get(com, 0) + 1

            # caclulate the transition probabilties for the combinations of pos
            combi = list(zip(pos, pos[1:]))
            for trans_val in combi:
                self.transition_dict[trans_val] = self.transition_dict.get(trans_val, 0) + 1
        total_sum_first = sum(self.first_pos_prob.values())
        self.first_pos_prob = {x: self.first_pos_prob[x] / total_sum_first for x in self.first_pos_prob}

        #calculate probabilties
        for key1, key2 in self.emission_dict:
            self.emission_dict[(key1, key2)] = self.emission_dict[(key1, key2)] / self.word_table[key1]
        for pos1, pos2 in self.transition_dict:
            self.transition_dict[(pos1, pos2)] = self.transition_dict[(pos1, pos2)] / self.key_table[pos1]


        for pos1, pos2 in self.rev_transition_dict:
            self.rev_transition_dict[(pos1, pos2)] = self.rev_transition_dict[(pos1, pos2)] / self.key_table[pos1]

        for pos1, pos2 in self.last_first_transition:
            self.last_first_transition[(pos1, pos2)] = self.last_first_transition[(pos1, pos2)] / self.last_first_key_table[pos1]

        for pos1, pos2, pos3 in self.double_transition:
            self.double_transition[(pos1, pos2, pos3)] = self.double_transition[(pos1, pos2, pos3)] / self.last_first_key_table[pos1]

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    # Case : simple
    # In this case we use the emission probabilties to find the possible part of speech and then  we take the argmax to find the best
    # possible pos and set it as the pos
    def simplified(self, sentence):
        output = []
        keys = tuple(self.emission_dict.keys())
        for word in sentence:
            word_list = [i for i in keys if i[0] == word]
            vals = [self.emission_dict.get(i, 0) for i in word_list]
            if len(word_list) > 0:
                output.append(word_list[np.argmax(vals)][1])
            else:
                output.append("noun")
        return output
    # Case : Complex MCMC
    # In this case, we use the bayes net and calculate probabilties using different methods for pos at different positions in the sentence.
    # Case 1: POS of the First Word -  We multiply the prior probability for the first state, with its emission probability and the transition probabiltiy for state n , given s1 and sn-1
    # Case 2: POS of the words from second to Second-last : We multiply the transition and emission probabilities  and the reverse transition from the current state to the previous for the subsequesnt states
    # Case 3: POS for the last state: Here we multiply the probability of the POS being at the first, second last and last locations .

    def complex_mcmc(self, sentence1):
        tuple_pos = tuple(self.key_table.keys())
        main_particle_dict = {}
        # particle zero
        if len(sentence1) > 1:
            for i in range(0, 750):  # 2000
                track_position = [i for i in range(len(sentence1))]

                x_0 = random.choices(tuple_pos, k=len(sentence1))
                while len(track_position) > 0:
                    # generate random index
                    index = random.choice(track_position)
                    # remove index from the list
                    track_position.remove(index)
                    probab_list = []
                    # case 1 : first
                    if index == 0:
                        for curr_pos in tuple_pos:
                            probab = self.rev_transition_dict.get((x_0[1], curr_pos), 0.0000001) * \
                                     self.last_first_transition.get(x_0[len(x_0) - 1], 0.0000001) * self.emission_dict.get(
                                (sentence1[index], curr_pos), 0.0000001) * self.first_pos_prob.get(curr_pos, 0.0000001)
                            probab_list.append(probab)
                    # case 3 : last
                    elif index == len(sentence1) - 1:
                        for curr_pos in tuple_pos:
                            probab = self.emission_dict.get((sentence1[index], curr_pos), 0.0000001) * \
                                     self.double_transition.get((curr_pos, x_0[len(x_0) - 2], x_0[0]), 0.0000001)
                            probab_list.append(probab)
                    # case 2 : middle
                    else:
                        for curr_pos in tuple_pos:
                            probab = self.emission_dict.get((sentence1[index], curr_pos), 0.0000001) * \
                                     self.rev_transition_dict.get((x_0[index + 1], curr_pos),
                                                             0.0000001) * self.transition_dict.get(
                                (x_0[index - 1], curr_pos), 0.0000001)
                            probab_list.append(probab)

                    new_pos = np.argmax(probab_list)
                    x_0[index] = tuple_pos[new_pos]

                main_particle_dict[i] = x_0
            size = len(main_particle_dict.get(0))

            max_d = {}
            l = []
            for i in range(size):
                for key in main_particle_dict.keys():
                    if key > 150:
                        l.append(main_particle_dict.get(key)[i])
                test_list = Counter(l)
                max_d[i] = test_list.most_common(1)[0][0]
                l.clear()


            return list(max_d.values())
        else:
            return ["noun"] * len(sentence1)

    # HMM - Case -
    #   Case 1: POS of the First Word -  We multiply the prior probability for the first state, with its emission probability
    #    Case 2: POS of the words from second to last : We multiply the transition and emission probabilities for the subsequesnt states

    def hmm_viterbi(self, line):
        keys = tuple(self.first_pos_prob.keys())
        list_prob = [self.first_pos_prob[j] * self.emission_dict.get((line[0], j), 0.0000000001) for j in keys]
        backtrack = []
        for word in line[1:]:
            states = []
            temp = []
            for current in keys:
                temp_list_prob = []

                for previous in keys:
                    temp_prob = list_prob[keys.index(previous)] * self.transition_dict.get((previous, current), 0.0000000001)
                    temp_list_prob.append(temp_prob)
                min_index = np.argmax(temp_list_prob)
                temp.append(self.emission_dict.get((word, current), 0.000000001) * np.max(temp_list_prob))
                states.append(min_index)
            list_prob = temp
            backtrack.append(states)
        last_index = np.argmax(list_prob)
        pos1 = []
        pos1.append(keys[last_index])
        for index in backtrack[::-1]:
            last_index = index[last_index]
            pos1.append(keys[last_index])
        return pos1[::-1]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

