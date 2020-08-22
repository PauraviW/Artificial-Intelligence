# a3
## part 1 : Part of speech tagging 

## problem 1: Simple Model
In the simple model, we calculated the emission probabilities given the observed state(in this example - words in the corpus). So, for each state in the sentence, we calculated the probability of a given state to have a particular pos given the observed state. Then we calculated the argmax for that particular postition in the sentence and assigned the pos as per that.

For the simple model, our accuracy for the words is 93.92% and for the sentences is 47.45%

## problem 2: HMM, Vitervi implementation and it's working.

## (1) a description of how you formulated each problem.

For calculating the MAP labeling of the sentnces, we have calculated the three required probabities for all combination of words i.e. Emission probability, Transition probabilty and Initial probability. 

## Emission probability:
This is the probability of a word being observed as certain POS (in our case one out of the 12 specified POS values) in the given training data. For each word, we have calculated the number of occurance of that word for each type of POS and then divided by the number of total count of that particular word as all POS. This gives us a probability of the word being each of POS. Similarly, we have calculated the probabilties for all possible combination in the given corpus.

## Transition probability: 
This is the probability of occurance of a POS after another POS i.e. the transition from one POS to another POS calculatd on training data. In similar fashiion, as used to calcualte emission probabilities, we have counted the number of times one POS accoured after another POS in the given corpus and then divided it by the total number of occurance of that particular POS for all possible combination of POS(in our case 12 possibilities). This gives us probability values between 0 and 1 for each possible combination of words.

## Initial probability: 
This is the probability of a POS at the beginning of a sentence in the given copus. To calcualte this we have counted all possible combination of POS that has come at the beginning of sentence in our corpus and then divided it by total number of sentences in our corups. This also gave us proabability of the word to be at the beginnning of a sentence based on our corpus.

## Implementation of Viterbi algorithm: 
We have used these calculated probabilites to implement the Viterbi algorithm to calculate the MAP labeling of the sentence.
the implementation is giving us a word accuracy of 94.33% and sentence accuracy of more than 50.35% in test runs.

## problem 3: Complex model and its working.
The implementation of Gibb's sampling also requires us to calcualted Emission, Transition and Initial probablities. Apart from the transition and emission probabilities that we calculated for Viterbi implementation, we need to calculate additional transition probabilities like probability table which has the transition from S(n) to S(n-1) and also the transition probabilty of S(n) to (S(1),S(n-1)). 

we have divided the samplling problem in three different parts based on four different cases.
## Case 1: when sampling S1
when we are sampling S1, the baye's net structure dependence is different, the probability of S1 is affected by  W1, S2, Sn and S(n-1). It will also depend on S(n-1) as the baye's net forms a "V" structure at node Sn and when Sn is know, it actives trail and which makes, S1 ans S(n-1) dependedent. Using all these probabilities we calculate the most probable POS of the sampling POS.
## Case 2: when sampling from S2 to S(n-2)
When we are sampling from S2 to S(n-2), the probability of Si is affected by Wi,S(i+1), and S(i-1) nodes. we use these probabilities from our transition and emission tables and calculate the most probable POS for the sampling POS.
## Case 3: when sampling for Sn
When we are sampling for Sn, the probability of Sn is affected by Wn, S(n-1), and S1. We used the transition and emission table to calculate the probability of the POS for the Sn and the assign the most probable POS to it.
## Case 4: When sampling for S(n-1) 
when we are sampling S(n-1), the baye's net structure dependence is different, the probability of S(n-1) is affected by  W(n-1), S1, Sn and S(n-2). The dependence on S1 is activated by the formation of "V" structure at node Sn and when Sn is know it actives trail and which makes, S1 ans S(n-1) dependedent. Using all the probability tables, we calculate the most probable POS of the sampling POS.

Based on all these cases, we have collected 1000 samples for each sentence and assigned the final POS based on the most number of accurance of a POS values for a word. For test run, the word accuracy is 92.41% and sentence accuracy is 43% on test runs.

## Discussion of any problem face, any assumtions, simplifications or design decision made.
While calculating transition probabilities, we come across some of the cases when the total number of counts for a particular POS was very less (less than 50) as compared to other and it was being divided by relatively smaller denominator, which makes it a special case in which a small quantity is dividec by a small number, so it will give high probability values for not so frequent occurance in english language. To address this, for all those cases where the total count was less than 50, while calculating probabilities for the same, we divided it by a bigger denominator (in our case, 10000). This small change in transition probability, improved our prediction by 1% in words accuracy. 

We test ran our code for different values of samples, like for 2000 samples for each sentence to 10,000 samples for each sentence. though increase sample size from 2000 to 10000, improves the accuracy but the run time increases dramatically for the whole test data. so, we have settled for an optimal value under time constraint.

We have also discarded first 250 samples that we have generated and using last 1000 samples to do the predictions.

# part 2 : Code breaking
## Description of how you formulated each problem
We started by first calculating the initial probabilities that we will need to plug-in in the Metropolis Hasting's algorithm.
1) Probability table that the word starts with this letter.
2) Transition probability that the letter follows another letter

## Description of how your program works
Here we first, split the corpus into tokens and store it for the later use.  
Then we calculate the probabiltities for the initial and the transition tables.  
According to the algorithm, randomly we generate the replacement and rearrangement tables.
The steps of the algorithm are as follows:

### Step 1:
Using the original tables, we decode the document first by de-rearranging followed by de-replacing. Then we calculate the probability of this document being correct using the probability formula provided in the docs and the probability tables calculated before.
We save this probability as the baseline probability and the tables as baseline tables.  

Now we run this part of the program 28,000 times for 10 iterations:
### Step 2:
- Here, we modify the guess by changing either of the tables.  
- Using the new values we decode the document
### Step 3:
- We now calculate the probability.
- If the new probabilty is better than the previous, then we incorporate this new guess in the original table.
- If the new probability is less than the previous, then we take a chance based on the ratio of the old and the new probabilties.
  We have used the numpy binomial to achieve this. Based on the result, we either replace the table values or dont. This helps us get out     of the local minima problem. 
- We replace, the original probability value with the new one based on the previous step. 

## Discussion of any problems you faced, any assumptions, simplifications, and/or design decisions you made.
- The main problem we faced was/is is the fact it takes very long to converge and gives us a readable output.
- While decoding, we wrote our own code to decode the text, but in the starter code we found a very efficient code for decoding, which will   be computationally fast and save us time.

# part 3: spam classification 

## Q1. A description of formulation of problem.

We have implemented Naive Bayes classifier with with Naive bayes assumtions for a bag of words model. We trained the model on the labelled traininig data which has 'spam' and 'notspam' emails preclassified. While trainig we calculated the probabilities of each words, starting from subject line of the mail to the end of mail, to be in spam mail and not spam mail. Apart from using the probabilities of the words alone, we have also calcualted the probability of mail being a spam mail or non spam mail based on the training data given to us. Essentially, this probability for spam class will be number of spam mails in our training data upon total training data and probability for notspam class will be number of notspam emails data upon total taining data. 

We have also used a Dirichlet prior with alpha value of 1.5, we have done a linear search to find the best suitable alpha value for this data set (the code for linear search of alpha value of Dirichlet prior has not been included in the mail code). The rational behind using the Dirichlet prior for this classification was to handle the cases when a test word has zero probabiltiy in any one of the training class i.e. spam or not spam. in that case the overall probabiltiy will become zero and we will not be able to classify the mail. though there are other ways to handle this problem, but we chose to use a prior for the model. This prior with alpha value of 1.5 adds a count of 0.5 to each word in spam and notspam class while calculating the probability, which prevents the probability to attend 'zero' value while classifying the data.

## Q2. Working of code

The code can be run as per the directions given in assignment. We first generate the probabilities for both the classes, spam and notspam. and store it in a dictionary which has probability for both the classes for all unique words in the whole training set. We also calculate the class probability for each class. Then for classifiying the main, we first convert the whole mail into a bag of words and then calculated posterior probability of the each word in the email. For each test mail, we calculate two probabilities, first the probability of the mail being in spam class and then the probability of the test mail being in notspam class. Once, calculated both, we compare these two probabilties and the based on whichever probability comes greater than other, we classify the test mail as being of that class. 

Along with the code, we have also submitted our test-run result file with name "current result". Our code is giving 96.95% accuracy for the test cases given. 


## Q3. Discussion of any problem face, any assumtions, simplifications or design decision made.

We have implemented some design decision for the model by using prior. We have implemented a Dirichlet prior for the model wiht alpha vlaue of 1.5, which essentially adds a value of 0.5 in each of word in the bag of word while calculating the probability of each word. This design decision is aligned with the Bayesian treatment of the model and make the overall classification model more accurate.
