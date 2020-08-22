# a2-2

# Report of Problem 1 : IJK

## 1. Description of how you formulated the problem.

  As per the rules of the game, the game is divided into two parts, deterministic and non-deterministic. For deterministic "A/a" is added at a predictable position, the position is predicted by heurisic and for non-deterministic, "a/A" is added randomly. 
  
## For deterministic part:  

## Heuristic formulation: 
For the current state of game, we are calculating the heuristic values of the alphabets present on the game board. We have assigned ASCII values to the alphabets on the board. Since, the ASCII values of small "a" and capital "A" start from different values, we have decreased the ASCII values of small alphabets by 32 to match the values of capital alphabets. Using these values, the heuristic will calculate the sum of total values on the board for "A" and "a", separately. After calculating the sum, we have subtracted the sum of small "a" from capital "A" to calculate the final value of the heuristic at any state of calculation.  

## Problems faces in heurtistic implimentation: 
a) Same heuristic values after final calculation for different moves: we had two ways of dealing with this problem, first was to define a method to break the tie in heuristic values and second, that we define such a heuristic that we dont run into this problem. During the course of finding the solution for this problem, we finally decided to implement both together. for running to less number of ties, we decided to assign odd prime values to each alphabets from "A" to "K" like,(1,3,5,7,11,13,17,19,23,29,31), the supposed advantage was that we would run into tie problem less often but at the same time we defined the a tie breaking ## method ##. We did implement this heuristic but at the end we didn't find any significant improvent in running time for a horizon value. so, we went with the old heuristic (using ASCII values) with a method for tie breaking.

## For Non-deterministic part:

## EXPECTIMINIMAX algorithm: 
For non deterministic applicaiton, we have used the expectiminimax alogrithm, so there will be a chance node after each min and max node. At the chance node we will calculate the expectated value. The min node or max node will choose one value from the chance node to fulfill their requirement. To find a move, we need to assign some values to each state( calculated by our heuristic function) and find weighted sum of the current state by multiplying it with a probability value. Before making a move, we are going down the game tree up to the horizon value ( taken here to be 5) and at each value we are finding the probability of adding a number at each empty sell. As we go down the game tree, the value of probability increases and then we find the corresponsing expected value at each chance node. Coming up this the game tree form horizon, we selected the optimal move that we give the maximum expected value at the horizon. 




## MINIMAX Implementation:
For making decision on the heuristic values, we have used minimax algorithm. The minimax algorithm, at any given iteration down the horizon, will make a decision depending on the node that it is in, MAX node or MIN node. The complete working of MINIMIAX in described in "brief descrition of program" part of the report.

## problems faced in minimax implementation
While implementing the minimax algorithm, initially we were stuck with the problem on tuple comparision while, comparing the heuristic values. But then we realised that we were passing the game state and moves also while going down till the horizon. For deciding our move, we didn't really need to know all the moves that were taken down the tree, we only needed to know the first move that resulted in finding the best value for us. So we removed the direction values and the current game state values to overcome this problem.

## Horizon : 
For evaluating the possible moves of MAX down the nodes and considering the MIN player takes the worst move for us, we have decided a horizon value.Tthe program will iterate up that that values down the game tree, find heuristic values at each node and then final decision is taken by the MINIMAX algorithm on move. After each step of the game, the heuristic values of calculated down the game tree till the horizon values and a decision is made. 

## Alpha-Beta pruninig: 
we have implemented alpha-beta pruning for making the decision making faster. we have assigned, alpha to the MAX node, alpha has the heuristic values calculated the and backed up for min node. and the beta value is assigned to MIN node, which are backup values from max node. For pruning, we discontinue the search below a MAX node "N" if its alpha value is greater than the veta values of a min ancestor of "N". we discontinue the search below a MIN node "N" if its bets values is less than the alpha values of a max ancestor of "N".

## 2. A brief description on how your program works.

Deterministic game: 

The first alphabet "A" is placed at co-ordinate (0,0) as a start, for the next move we calculate the heuristic value down the game tree up the value of horizon, (we have kept the horizon value at 5, as optimal value for both deterministic and non deterministic part), while going down the tree, at each max node the minimax backs up the maximum value from the child min node and at each min node it back up the minimum values from the child max node. The heuristic values are backed up in same manner till we again reach the top. On the basis on the backed up values and the first move taken, the minimax algorithm decied the next move to to taken. the game moves in same manner.

Non-deterministic game: 

For non deterministic applicaiton, we have used the expectiminimax alogrithm, so there will be a chance node after each min and max node. At the chance node we will calculate the expectated value. The min node or max node will choose one value from the chance node to fulfill their requirement. To find a move, we need to assign some values to each state( calculated by our heuristic function) and find weighted sum of the current state by multiplying it with a probability value. Before making a move, we are going down the game tree up to the horizon value ( taken here to be 5) and at each value we are finding the probability of adding a number at each empty sell. As we go down the game tree, the value of probability increases and then we find the corresponsing expected value at each chance node. Coming up this the game tree form horizon, we selected the optimal move that we give the maximum expected value at the horizon. 

# Problem 2 : Horizon Finding
## 1)A description of how you formulated each problem:
We approached this problem with a view that we can reuse the code for each solution. Thus, we calculated some common probabilities.

## i) Emission Probability:
For any HMM problem, we need to have emission probabilities for the states. Since, the edge_strength gradient already gave us the probabilities based on the intensity of the pixel at that point, we calculated the probabilities by dividing each cell with the total of the intensity values of the column in which the cell belongs. This made sure that we have all the values in the scale 0 to 1.

## ii) Transition Probability:
Here, our thought was to find out probability based on the distance. So, the probability of the pixels next to each other should be highest and the the probability of the pixels farthest should be lowest. For that we twisted the way weighted averages are used. Firstly, we calculated the manhattan distance of each pixel in one column to all the pixels in the next column. Then, we summed up the distance and divided the sum with the individual values. After that, we again summed up the values that we obtained after the division. And then, we divided these values obtained before with their sums for each column. This way, we made sure, that the pixels farthest from the current pixel have less transition probability and vice versa.

## iii) Initial State distribution :
For detecting the starting point, we took the index of the pixel with maximum emission probability in the first column and used it as our starting point.

There are 3 different ways as stated in the question:

## a) Bayes Net:
So, the most intuitive solution that we could find here was finding out the row corresponding to each column, which gives us the maximum emission probability. We used the numpy argmax method to obtain the ridge. And then we appended all the maximum values to the ridge array and displayed it as the output. 

## b) Viterbi and backtracking
We have used Viterbi algorithm to find the MAP of the ridge line. To populate the viterbi matrix, we multiplied emission value with the maximum of the the product of last state with the transition value, as calculated and explained above. For each state of each column of the picture, we keep the track of the state that gives the maximum value for the current state. In similar way, we have populated each column of the viterbi keeping tack of maximum values of for each cell of the column.

once we reach the final column of the picture and find out the maximum value of for that column and from that value, we back tack the values for each column using the index that we saved while moving forward. We save this values in ridge list and pass it to the draw the ridge line.

## c) Human Intervention
We improved the position of the ridge, by adding human inputs. Here, we replaced the probabilities of pixels based on the row and the column co-ordinates provided. For the exact pixel provided to us by the row and the column co-ordinate, we replace the probability by 1. And for every other pixel in the given column, we replace the probability with 0. This way, we help, the Viterbi to manipulate the results.

## 2) A brief description of how your program works
For each problem, we have tried to basically maximize the probability of getting the probability of the pixel which gives us the ridge. Thus, we fitted out model in a way that resembles the Viterbi.

We firstly calculate the values for emission, transition and the initial distribution.
Then, we maximize the probability for each state by multiplying the maximum in the previous state by the transition probability.
After that, we multiply the maximum from that list with the emission of the current state.
This, way, we get maximum values for each state and then we select the best amongst them.
While, traversing, we maintain a list of row indices which we obtain by backtracking to the row in the previous column which gave us the maximum probability.
This way, we obtain a ridge and then use it to plot it on the image.

## (3) and discussion of any problems you faced, any assumptions, simplifications, and/or design decisions you made.  
Sometimes, our ridge would just jump up 10-15 pixels. So, there was no continuation in the ridge.
## Solution:
We tweaked our transition probabilities so that the farthest pixels get the even less probabilities. This way, we made sure, that there was a smooth transitioning going on in the ridge.

## The images with blurred horizons.
## Solution:
The problem was that the emission probabilities dominated the viterbi algo and thus, we got incorrect ridges. Here, we basically, normalized the emission probabilities to get the desired output.
