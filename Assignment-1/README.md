# Part 1

Report
Q1.	Description of state space, successor, goal state, initial state, cost function

State space: It is the combination of all possible configuration which are valid under given move.

Goal state: To find the shortest sequence of moves that restores the canonical configuration given an initial board configuration.

Initial state: It is the state of board provided with location of ‘zero’ as the place of empty tile.

Successor: Successor state is the combination of valid states which are generated under stated method, i.e. “original”, “circular” and “luddy”, as defined originally in the question. The successors need to be validated to be inside the board configuration.

Cost function: cost function varies depending upon the method used:

Original method: every movement of tile is considered as one-unit of cost.
Circular method: every movement of tile to the adjacent vacant place is counted as one-unit cost and every valid circular move is also counted as one-unit cost. The total cost is the combination of valid circular move added with number of adjacent moves made.
Luddy method: every valid luddy move is counted as one-unit cost. 

Q2.	Why code fails

•	Only the “original” puzzle was supported.
•	Heuristic function was not implemented for any of the move, so it was not giving optimal solution.
•	A* search was not implemented.
•	Stack was used. Which was not optimizing the search for this problem.

Q3.	How to fix code
•	Only the “original” puzzle was supported: - defined the circular and the luddy moves based on the same format provided in the raw code. While implementing this, the key-value of the dictionary was needed to be inverted as the moves were named with same alphabet in circular method.

Table 1. luddy moves.
Row change	Column change	Move name
+1	             +2	          E
+1	             -2	          F
-1	             +2          	G
-1	             -2	          H
-2	             -1           D
-2	             +1	          C
+2	             -1	          B
+2	             +1	          A

Table 2. additional move for circular method.
Row change	Column change	Move name
0	             +3           	L
0	             -3	           R
+3	             0	           D
-3	             0	           U

•	Implementation of heuristic function: for each method, we implemented a different heuristic.
For original  and luddy methods: Number of misplaced tile plus steps already taken.
f(s) = g(s) + h(s)  
g(s) -> moves already taken
h(s) -> sum of distance of numbered tile to their goal.
h(s) -> number of misplaced tiles
We have used this heuristic to implement original method using heapq. It counts the number of steps taken so far and add it with the number of misplaced tiles in the current board. Heapq makes sure that the lowest priority board is popped for new execution.
( the implementation method is different for both methods, which is why two separate functions are written.)

For circular method: Manhattan distance with circular move consideration.
The circular move is capable of solving both even parity and odd parity boards when number of misplaced tiles is taken as heuristic. But a faster heuristic can be used, which uses the idea of manhattan distance but also considering the circular move defined in the problem. Every circular move is counted as one step and for rest journey the manhattan distance is calculated. This heuristic gives faster solution by distinguishing between original movements and circular movements.

•	A* search was implemented using heapq and a suitable heuristic function for each method.
•	Stack was used in the raw code, we changed it to heapq for faster implementation with suitable heuristic function.

Q4.	Implementation of code: The raw code which was provided was sufficient to give the idea and direction of implementation. First of all, we encoded the valid moves for each method separately into dictionary. For each method, we defined a separate function which is will swap the tiles if moves are found valid. 
The main work was the implementation of heuristic function, which we have implemented separate for each method as described above. The reason for doing it separately was that, different heuristic was giving faster result for different method.  For even better and faster implementation of luddy method, we used visited state count and eliminated all repeated expansion of visited state. Still, our code was talking longer time in solving board4 using luddy method and we needed to optimize the sear time to make it faster. So, we used a dictionary key search instead of going through all the elements of list. Which gave almost 10x improvement in search time.
.


# Part 2

State space: All possible combination of cities and highways connecting them, along with their related parameters like distance, speed limits, highway names, the time it takes to travel through that highway and mpg value.
State Space is maintained in the class Graph.
Graph maintains a key value pair of city name with its city class object. Each city class has a list of connected highway class objects and each highway class object is inturn connected to another city object.

Initial state: It is the node in the graph that will represent the start city and its parameters.

Goal state: The optimal path to the destination node i.e. the end city, depending upon the passed cost function.

Successor function: Function which will generate the connected cities and the cost at those cities. Below is the function responsible for this:


         def get_connected_cities(roads, city_sol):
             cities = []
             city = city_sol[0]
             c = roads.get_city(city)
             for neighbour in c.adjacent:
                 sol = city_sol[1]
                 cities.append((neighbour, updated_sol(sol, c.adjacent[neighbour])))

             return cities


Cost function: Cost Function will return the optimal cost which will be pushed into the fringe as its priority.

A* Algorithm has been implemented with h(n) = 0. To limit the number of revisited states, below criteira has been implemented:

If a node has been generated again from some other path with some other cost, however, this time the cost is less compare to previously visited path's cost, add that node back into fringe so that it can be revisited.

A solution class has been created to store the visited path related information at every node.


# Part 3  
Choosing a team

1. Description of how we formulated the search problem, including precisely defining the state space, the successor function, the edge weights, the goal state.

Ans: 
This search problem required us to find the best combination of robots with the highest number of skills, which are within the budget.  
 
State Space: All possible combinations of robots with their skills and costs.
 
Initial State: An empty list with no robot selected with given budget.

Goal State: A team of robots within the provided budget with the maximum amount of skill. 

Successor Function: This function will generate a list of robots at each state, with total cost within the budget. This function makes sure that no duplicates are added and the the total cost never exceeds the budget.

Cost function:The total cost and total skills of the combination of robots at any state is the cost function.
At each newly generated state i.e. when a robot gets appended to an already existing combination of robots, we add the skills and the cost to the respective total skill and cost at that state, provided the total cost is within the budget. 

2. Brief description of how our search algorithm works:
Ans:
For this search problem, we have implemented the Breadth First Search method.
At first, we sort the dictionary on the basis of skill to cost ratio. We traverse the tree on that basis and create combinations of robots such that the best robots with best ratio get appended first.

We use the same sorted list to append the robots which helps us prune the tree efficiently as explained below.
If the list of robot (in the descending order of skill to cost ratio) is A,B,C,D,E, then, we create nodes AB,AC,AD,AE along with the index of the last added robot in the list, i.e. for the combination AB, we will append 1 as the index. Then in the next iteration, we make sure that our algorithm traverses the list starting from the index provided, i.e. For the node AB which has 1 as the appended index, the traversal will start from index 2. This helps us save the time by not traversing the nodes which have been already appended/visited. This way, we make sure that all the nodes get appended only once, without having to maintain a separate data structure to store the visited nodes.  

For every iteration, we maintain a highest node which gives us the maximum skill within the budget. If we find a node that yields a better solution, we replace that with the highest skill node. The best combination obtained is the solution.

Problems faced:
We initially maintained a visited node list, that helped us eliminate the duplicated nodes. But this was a time-consuming and a memory consuming problem. So, instead of maintaining such a list, we made sure we do not traverse already appended nodes by updating the index while traversing through the list.
Another problem that we faced was the time the algorithm took to complete due to the extensive usage of lists. We replaced it using the tuples and thus we decreased the running time as well.  
Then, we faced an issue where we traversed the list from the beginning every time we wished to append a robot. This operation resulted in unnecessary traversal through the already appended nodes. So instead we kept a track of the index of the last appended node.  
