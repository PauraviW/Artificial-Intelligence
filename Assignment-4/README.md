*The topmost rows of each algorithm indicate the max accuracy obtained on the testing set*
# K-Nearest Neighbour
The parameters that provided the best accuracy are as follows:   

| Strategy      | Neighbours    | Batch-size  | Accuracy(%) | Running Time(seconds) |
| ------------- |:-------------:| -----------:|------------:| ---------------------:|
| **Euclidean     | 55            | 1000        | 71.36       | 6.96**               |
| Euclidean     | 100           | 1000        | 70.30       | 8.34                  |
| Euclidean     | 100           | 100         | 70.30       | 7.17                  |
| Euclidean     | 3             | 1000        | 68.7        | 6.88                  |

As we can see here, that increasing the number of neighbours improves the accuracy. I would suggest a batch size of 1000 and 55 nearest neighbours. 

To Run:

./orient.py train train-data.txt knn_model.txt nearest

./orient.py test test-data.txt knn_model.txt nearest

Incorrectly classified images  

![](knn-wrong/9623930399_525ddf3a3b_m.jpg)    
![](knn-wrong/9716486235_202a2087de_m.jpg)  

![](knn-wrong/9926949024_82eb20b155_s.jpg)

As we can see here, these images are difficult to classify even for us as the pixel intensities accross the image is constant. Thus, a the network could not distinguish between the photos.  
Correctly Classified  
![](knn-right/9620904035_b3ff37b6c6_m.jpg)    
![](knn-right/9694305901_782c62dfdb_m.jpg)  

![](knn-right/9831599156_edc4eb763f_m.jpg)

*We could not upload the trained model for KNN as its size exceeded 100 MB*
# Neural Networks

### suported/configurable features
- Activation fucntions: Relu, Sigmoid, Softmax
- Loss: Categorical Cross Entropy
- Gradient Descen optimazation using RMSProp (if set to True)

**Architecture: (Best) 

In this we have used 4 layers. 

| Layers        | Activation function 
| ------------- |:-------------------:
| Layer 0       | relu                
| Layer 1       | relu              
| Layer 2       | relu                
| Layer 3       | softmax             

RMSProp is set to False for all below runs.

| learning rate  | Epoch| Accuracy(%) | Running Time(seconds) | Batch Size
| --------------:|-----:|------------:| ---------------------:|-----------:|
|   **  0.0005     |  100 | 72.6       | 0.13                  |     1
|     0.00005    |  80  | 68.61       | 0.145                 |    128
|     0.000005   | 150  | 66.17       | 0.12                  |   128

To Run:  ./orient.py test test-data.txt nnet_model.txt nnet

Incorrectly classified images  

![](nn-wrong/9926949024_82eb20b155_m.jpg)    
![](nn-wrong/9995085083_caaedd981c_m.jpg)  

![](nn-wrong/9716486235_202a2087de_m.jpg)

As we can see here, these images are difficult to classify even for us as the pixel intensities accross the image is constant. Thus, a the network could not distinguish between the photos.  
Correctly Classified  
![](nn-right/9646375952_6dc31aa001_m.jpg)    
![](nn-right/9760490031_5509d5779f_m.jpg)  

![](nn-right/9831599156_edc4eb763f_m.jpg)


# Decision Tree


| Max - Depth   | Minimum- leaves |  Accuracy(%) | Running Time(seconds) |
| ------------- |:---------------:| ------------:| ---------------------:|
| **10            | 10              |    67.23     | 3.96**                  |
| 100           | 2               |    62.56     | 9.34                  |
| 200           | 100             |    58.66     | 8.16                  |
| 15            | 76              |    60.87     | 1.58                  |

To Run:  ./orient.py test test-data.txt tree_model.txt tree

Incorrect classification  

![](Decision-tree-wrong/10313218445_8a5107f499_m.jpg)    
![](Decision-tree-wrong/10352491496_1923908631_m.jpg)  

![](Decision-tree-wrong/10484444553_c200554bb3_m.jpg)

Correct classification  

![](Decision-tree-right/10099910984_9fe6cd5969_m.jpg)    
![](Decision-tree-right/10164298814_9cc7895a0f_m.jpg)  

![](Decision-tree-right/102461489_b58c3a4bfa_m.jpg)


Our best model is using neural network which gives us the accuracy of 72.6% with learning rate 0.0005 and 100 epochs. We would recommend this.
