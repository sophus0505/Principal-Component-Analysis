# IN3050/IN4050 Mandatory Assignment 3: Unsupervised Learning

**Name:**

**Username:**


# Principal Component Analysis (PCA)
In this section, you will work with the PCA algorithm in order to understand its definition and explore its uses.

### Principle of Maximum Variance: what is PCA supposed to do?
First of all, let us recall the principle/assumption of PCA:

 1. What is the variance?
     - The variance is a measure of how far away a dataset spreads out from its mean value. It is calculated by taking the square of the standard deviation.
 2. What is the covariance?
     - The covariance measures the relation between two datasets. If the covariance is high, it means that the two datasets are likely to have similar values (high/low) at the same time (they move in the same direction).
       If the covaiance is negative, it means that the datasets tend to have different values at the same time (they move in opposite directions). 
 3. How do we compute the covariance matrix?
     - Lets say we have a dataset $X \in \mathcal{R}^{N \times M}$, consisting of $N$ datapoints $\mathbf{x}_i = (\mathbf{x}_{1,i} \;, \mathbf{x}_{2,i}\;, \dots, \mathbf{x}_{M,i})$ with $M$ features.
       To calculate the covariance matrix $Cov(X)$ it would first be beneficial to centre the data in a new matrix $P$ by subtracting each column by its mean. 
       The covariance matrix can then be computed as $ Cov(X) = 1/N \cdot P^TP$.
 4. What is the meaning of the principle of maximum variance?
     - The principle of maximum variance means that we should rotate the dataset such that one of the axis point in the direction of the maximum  variance. 
 5. Why do we need this principle?
     - the principle of maximum variance should be followed to make sure that we keep as much information as possible when we remove dimensions from the original          dataset. If we don't follow the principle, we might end up with a clustering of non-separable data, which would be hard to analyze. 
 6. does the principle always apply?
     - Yes, because we always want to keep information about the different features in the dataset, and using the principle of maximum variance keeps the maximum amount of information.

## Task 1: Implementation of PCA

We implement the PCA algorithm by using the framework from the notebook and instructions given in Marshland. 
the algorithm can be summarized in the following steps: 

1. Collect the data in a matrix $X \in \mathcal{R}^{N \times M}$, where every column represents an attribute. 
2. Centre the data by subtracting the mean of each column off X and call the new matrix $B$.
3. Compute the covariance matrix $ Cov(X) = 1/N \cdot P^TP$.
4. Compute the eigenvalues and eigenvectors of $Cov(X)$.
5. Sort the eienvalues in decreasing order, and apply the same order to the matrix of eigenvectors. 
6. Remove the last eienvectors according to how many ($m$) attributes you want to keep. 

Below is an implementation of the PCA algorithm that follows the instructions above.

```python
def pca(A,m):
    # INPUT:
    # A    [NxM] numpy data matrix (N samples, M features)
    # m    integer number denoting the number of 
    #      learned features (m <= M)
    #
    # OUTPUT:
    # pca_eigvec [Mxm] numpy matrix containing 
    #               the eigenvectors (M dimensions, m eigenvectors)
    # P          [Nxm] numpy PCA data matrix 
    #               (N samples, m features)
    
    B = center_data(A)
    C = compute_covariance_matrix(B)
    eigval, eigvec = compute_eigenvalue_eigenvectors(C)
    sorted_eigval, sorted_eigvec = sort_eigenvalue_eigenvectors(
                                    eigval, 
                                    eigvec
                                   )
    pca_eigval = sorted_eigval[:m]
    pca_eigvec = sorted_eigvec[:,:m]
    P = pca_eigvec.T @ B.T

    return pca_eigvec, P.T
```
\newpage

Next we test the algorithm on some test data. We load a small synthetic dataset with two features from the module *syntheticdata*.
The data is visualized in figure \ref{fig: data1} below. 

    
![A figure of the test data.\label{fig: data1}](output_36_0.png){ width=320px }
    
A figure showing the data after centering is shown in figure \ref{fig: data1 centered} below.

    
![The test data after being centered.\label{fig: data1 centered}](output_38_0.png){ width=320px }
    
\newpage

### Visualize the first eigenvector

We want to visualize how the PCA algorithm works on the synthetic dataset. We use our PCA function to calculate the new data and the eigenvector that points in the direction of maximum variance. The results are shown in figure \ref{fig: data1 eigvec} and figure \ref{fig: data1 PCA}.
    
![The first dataset with the corresponding eigenvector pointing in the direction of maximum variance. \label{fig: data1 eigvec}](output_40_1.png){ width=320px }
    

![The result on the first dataset after running it through PCA. \label{fig: data1 PCA}](output_42_0.png){ width=320px }
    
\newpage

## Evaluation: when are the results of PCA sensible?
So far we have used PCA on synthetic data. Let us now imagine we are using PCA as a pre-processing step before a classification task. This is a common setup with high-dimensional data. We explore when the use of PCA is sensible.

We load a similar set of data, with labels and perform the same steps as above. The results are shown in figure \ref{fig: data2} and figure \ref{fig: data2 PCA}.

    
![Dataset with two classes. \label{fig: data2}](output_47_0.png){ width=320px }
    

![The second dataset after being run though PCA. \label{fig: data2 PCA}](output_47_1.png){ width=320px }
    


**Comment:** 
 - We can see that the dataset is almost separable and have a high variance, both are characteristics that are preserved after using PCA. In this case it seems that the use of PCA is reasonable. 

 \newpage

### Loading the second set of labels
We now load a third dataset with different labels and run the procedure as normal. 
This time we also plot the direction of the eigenvectors as well as their sum (which is also an eigenvector). 
The dataset is shown in figure \ref{fig: data3}, and the results after PCA is shown in figure \ref{fig: data3 PCA}.

    
![The third dataset, with the direction of the eigenvectors and their sum plotted on top. \label{fig: data3}](output_52_0.png){ width=400px }
    
    
![The results after running PCA on the third dataset. \label{fig: data3 PCA}](output_52_1.png)
    
**Comment:** 
 - We see that using the first eigenvector we get bad separation between the classes as they lie on top of each other in the direction of maxmimum variance. However the result is more separable when using the second eigenvector, which means that it might be more beneficial in this case to use the second eigenvectore instead. The second eigenvector is orthogonal to the first eigenvector (it looks skewed because of the plotting) so it will preserve more information from the data after PCA. Lastly we see that if we add the two vectors together we get a third vector that is also by definition an eigenvector, that can be used instead. It it represented in the figure above, and would possibly give even better separation in 1D. 

- To quantify how separable the two resulting 1D datasets are we use a linear percetron algorithm to see how well it is able to separate the data. When running it we get the following result:

```terminal
Score of perceptron using the first eigenvec  = 0.5
Score of perceptron using the second eigenvec = 0.84
```
 - We have considered both the first and second eigenvecors and also made an assumption about the sum of the two. We see in the figure above that the second eigenvector produces the best results. We have also used the perceptron algorithm to study how separable the two resulting 1D datasets are. The results show that using the first eigenvector we get a precision of 0.5 (equivalent to guessing) while the second eigenvector gives a precision of 0.84, which is significantly better. 

## Case study 1: PCA for visualization
We now consider the *iris* dataset, a simple collection of data (N=150) describing iris flowers with four (M=4) features. The features are: Sepal Length, Sepal Width, Petal Length and Petal Width. Each sample has a label, identifying each flower as one of 3 possible types of iris: Setosa, Versicolour, and Virginica.

Visualizing a 4-dimensional dataset is impossible; therefore we will use PCA to project our data in 2 dimensions and visualize it.
To do this we select two random features and show them in a figure before and after PCA.
The results are shown in figure \ref{fig: iris} and figure \ref{fig: iris PCA}.


    
![A visualization of two randomly chosen features from the iris dataset before PCA. \label{fig: iris}](output_61_0.png){ width=320px }
    
    
![A visualization of the two randomly chosen features from the iris dataset after PCA. \label{fig: iris PCA}](output_63_0.png){ width=320px }
    


**Comment:** 
 - We see that the dataset consists of almost linearly separable data, and that the PCA tends to precerve this . 



## Case study 2: PCA for compression
We now consider the *faces in the wild (lfw)* dataset, a collection of pictures (N=1280) of people. Each pixel in the image is a feature (M=2914).
We use PCA on the images to compless them by removing potentially unnecessary features and then reconstruct the image back into its original size. The results using different values of $m$ is shown in figure \ref{fig: faces}.


![\ ](output_79_0.png){ width=350px margin=auto}
\ 
  
![\ ](output_79_1.png){ width=350px}
\   
  
![\ ](output_79_2.png){ width=350px}
\ 

![\ ](output_79_3.png){ width=350px}
\ 

  
![Faces from the dataset with PCA compression for different values of $m$.\label{fig: faces}](output_79_4.png){ width=350px}
    


**Comment:**
 - We see that (not surprisingly) the images become more distorted and less recognizable for lower values of m. 
   This is because we effectively remove features from the faces using PCA and only keep the m most relevant features.
   The interesting part is that even though the faces appear weird and distorted for low values of m, they are still recognizable, which means that they should still be usable as a simplified dataset. 

## Master Students: PCA Tuning
If we use PCA for compression or decompression, it may be not trivial to decide how many dimensions to keep. In this section we review a principled way to decide how many dimensions to keep.

The number of dimensions to keep is the only *hyper-parameter* of PCA. A method designed to decide how many dimensions/eigenvectors is the *proportion of variance*:
$$ \textrm{POV}=\frac{\sum_{i=1}^{m}{\lambda_{i}}}{\sum_{j=1}^{M}{\lambda_{j}}}, $$
where $\lambda$ are eigenvalues, $M$ is the dimensionality of the original data, and $m$ is the chosen lower dimensionality. 

Using the $POV$ formula we may select a number $M$ of dimensions/eigenvalues so that the proportion of variance is, for instance, equal to 95%.

Implement a new PCA for encoding and decoding that receives in input not the number of dimensions for projection, but the amount of proportion of variance to be preserved.

We use the *proportion of variance* method to decide the value of $m$ to use in the PCA algorithm. We set the proportion of variance equal to 0.9 which we use to calculate $m$. The result is that $m=85$.
An example face before and after tuning is given in figure \ref{fig: face} and figure \ref{face tuned}.
    
![Face before PCA tuning. \label{fig: face}](output_86_0.png)

    
![Face after PCA tuning using $m=85$. \label{fig: face tuned}](output_86_1.png)

**Comment:** 
 - We see that the tuning sets m to 85 wich is a low value and leads to a pretty distorted face. This might mean that the algorithm thinks that is enough dimensions to keep while still keeping enough variance in the data. 

# K-Means Clustering (Bachelor and master students)
In this section you will use the *k-means clustering* algorithm to perform unsupervised clustering. We perform the algorithm on the iris data used before. The results are shown in figure \ref{fig: iris data} and figure \ref{fig: iris cluster}.

    
![Original data from the iris dataset. \ref{fig: iris data}](output_99_0.png){ width=350px }
    
![Results after K-Means clustering for different $k$-values. \ref{fig: iris cluster}](output_99_1.png)
    


**Comment:**
 - We see that the K-Means algorithm manages to separate clusters effectively and that for k=3 it resembles the original data quite accurately. However, it does not know how to label the clusters, so the actual accuracy is close to useless. 

# Quantitative Assessment of K-Means (Bachelor and master students)

We are given the following tasks to perform:
- Train a Logistic regression model using the first two dimensions of the PCA of the iris data set as input, and the true classes as targets.
- Report the model fit/accuracy on the training set.
- For each value of K:
  - One-Hot-Encode the classes output by the K-means algorithm.
  - Train a Logistic regression model on the K-means classes as input vs the real classes as targets.
  - Calculate model fit/accuracy vs. value of K.
- Plot your results in a graph and comment on the K-means fit.


After training a logistic regression model using the first two dimension of the PCA of the iris data set as input we get the following result:

```
The accuracy score on the training set is: 0.967
```

Now we perform the K-Means algorithm for $k = 2, 3, 4, 5$ and report the accuracy after using a logistic regression model to accurately label the clusters. We get the following results:


```
    k = 2, acc = 0.667
    k = 3, acc = 0.887
    k = 4, acc = 0.840
    k = 5, acc = 0.900
```

Then we visualize how the original data compare to the different results using K-Means clustering for all values of k,
the results are shown in figure \ref{fig: comp}.
\newpage
    
![png](output_110_1.png)
\ 
![png](output_110_2.png)
\ 
![png](output_110_3.png)
\ 


![How the original data compare to the different results ising K-Means clustering for different $k$-values. \label{fig: comp}](output_110_4.png)

    

Lastly we plot the accuracy as a function of $k$ in figure \ref{fig: acc}
    
![The accuracy of K-Means clustering as a funciton of $k$. \label{fig: acc}](output_111_1.png)
    


**Comment:** 
 - We see that using K-Means clustering is pretty effective for labeling the data. We get a good accuracy of 90% already for k=5 and would excpect that to be higher for larger values of k. We do, however, have to use logistic regression for labeling the data correctly and since the logistic regression classifier performed with an accuracy of 96% I wonder why we don't just use that alone instead. 
