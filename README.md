# machine-learning-1-week-13-latent-variable-models-clustering-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 1 Week 13-Latent Variable Models Clustering Solved](https://www.ankitcodinghub.com/product/machine-learning-1-week-13-latent-variable-models-clustering-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98772&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 1 Week 13-Latent Variable Models Clustering Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="section">
<div class="layoutArea">
<div class="column">
Weighted K-Means Clustering

In this exercise we will simulate finding good locations for production plants of a company in order to minimize its logistical costs. In particular, we would like to place production plants near customers so as to reduce shipping costs and delivery time.

We assume that the probability of someone being a customer is independent of its geographical location and that the overall cost of delivering products to customers is proportional to the squared Euclidean distance to the closest production plant. Under these assumptions, the K-Means algorithm is an appropriate method to find a good set of locations. Indeed, K-Means finds a spatial clustering of potential customers and the centroid of each cluster can be chosen to be the location of the plant.

Because there are potentially millions of customers, and that it is not scalable to model each customer as a data point in the K-Means procedure, we consider instead as many points as there are geographical locations, and assign to each geographical location a weight wi corresponding to the number of inhabitants at that location. The resulting problem becomes a weighted version of K-Means where we seek to minimize the objective:

∑iwi mink | | xi − ck | | 2 J(c1, …, cK) = ∑ w ,

ii

where ck is the kth centroid, and wi is the weight of each geographical coordinate xi. In order to minimize this cost function, we iteratively perform the

following EM computations:

Expectation step: Compute the set of points associated to each centroid:

∀1≤k≤K: C(k)←{i:k=argmin‖xi−ck‖2} k

Minimization step: Recompute the centroid as a the (weighted) mean of the associated data points: ∀1≤k≤K: ck←∑i∈C(k)wi⋅xi

∑i∈C(k)wi

until the objective J(c1, …, cK) has converged. Getting started

Inthisexercisewewillusedatafromhttp://sedac.ciesin.columbia.edu/(http://sedac.ciesin.columbia.edu/),thatwestoreinthefiles data.mat aspartof the zip archive. The data contains for each geographical coordinates (latitude and longitude), the number of inhabitants and the corresponding country. Severalvariablesandmethodsareprovidedinthefile utils.py:

utils.population A 2D array with the number of inhabitants at each latitude/longitude.

utils.plot(latitudes,longitudes) Plot a list of centroids given as geographical coordinates in overlay to the population density map. The code below plots three factories (white squares) with geographical coordinates (60,80), (60,90),(60,100) given as input.

</div>
</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="section">
<div class="layoutArea">
<div class="column">
In [1]:

import utils, numpy

%matplotlib inline utils.plot([60,60,60],[80,90,100])

Also, to get a dataset of geographical coordinates associated to the image given as an array, we can use:

In [2]:

<pre>x,y = numpy.indices(utils.population.shape)
locations = numpy.array([x.flatten(),y.flatten()]).T
</pre>
Initializing Weighted K-Means (25 P)

Because K-means has a non-convex objective, choosing a good initial set of centroids is important. Centroids are drawn from from the following discrete probability distribution:

P(x, y) = Z1 ⋅ population(x, y)

where Z is a normalization constant. Furthermore, to avoid identical centroids, we add a small Gaussian noise to the location of centroids, with standard

deviation 0.01. Task:

Implement the initialization procedure above.

In [3]:

def initialize(K,population):

# YOUR CODE HERE

import solution

centroids = solution.initialize(K, population) return centroids

<pre>    # --------------
</pre>
The following code runs the initialization procedure for K=200 clusters and visualizes the centroids obtained with the initialization procedure using utils.plot .

</div>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="layoutArea">
<div class="column">
In [4]:

<pre>centroids_init = initialize(200, utils.population)
utils.plot(centroids_init[:,0], centroids_init[:,1])
</pre>
Implementing Weighted K-Means (75 P) Task:

Implement the weighted K-Means algorithm. Your algorithm should run for nbit iterations and print the value of the objective after training. If verbose , it should also print the value of the objective at each iteration.

In [5]:

def wkmeans(centroids, points, weights, verbose, nbit):

# YOUR CODE HERE

import solution

centroids = solution.wkmeans(centroids, points, weights, verbose, nbit) return centroids

<pre>    # --------------
</pre>
The following code runs the weighted k-means on this data, and displays the final centroids.

<pre>In [6]:
weights   = utils.population.flatten()*1.0
</pre>
centroids = wkmeans(centroids_init, locations, weights, True, 50) utils.plot(centroids[:,0], centroids[:,1])

</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="section">
<div class="layoutArea">
<div class="column">
it= 1:J= 12.66 it= 2:J= 10.08 it = 3: J = 8.39 it = 4: J = 7.82 it = 5: J = 7.52 it = 6: J = 7.35 it = 7: J = 7.22 it = 8: J = 7.14 it = 9: J = 7.10 it=10:J= 7.07 it=11:J= 7.04 it=12:J= 7.01 it=13:J= 6.98 it=14:J= 6.96 it=15:J= 6.95 it=16:J= 6.95 it=17:J= 6.94 it=18:J= 6.94 it=19:J= 6.93 it=20:J= 6.93 it=21:J= 6.93 it=22:J= 6.93 it=23:J= 6.93 it=24:J= 6.93 it=25:J= 6.93 it=26:J= 6.93 it=27:J= 6.93 it=28:J= 6.93 it=29:J= 6.93 it=30:J= 6.93 it=31:J= 6.93 it=32:J= 6.93 it=33:J= 6.93 it=34:J= 6.93 it=35:J= 6.93 it=36:J= 6.93 it=37:J= 6.93 it=38:J= 6.93 it=39:J= 6.93 it=40:J= 6.93 it=41:J= 6.93 it=42:J= 6.93 it=43:J= 6.93 it=44:J= 6.93 it=45:J= 6.93 it=46:J= 6.93 it=47:J= 6.93 it=48:J= 6.93 it=49:J= 6.93 it=50:J= 6.93

</div>
</div>
<div class="layoutArea">
<div class="column">
Observe that the k-means algorithm is non-convex, and arrives in local optima of different quality depending on the initialization:

</div>
</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="section">
<div class="layoutArea">
<div class="column">
In [7]:

for i in range(5):

wkmeans(initialize(200, utils.population), locations, weights, False, 50)

it=50:J= 6.57 it=50:J= 7.37 it=50:J= 8.33 it=50:J= 8.10 it=50:J= 7.77

</div>
</div>
</div>
</div>
