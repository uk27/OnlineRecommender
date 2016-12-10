# Tutorial on Spark MLib

#### Members:
- Utkarsh Kajaria
- Eric Weber
- Eric Passe
- Yuemin Xu

#### Topic
> Overall Problem Statement

Some Description

## Tutorial 1:  Online Recommendation Engine

In this tutorial we will learn how to use the Spark machine learning library Mlib to build a movie recommendation engine. Not only this, we will take our engine live by hosting it as a web service on AWS. For this tutorial, we use the popular MovieLens data set. A description of this dataset can be found at the Grouplens [website](http://grouplens.org/datasets/movielens/latest/).


For the sake of simplicity, we have broken this into three parts: In Section 1.1 we will walk through the code for our recomendation engine. In Section 1.2 we will talk about exposing the engine utilies as web service. In Section 1.3, we will see how to access this service in real-time by hosting it on AWS.

### 1.1 The Engine

We will refer to the part that exclusively deals with training and providing recommendations by using Spark MLib as the engine. Before delving into how the engine is constructed, we need to be familiar with a few terminologies:
  
  **_Collaborative Filtering_** : A set of technologies that predict which items in a set of products (or information) a particular customer will like based on the preferences of lots of other people i.e. If I know your movie preferences, and those of lots of other people, I can make pretty good movie recommendations for you. [1]

  An obvious idea would be to look at a user's preferences and those in his *neighbourhood* of people and use the ratings given by these people to recommend new movies to our user. In fact, it was this idea on which most of the early recommendation algorithms were based upon. Over time people have developed more sophisticated ways of computing essentially the same thing. One of these is the Alternating Least Squares method.

  **_ALS_** : The idea behind ALS is that instead of building a list of neighbors for you, we use everyone’s ratings to form some type of intermediate representation of movie tastes — something like a taste space in which we can place you, and the movies — and then then use that representation to quickly evaluate movies to see if they match this community-based representation of your tastes. This taste space is called latent factors. [2]


The engine will have the foloowing five major tasks:

  1. Training the ALS model
  2. Predict ratings for a given movie for a user
  3. Get the top n rated movies for a user
  4. Add the ratings for a new user


###### Figure 1: Code for XX
```Python

#Python Code comes here
def doSomething():
  #line 1
  #line 2
  return 

```

Code Description


### 1.2 The Engine as a service


The recommender in itself is just a set of methods that provide us with the nescessary functionality. However, to be able to make use of it as a web-service, we need a way to translate a web request into one or more of the functions exposed by the recommender. This is where [Flask](http://flask.pocoo.org) comes in. 
Flask is a web application framework in python which consists of libraries using which we map the web-requests to appropriate methods of our recommendation engine.

This mapping is done in the python file named app.py. Let's look at one of them:

###### Figure 2: app.py::
```Python

@main.route("/<int:user_id>/ratings/top/<int:count>", methods=["GET"])
def top_ratings(user_id, count):
    logger.debug("User %s TOP ratings requested", user_id)
    top_ratings = recommendation_engine.get_top_ratings(user_id,count)
    return json.dumps(top_ratings)

```

In the above code snippet, the argument to main.route() is essentially a url. The url itself contains two parameters: user_id and count. Basically, when we access this url, we are asking the server to return the top 'count' number of recommendations for the user 'user_id'. When the server recieves this request, it passes these parameters to the top_rating() method of the engine which computes and returns the result which is outputted to the browser.


You would have noticed that we need to access the top_rating() method using an object recommendtion_engine. This means we have to create this object first. This and other standard Flask initializations ([see docs](http://flask.pocoo.org/docs/0.10/blueprints/#the-concept-of-blueprints)) are done in the create_app() method as shown below:


```Python
def create_app(spark_context, dataset_path):
    global recommendation_engine 

    recommendation_engine = RecommendationEngine(spark_context, dataset_path)    
    
    app = Flask(__name__)
    app.register_blueprint(main)
    return app 

```

After registering the services in the app.py, we need to create a server that will be serving these requests. We do this using [CherryPy](http://cherrypy.org), an immensely popular [WSGI](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface) framework.

In our tutorial, this is done in server.py. 

```Python
if __name__ == "__main__":
    # Init spark context and load libraries
    sc = init_spark_context()
    dataset_path = os.path.join('s3a://project-bucket-group5/data', 'ml-latest')
    app = create_app(sc, dataset_path)
 
    # start web server
    run_server(app)

 ```

This python file is our main server where we:

* Configure and initialize SparkContext (the init_spark_context() method).
* Specify the data source (i.e. dataset_path) by giving the path to an S3 bucket that contains all the ratings data downloaded from the Grouplens website.
* Create the flask application that we created in the previously (the create_app() method).
* Configure and start the server by specifying the host and port number (the run_server(app) method).

For a detailed implementation please look at the server.py file.
 
### 1.3 The Engine as a service hosted on AWS

Hosting the service on AWS might appear to be a complicated process if you are doing it for the first time. However, if you closely follow the steps below, you can be up and running in a matter of minutes! Let's see how it's done.

1. Log on to AWS console

2. Click on EMR 


  <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/EMR.png" alt="alt text" width="700" height="250">


3. Create a cluster:

  1. Enter name of your cluster

  2. Select the S3 folder for logs. <Link to how to create a bucket> 


    <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/create_cluster_1.png" alt="alt text" width="800" height="250" >


  3. Choose the suite of Applications that installs Spark.

    <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/create_cluster_2.png" alt="alt text" width="700" height="250" >

  4. In the Security and Access tab, choose your key pair from the drop down.

    <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/create_cluster_3.png" alt="alt text" width="800" height="300" >

  5. Leave the default value for everything else.

4. Open an ssh connection from the command line. For this: 

  1. Click on the ssh on your cluster dashboard to find the command for the terminal. 

    <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/ssh.png" alt="alt text" width="600" height="200" >

  2. Open an ssh connection with the cluster using your terminal. Please make sure you navigate to the folder that contains your key before running this command. 

    <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/Terminal.png" alt="alt text" width="600" height="300" >


5. Install additional software on the cluster. For this tutorial we will need:

  * Git : Source Version Control
  * CherryPy : A microframework in Python that exposes REST services. We use this to create our server.
  * Paste :  Logging Tool
  * Flask: Another REST enabling microframework in Python. We use it to bind services with the urls.

  Use the following commands :

  ``` 
  sudo yum install git
  sudo pip install cherrypy
  sudo pip install paste
  sudo pip install flask
  ```

6. Get the recommender code by cloning the repository. Clone the code by using:

  ``` 
  git clone https://github.com/uk27/OnlineRecommender.git
  ```
  
  This will create a folder called OnlineRecommender that will contain four files: 

  * engine.py : This contains the logic for collaborative filtering using Mlib
  * app.py :  This contains the logic for routing the web requests and calling appropriate functions. This is essentialy where the flask logic is.
  * server.py : This is where we configure and initialize the spark context, give the host and port of the service start the app.  By default it uses 5432 as port number.
  * user_rating.file : This contains the ratings information for a new user that we will be adding to the existing set.

7. Navigate into the directory containing the source code 

  ```
  cd OnlineRecommender/src
  ```

8. Run the server by typing in the code:

  ```
  spark-submit server.py
  ``` 
  This essentially runs the server.py which initializes SparkContext, trains our recommender and starts listening on the port 5432.

9. We need to tell EC2 that it needs to accept the requests on the port 5432 and route it to the master node. For this:

  1. Go to EC2. 
  2. Under Networks and Security choose Security groups 

    <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/SecurityGroup.png" alt="alt text" width="700" height="200" >


  3. Select the master node from the list

  4. In the inbound tab, select edit.

    <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/editInbound.png" alt="alt text" width="800" height="300" >

  5. Click on add a new rule. Select TCP rule, type '5432' for port number and  select source as 'Anywhere' from the drop down.

    <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/Addrule.png" alt="alt text" width="800" height="150" >

  Note: The port number entered here needs to be same as the port number in server.py

10. Now the only thing that we need to access the service from a web browser is the public ip address of the master node. EMR by default assigns a public ip to each instance in the cluster unless otherwise specified. This can be found as follows:

  1. Go to EC2 and click on instances.

    <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/Instances.png" alt="alt text" width="700" height="250" >


  2. In the tab that appears, scroll to the right to find the public ip.

    <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/publicip.png" alt="alt text" width="800" height="100" >

    
11. That’s it! We can start using the service now. In your browser paste any of the following:

  * Retrieve the top ten ratings for the user number 1.
  
    ``` 
    http://<Public-IP-Address-Of-The-Master>:5432/1/ratings/top/10
    ```
    
  * Retrieve the rating for the movie 'The Quiz' for user 1.
    
    ```
    http://<Public-IP-Address-Of-The-Master>:5432/1/ratings/500
    ```
    
    
  * We can also add a new user to the existing datatset set by associating a set of ratings with a new user number. To do this, we use the command line as we need to pass the list of movies to the our engine.

    ```   
    curl --data-binary @user_ratings.file http://<Public-IP-Address-Of-The-Master>:5432/0/ratings
    ```
    
12. After playing with your new web-service, remember to terminate the cluster.

  <img src="https://github.com/uk27/OnlineRecommender/blob/master/images/Terminate.png" alt="alt text" width="700" height="200" >

    

## 2  Second Tutorial
> Problem Statement

## 2  Second Tutorial
> Problem Statement

Some Description

### 2.1 Subsection Topic

Description comes here.


###### Figure 3: Code for XX
```Python

#Python Code comes here
def doSomething():
  #line 1
  #line 2
  return 

```

Code Description


### 2.2 Subsection Topic

Description comes here.


###### Figure 4: Code for XX
```Python

#Python Code comes here
def doSomething():
  #line 1
  #line 2
  return 

```

Code Description

## 3  Third Tutorial: Guassian Mixture Modeling
> The purpose of this tutorial is to use Spark to identify "hidden" distributions that make up the overall distribution of data present. Gaussian Mixture Modeling is a process quite similar to k-means clustering in that it produces random parameter estimates of a given number of underlying normal distributions that comprise the distribution of the data observed.

 Gaussian Mixture Modeling relies on a method called Expectation Maximization. In effect, this algorithm assumes each data point is a linear combination of multivariate Guassian distributions. Like with k-means, it is an iterative algorithm that stops once it reaches a "best" solution, though that solution may not be globally the "best". The basic idea is we have M clusters and points in that cluster follow a normal distribution. 

 If you are familiar with k-means, this algorithm is not a huge step. Instead of using hard assignments to clusters, we now use soft assignment, where each point has a probability of belonging to a particular cluster. As with most algorithms, there are some key assumptions that must be met. First, the dataset should be Gaussian. What happens when this assumption is not met? If the number of clusters is known in advance, the algorithm is not guaranteed to find the "right" clusters. If the number of clusters is unknown, this actually is a bit better! If those clusters have a normal distribution then an information criterion is typically good enough to find the best solution. Second, the cluster sizes must be relatively close to each other. If the clusters are not the same size, a larger cluster will show up as more important than any smaller cluster.


### 3.1 Introduction to GMM Using Spark

SparkMLLib contains a number of powerful methods for implementing algorithms that use Expectation Maximization, both for k-means and Gaussian Mixture Modeling. The following example builds on an example dataset provided by Spark and saved for you as "gmm_data.txt". 

It is important to know there are key parameters in control of the user. The SparkMLLib documentation identifies them as follows:

- **k** is the number of desired clusters.
- **convergenceTol** is the maximum change in log-likelihood at which we consider convergence achieved.
- **maxIterations** is the maximum number of iterations to perform without reaching convergence.
- **initialModel** is an optional starting point from which to start the EM algorithm. If this parameter is omitted, a random starting point will be constructed from the data.

The following code assumes you are working on AWS. Please refer to the directions for AWS provided in this document for submission directions to AWS. Please keep in mind you will need to have the _gmm_data.txt_ file uploaded to your S3 bucket for this work.

We first import the necessary functions. These include a print function for displaying the Spark results in a reasonable format on AWS, an operating system import function, a Spark Context and the Guassian Mixture Model functions appropriate for pyspark. Please note that if you would like to complete these in Scala, the documentation is available on the SparkMLLib online reference guide.

```Python
from __future__ import print_function
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
```

In the next step, we create a configuration for Spark which allows us to create an app name here. For more information on SparkConf see this [reference]
(https://spark.apache.org/docs/1.6.2/api/java/org/apache/spark/SparkConf.html). We also create a Spark context based on this Spark configuration which allows us to then use SparkMLLib which comes as part of the base build for Spark.


```Python

conf = SparkConf().setAppName("Gaussian Mixture Modeling")
sc = SparkContext(conf=conf)

```

The following commands demonstrate how to point to the S3 bucket in AWS, create a file path and use that file path to generate an RDD called gmmData that contains the observations in the dataset. It is worthwhile to confirm that the data was read in correctly. Feel free to call gmmData.take(5) to make sure the first five observations match what is in the file itself.

```Python

datasets_path = "s3a://bigdataweberproject"
data_file = os.path.join(datasets_path,'gmm_data.txt')
gmmData = sc.textFile(data_file)

```

The next step ensures that the data are in the appropriate format for the Gaussian Mixture Model in Spark. The algorithm works with data of type float. The following code calls the gmmData RDD, uses map to iterate over each of its lines, and in each line, converts each object to type float. Note that the split function assumes the data is space separated, not comma separated as is often present in Spark's dataset examples.

```Python

formattedData = gmmData.map(lambda line: [float(x) for x in line.strip().split(' ')])

```

Up to this point, we have created an RDD called _formattedData_ that consists of observations from the original data in float format. The next step demonstrates the efficiency of code using SparkML, as we train the model on the formattedData RDD with the assumption that there are two underlying normal distributions. It is critical to note that the number of distributions is user specified here. This is also known as the number of desired clusters. This number often requires knowledge of the data and perhaps previous research related to similar dataset types. This returns _gmm_model_ which contains information about the clusters.

```Python

gmm_model = GaussianMixture.train(formattedData, 2)

```

With the _gmm_model_ created, we can do a number of things on the model object itself. Specifically, we can call _weights_, _gaussians_ and _gaussians.sigma_ to obtain parameters from the model. The following print call displays (for each clusters) the weights associated with the cluster, and the mean and variance parameters for the Gaussian distribution (mu and sigma, here). For ease of use, we have not output the parameter values to a text file. The values will just display in your local console connected to AWS.

```Python

for i in range(2):
print("weight = ", gmm_model.weights[i], "mu = ", gmm_model.gaussians[i].mu,
"sigma = ", gmm_model.gaussians[i].sigma.toArray())

```

The above code walked you through each part of the python file. Note that in AWS you will need to submit the code in aggregate. For your ease of use, the code above is collected below.

```Python
from __future__ import print_function
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel

datasets_path = "s3a://bigdataweberproject"
data_file = os.path.join(datasets_path,'gmm_data.txt')
gmmData = sc.textFile(data_file)

formattedData = gmmData.map(lambda line: [float(x) for x in line.strip().split(' ')])

gmm_model = GaussianMixture.train(formattedData, 2)

for i in range(2):
print("weight = ", gmm_model.weights[i], "mu = ", gmm_model.gaussians[i].mu,
"sigma = ", gmm_model.gaussians[i].sigma.toArray())

```

### 3.2 Old Faithful Eruptions Using Gaussian Mixture Modeling

> The following dataset comes from a set of internal data available in R. _faithful.csv_ contains observations of length of eruptions measured in minutes and time between eruptions. A simple plot of the data suggests there may be two underlying Gaussian distributions in this case. 

We again import the necessary tools to carry out the Mixture Modeling. 

```Python

from __future__ import print_function
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel

```

Create a configuration and context for Spark. Recall that these steps are necessary if you are working with AWS but if you are working with pyspark on a virtual machine the SparkContext may have been created for you already.

```Python

conf = SparkConf().setAppName("Gaussian Mixture Modeling Faithful Data")
sc = SparkContext(conf=conf)

```

Point to the S3 bucket created in earlier steps as well as the demonstration that goes with the beginning of these tutorials. We create a faithfulData RDD to prepare for analysis.

```Python

datasets_path = "s3a://bigdataweberproject"
data_file = os.path.join(datasets_path,'faithful.csv')
faithfulData = sc.textFile(data_file)

```

>Note that the data here is comma separated which requires splitting on a different character than in the previous example. However, unlike our earlier example, we only want to retain the "waiting" observations not the eruption time in minutes. Here, we split by a different character, retain only one value from that split and ensure that value is in float format.

```Python

formattedFaithfulData = faithfulData.map(lambda line: [float(x) for x in line.strip().split(',')[0]])

```

The time between eruptions variable appears to have a multivariate distribution with two normal distributions. In other words, the _k_ parameter, which indicates the number of clusters, is two.

```Python

gmm_model_faithful = GaussianMixture.train(formattedFaithfulData, 2)

```

Lastly, we can return the model parameters as we did in the original example.

```Python

for i in range(2):
print("weight = ", gmm_model_faithful.weights[i], "mu = ", gmm_model_faithful.gaussians[i].mu,
"sigma = ", gmm_model_faithful.gaussians[i].sigma.toArray())

```

Putting all of this together, we have the following code:

```Python

from __future__ import print_function
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel

conf = SparkConf().setAppName("Gaussian Mixture Modeling Faithful Data")
sc = SparkContext(conf=conf)

datasets_path = "s3a://bigdataweberproject"
data_file = os.path.join(datasets_path,'faithful.csv')
faithfulData = sc.textFile(data_file)

formattedFaithfulData = faithfulData.map(lambda line: [float(x) for x in line.strip().split(',')[0]])

gmm_model_faithful = GaussianMixture.train(formattedFaithfulData, 2)

for i in range(2):
print("weight = ", gmm_model_faithful.weights[i], "mu = ", gmm_model_faithful.gaussians[i].mu,
"sigma = ", gmm_model_faithful.gaussians[i].sigma.toArray())

```

### 3.3 Rainfall Data Using Gaussian Mixture Modeling

> The following dataset called "snoq.csv" contains rainfall data for Snoqualmie Peak in Washington over a period of years up to 1983 and can be found [here](http://www.stat.cmu.edu/~cshalizi/402/lectures/16-glm-practicals/snoqualmie.csv). Note that one issue with the data is that it contains missing values for leap years. The dataset provided to you here has missing values removed but consider the problem of what it would require for Spark to parse missing values correctly. 
Produce the two and three cluster solutions for this data. 

Standard setup.

```Python

from __future__ import print_function
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel

conf = SparkConf().setAppName("Gaussian Mixture Modeling Faithful Data")
sc = SparkContext(conf=conf)

```

Read in data.

```Python

datasets_path = "s3a://bigdataweberproject"
data_file = os.path.join(datasets_path,'snoq.csv')
rainfallData = sc.textFile(data_file)

```

Ensure data is in float format. It is important to note we need not split lines here as they are already provided in a single column format. 

```Python

formattedRainfallData = faithfulData.map(lambda x: float(x))

```

Produce both the two and three cluster solution. 

```Python

gmm_model_rainfall_2 = GaussianMixture.train(formattedRainfallData, 2)
gmm_model_rainfall_3 = GaussianMixture.train(formattedRainfallData, 3)

```

Print out the model parameters for both solutions and compare. 

```Python

for i in range(2):
    print("weight = ", gmm_model_rainfall_2.weights[i], "mu = ", gmm_model_rainfall_2.gaussians[i].mu,
    "sigma = ", gmm_model_rainfall_2.gaussians[i].sigma.toArray())

for i in range(2):
    print("weight = ", gmm_model_rainfall_3.weights[i], "mu = ", gmm_model_rainfall_3.gaussians[i].mu,
    "sigma = ", gmm_model_rainfall_3.gaussians[i].sigma.toArray())

```

References

1. [Joseph Konstan's answer to "What is collaborative filtering in layman's terms?"](http://qr.ae/THH9O0)
2. [Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques for recommender systems." Computer 42, no. 8 (2009): 30-37.](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

