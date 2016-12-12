# Tutorial on Spark MLib

#### Members:
- Utkarsh Kajaria
- Eric Weber
- Eric Passe
- Yuemin Xu


##Intoduction

In this sequence of tutorials, we will talk about how Spark MLib can be used to implement multiple machine learning algorithms. We will also talk about ways of how these implementations can be used as interesting real-time applications by employing state-of-the-art technologies. 


## Tutorial 1:  Online Recommendation Engine

In this tutorial we will learn how to use the Spark machine learning library Mlib to build a movie recommendation engine. Not only this, we will take our engine live by hosting it as a web service on AWS. For this tutorial, we use the popular MovieLens data set. A description of this dataset can be found at the Grouplens [website](http://grouplens.org/datasets/movielens/latest/).

For the sake of simplicity, we have broken this into three parts: In Section 1.1 we will walk through the code for our recomendation engine. In Section 1.2 we will talk about exposing the engine utilies as web service. In Section 1.3, we will see how to access this service in real-time by hosting it on AWS.

### 1.1 The Engine

We will refer to the part that exclusively deals with training and providing recommendations by using Spark MLib as the engine. Before delving into how the engine is constructed, we need to be familiar with a few terminologies:
  
  **_Collaborative Filtering_** : A set of technologies that predict which items in a set of products (or information) a particular customer will like based on the preferences of lots of other people i.e. If I know your movie preferences, and those of lots of other people, I can make pretty good movie recommendations for you. [1]

  An obvious idea would be to look at a user's preferences and those in his *neighbourhood* of people and use the ratings given by these people to recommend new movies to our user. In fact, it was this idea on which most of the early recommendation algorithms were based upon. Over time people have developed more sophisticated ways of computing essentially the same thing. One of these is the Alternating Least Squares method.

  **_ALS_** : The idea behind ALS is that instead of building a list of neighbors for you, we use everyone’s ratings to form some type of intermediate representation of movie tastes — something like a taste space in which we can place you, and the movies — and then then use that representation to quickly evaluate movies to see if they match this community-based representation of your tastes. This taste space is called latent factors. [2]


The engine will have the following four major tasks:
  
  1. Training the ALS model
  2. Get the top n rated movies for a user
  3. Predict ratings for a given movie for a user
  4. Add the ratings for a new user

Given that the functionalities of the engine can be broken down into separate steps, it makes sense that we define the engine as a class and put these functionalities as separate methods of that class. 
In the following code we will define the Recommendation engine and initialize the common variables that will be used by all the methods. 

```Python
class RecommendationEngine:

  def __init__(self, sc, dataset_path):

        """Init the recommendation engine given a Spark context and a dataset path
        """

        logger.info("Starting up the Recommendation Engine: ")

        self.sc = sc

        # Load ratings data for later use
        logger.info("Loading Ratings data...")
        ratings_file_path = os.path.join(dataset_path, 'ratings.csv')
        ratings_raw_RDD = self.sc.textFile(ratings_file_path)
        ratings_raw_data_header = ratings_raw_RDD.take(1)[0]
        self.ratings_RDD = ratings_raw_RDD.filter(lambda line: line!=ratings_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
        # Load movies data for later use
        logger.info("Loading Movies data...")
        movies_file_path = os.path.join(dataset_path, 'movies.csv')
        movies_raw_RDD = self.sc.textFile(movies_file_path)
        movies_raw_data_header = movies_raw_RDD.take(1)[0]
        self.movies_RDD = movies_raw_RDD.filter(lambda line: line!=movies_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
        self.movies_titles_RDD = self.movies_RDD.map(lambda x: (int(x[0]),x[1])).cache()
        # Pre-calculate movies ratings counts
        self.__count_and_average_ratings()

```

The __init__ method above can be thought of as Python's counterpart to C++ or Java Style Constructors. Both C++ and Java by default pass the reference to an object as a hidden parameter and access it within the class using *this* keyword. However in python, we need to explicitly define it and refer to it using *self*. The subsequent lines of code  simply load the data from the ratings.csv and movies.csv files and store them into ratings_RDD and movies_RDD respectivey after performing the nescessary typecasting. We also pass sc so that methods of this class can perform Spark operations. The dataset_path is the path to our *S3 bucket* where we have stored all the data.

In addition, we need to initialize the paramters for the ALS algorithm and perform the model training. These lines are added into the same __init__ method. 

```Python
# Train the model
self.rank = 8
self.seed = 5
self.iterations = 10
self.regularization_parameter = 0.1
self.__train_model() 
 ``` 

Now we will look at each of the functions in detail.

####1.1.1 Training

This is where we see the magic of MLib at work. The library takes care of implementing the complex algorithm once we specify it's required parameters using it's easy to understand interface.

```Python
self.model = ALS.train(self.ratings_RDD, self.rank, seed=self.seed, \
                               iterations=self.iterations, lambda_=self.regularization_parameter)
logger.info("ALS model built!")
```

The ALS.train() takes the paramteres that we initialized in the __init__ method, trains the model and stores it in the class variable model. Now we can get some recommendations. To ensure better results, we will only recommend to a user, the movies which have more than a minimum number of ratings given by other people. For this, we need to count the number of ratings per movie. This can be done as follows:

```Python
def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)
movie_ID_with_ratings_RDD = self.ratings_RDD.map(lambda x: (x[1], x[2])).groupByKey()
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
self.movies_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))
```
where get_counts_and_averages is a helper function that takes a tuple (movieID, ratings_iterable) and returns (movieID, (ratings_count, ratings_avg))


####1.1.2 Getting Top Recommendations

To get recommendations, we first need all the movies the new user hasn't rated yet. We will then use our model on this data to predict ratings.

```Python
def get_top_ratings(self, user_id, movies_count):
  user_unrated_movies_RDD = self.ratings_RDD.filter(lambda rating: not rating[0] == user_id)\
                                                 .map(lambda x: (user_id, x[1])).distinct()
  # Get predicted ratings
  ratings = self.__predict_ratings(user_unrated_movies_RDD).filter(lambda r: r[2]>=25).takeOrdered(movies_count, key=lambda x: -x[1])
```


where the predict_ratings method takes a (userID, movieID) formatted RDD and returns an RDD with format (movieTitle, movieRating, numRatings) as shown below:

```Python
def __predict_ratings(self, user_and_movie_RDD):
        predicted_RDD = self.model.predictAll(user_and_movie_RDD)
        predicted_rating_RDD = predicted_RDD.map(lambda x: (x.product, x.rating))
        predicted_rating_title_and_count_RDD = \
            predicted_rating_RDD.join(self.movies_titles_RDD).join(self.movies_rating_counts_RDD)
        predicted_rating_title_and_count_RDD = \
            predicted_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
        
        return predicted_rating_title_and_count_RDD
```
The cascasded join operation above gives us an RDD of the form (movieID, ((rating, movie_title), num_ratings)) which is then flattened out to give (movieTitle, movieRating, numRatings) that we expect.

Finally we filter this list further to get only those movies which have a minimum of 25 ratings and then select the top *movies_count* number of recommendations out of them. 

The final output will be of the form :

```
    (u'Band of Brothers (2001)', 8.225114960311624, 4450)
    (u'Generation Kill (2008)', 8.206487040524653, 52)
    (u"Schindler's List (1993)", 8.172761674773625, 53609)
```

####1.1.3 Ratings For Particular Movies For a User

This is in fact a simpler case than getting top recommendations for a user. Since we already have the __predict_ratings method in place that takes an RDD containing userid and unrated movie to give recommendation, we simply need to call it as shown below:

```Python
def get_ratings_for_movie_ids(self, user_id, movie_ids):
    #Create an RDD of the form (user_id, movie_id) to pass to the __predict_rating() method
        requested_movies_RDD = self.sc.parallelize(movie_ids).map(lambda x: (user_id, x))
        # Get predicted ratings
        ratings = self.__predict_ratings(requested_movies_RDD).collect()

        return ratings
```

This will give us an output like the following:

```
[Rating(user=0, product=122880, rating=4.955831875971526)]
```

####1.1.4 Add a new user to the Dataset

To add a new user, we first need a set of movies in which the user has given his ratings. We add this information to the existing dataset and train the model again so that we can get new recommendations for this user. The user_rating.file contains the ratings information for a new user that we will be adding to the existing set. This file is of the form movie_id,rating:

```
1,8
16,7
25,8
...
```

This can be then passed to the follwing function that constructs and RDD and trains the model with it.

```Python
def add_ratings(self, ratings):
        # Convert ratings to an RDD
        new_ratings_RDD = self.sc.parallelize(ratings)
        # Add new ratings to the existing ones
        self.ratings_RDD = self.ratings_RDD.union(new_ratings_RDD)
        # Re-compute movie ratings count
        self.__count_and_average_ratings()
        # Re-train the ALS model with the new ratings
        self.__train_model()
        
        return ratings
```

That's all! We now have a full-fledged recommendation engine which we can use as a service. We will see what this means in the following section.

### 1.2 The Engine as a service


The recommender in itself is just a set of methods that provide us with the nescessary functionality. However, to be able to make use of it as a web-service, we need a way to translate a web request into one or more of the functions exposed by the recommender. This is where [Flask](http://flask.pocoo.org) comes in. 
Flask is a web application framework in python which consists of libraries using which we map the web-requests to appropriate methods of our recommendation engine.

This mapping is done in the python file named app.py. Let's look at one of them:

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

    
##Tutorial 2: Classification Using Naive Bayes
This pair of tutorials will go through two separate classification systems and evaluate the model for each, comparing the results on one set of data. Classifiers take predictor variables and outcome variables specified by the user, and return a system for predicting future outcomes with those predictor variables.


Naive Bayes is the first model used in this tutorial. It is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. It is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable.

Please note that one aspect of the tutorial setup is devoted to setting up your local machine to run the program, and one aspect is set up for running on Amazon Web Services. They require different libraries, and Spark Context does not need to be set if using in pyspark, only on AWS.

> AWS SETUP:
- This section is used if you are setting up the file to run on Amazon Web Services
- If you are running the local machine, skip to Step 1b
- Make sure you replace the datasets path with the  S3 path where the file is located

This block of code imports the necessary libraries, locates the file, and creates an RDD from the data.

```Python

# Begin Step 1a
# Use Step 1a if you are setting up a .py file to run on Amazon Web Services (AWS)...
# Import Relevant Libraries
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from numpy import array

#Spark context needs to be created.
conf = SparkConf().setAppName("NaiveBayes")
sc = SparkContext(conf=conf)

# Create dataset path & output path.
# These will be S3 bucket locations
datasets_path = "s3a://msbatutorial/data"
output_path = "s3a://msbatutorial/output"

# Boston file
filepath = os.path.join(datasets_path, 'boston50.txt')

# Make RDD
bostonRDD = sc.textFile(file)
bostonRDD.cache()
# End Step 1a

```

> LOCAL MACHINE SETUP:
- This section is used if you are setting up the file to run on Amazon Web Services
- If you set up your files to run on AWS, skip this section
- Make sure you replace the datasets path with the local path where the file is located

This block of code imports the necessary libraries, locates the file, and creates an RDD from the data.

```Python

# Begin step 1b
# Use Step 1b if you are setting up your local machine to run the model
# Import Relevant Libraries
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from numpy import array

#Spark context only needs to be called
sc

# Write the absolute path for the file you wish to use
filepath = "file:/home/training/training_materials/data/boston50.txt"
bostonRDD = sc.textFile(filepath)
bostonRDD.cache
# End step 1b



All code below will be used on both AWS and your local machine.

The boston file is tab delimited. For this case, replace tab delimited with comma delimited.

```Python

new_b = bostonRDD.map(lambda line: line.replace("\t", ","))

```

Define a function to take in the data. This identifies the predictor and outcome variables in the data. If using "boston.txt" as the data, the 13th column is designated as the outcome variable.
- The function parsePoint takes in an array and creates float values for each comma separated value. Then calls the LabeledPoint function to define one point as the outcome variable.
- parsePoint() is then called using the map function, and is applied to each row in the selected RDD

```Python

def parsePoint(line):
values = [float(x) for x in line.split(",")]
return LabeledPoint(values[13], values[0:13])

parsedData = new_b.map(parsePoint)

```

Split the data into a training set and a test set. The results of the model on the training set will be tested on the test set. Then call the NaiveBayes.train() function to train the data.
- NaiveBayes.train() allows for an optional smoothing parameter.
- NaiveBayes.train() also allows for model type parameter. The default type is 'multinomial'.

```Python

# Split data approximately into training and test
training_boston, test_boston = parsedData.randomSplit([0.6, 0.4])

# Train a naive bayes model. It takes an RDD of LabeledPoint and an optional smoothing parameter lambda as input, an optional model type parameter (default is “multinomial”)
model_boston = NaiveBayes.train(training_boston, 1.0)

```

Make the prediction and test the model. Then the number of correct values can be compared with the total number of values. The output will then be what percentage of outcomes in the testing data have been predicted accurrately by the model.

```Python

# Make prediction and text accuracy
# Create a new RDD where each line contains a tuple of predicted label and the original label
predictionAndLabel_boston = test_boston.map(lambda p: (model_boston.predict(p.features), p.label))

# Make prediction and text accuracy. Count how many predicted labels are equal to the original labels divied by the number of records, then we get the accuracy rate for our prediction
accuracy_boston = 1.0 * predictionAndLabel_boston.filter(lambda(x,v): x == v).count() /test_boston.count()

```

Print the model accurracy. Because of Spark's output, it is recommended to print something else that will help locate the result. The code below contains two strings of periods.

```Python

# Print Model Accuracy
print('..........................................................................................................')
print('model accuracy: {}'.format(accuracy_boston))
print('-----Finished NaiveBayes Model-----')
print('..........................................................................................................')

```

This code can all be put together and saved as a .py file as well, and either run on a local machine or on a service such as Amazon Web Services. Below is the full code for naiveBayesModel.py

```Python

# Import Relevant Libraries
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from numpy import array
conf = SparkConf().setAppName("NaiveBayes")
sc = SparkContext(conf=conf)

# Create dataset path & output path.
# These will be S3 bucket locations
datasets_path = "s3a://msbatutorial/data"
output_path = "s3a://msbatutorial/output"

# Boston file
file = os.path.join(datasets_path, 'boston50.txt')

# Make RDD
bostonRDD = sc.textFile(file)
bostonRDD.cache()

# Project boston data
# Replace '\t' with ','
new_b = bostonRDD.map(lambda line: line.replace("\t", ","))

# Split the data by
def parsePoint(line):
    values = [float(x) for x in line.split(",")]
    return LabeledPoint(values[13], values[0:13])

parsedData = new_b.map(parsePoint)

# Split data approximately into training and test
training_boston, test_boston = parsedData.randomSplit([0.6, 0.4])

# Train a naive bayes model
model_boston = NaiveBayes.train(training_boston, 1.0)

# Make prediction and text accuracy
predictionAndLabel_boston = test_boston.map(lambda p: (model_boston.predict(p.features), p.label))

accuracy_boston = 1.0 * predictionAndLabel_boston.filter(lambda(x,v): x == v).count() /test_boston.count()

# Print Model Accuracy
print('..........................................................................................................')
print('model accuracy: {}'.format(accuracy_boston))
print('-----Finished NaiveBayes Model-----')
print('..........................................................................................................')

```

### Tutorial 3: Classification Using Decision

Code Description

The second classificatier used in this tutorial is a Decision Tree model. The decision tree is a greedy algorithm that performs a recursive binary partitioning of the feature space. The tree predicts the same label for each bottommost (leaf) partition. Each partition is chosen greedily by selecting the best split from a set of possible splits, in order to maximize the information gain at a tree node. In other words, the split chosen at each tree node is chosen from the set argmax s IG(D,s) where IG(D,s) is the information gain when a split ss is applied to a dataset D.

> AWS SETUP:
- This section is used if you are setting up the file to run on Amazon Web Services
- If you are running the local machine, skip to Step 1b
- Make sure you replace the datasets path with the  S3 path where the file is located

This block of code imports the necessary libraries, locates the file, and creates an RDD from the data.

```Python

# Begin Step 1a
# Use Step 1a if you are setting up a .py file to run on Amazon Web Services (AWS)...
# Import Relevant Libraries
import os
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from numpy import array

#Spark context needs to be created.
conf = SparkConf().setAppName("NaiveBayes")
sc = SparkContext(conf=conf)

# Create dataset path & output path.
# These will be S3 bucket locations
datasets_path = "s3a://msbatutorial/data"
output_path = "s3a://msbatutorial/output"

# Boston file
filepath = os.path.join(datasets_path, 'boston50.txt')

# Make RDD
bostonRDD = sc.textFile(file)
bostonRDD.cache()
# End Step 1a


```

> LOCAL MACHINE SETUP:
- This section is used if you are setting up the file to run on Amazon Web Services
- If you set up your files to run on AWS, skip this section
- Make sure you replace the datasets path with the local path where the file is located

This block of code imports the necessary libraries, locates the file, and creates an RDD from the data.

```Python

# Begin Step 1b
# Use Step 1b if you are setting up your local machine to run the model
# Import Relevant Libraries
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from numpy import array

#Spark context only needs to be called
sc

# Write the absolute path for the file you wish to use
filepath = "file:/home/training/training_materials/data/boston50.txt"
bostonRDD = sc.textFile(filepath)
bostonRDD.cache
# End Step 1b

```

All code below will be used on both AWS and your local machine.

The boston file is tab delimited. For this case, replace tab delimited with comma delimited.

```Python

new_b = bostonRDD.map(lambda line: line.replace("\t", ","))

```

Define a function to take in the data. This identifies the predictor and outcome variables in the data. If using "boston.txt" as the data, the 13th column is designated as the outcome variable.

```Python

def parsePoint(line):
values = [float(x) for x in line.split(',')]
return LabeledPoint(values[13], values[0:13])

parsedData = new_b.map(parsePoint)

```

Separate the data into two sets. One will be used for the training of the model, the second will be used for the testing of the model accurracy.

```Python

# Split data approximately into training and test
training_boston, test_boston = parsedData.randomSplit([0.6, 0.4])

```

Call the DecisionTree.trainClassifier function.

```Python

# Use the decision tree algorithm to fit a model
model = DecisionTree.trainClassifier(training_boston, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)

```

Execute the prediction and return the classifications.

```Python

# Use the model we got from last step, we do the perdiction on each records and get their classifications.
prediction = model.predict(training_boston.map(lambda x: x.features))

```

Combine the training data and the classifications for each row

```Python

# Combine the original training data classification labels and the we predicted. 
labelsAndPredictions = training_boston.map(lambda lp: lp.label).zip(prediction)

```

Compare the labels on the training data. The function .filter() will select the records that did not match, finding the error rate.

```Python

# Comparing these two labels and geting the error rate.
train_Err = labelsAndPredictions.filter(lambda (v, p): v != p).count()/float(training_boston.count())

```

Print the training error to see how the model performs on the training data.

```Python

print('..........................................................................................................')
# Print Training Error
print('Training error = ' + str(train_Err))
print('..........................................................................................................')

```

Now test the model on the test data rather than the training data.

```Python

# Tests the predictions from the model
prediction_test = model.predict(test_boston.map(lambda x: x.features))

# Tests labels & predictions
labelsAndPredictions_test = test_boston.map(lambda lp: lp.label).zip(prediction_test)

```

Compare the number of correct results with the total number of values. The output will then be what percentage of outcomes in the testing data have been predicted accurrately by the model.

```Python

# Calculating the accuracy of the model on the observed data
accuracy_boston = labelsAndPredictions_test.filter(lambda (v, p): v == p).count()/float(test_boston.count())

```

Print the model accurracy. Because of Spark's output, it is recommended to print something else that will help locate the result. The code below contains two strings of periods.

```Python

print('..........................................................................................................')
print('model accuracy: {}'.format(accuracy_boston))
print('-----Finished Classifitation Tree Model-----')
print('..........................................................................................................')

```

This code can all be put together and saved as a .py file as well, and either run on a local machine or on a service such as Amazon Web Services. Below is the full code for treeModel.py

```Python

# Begin Step 1a
# Use Step 1a if you are setting up a .py file to run on Amazon Web Services (AWS)...
# Import Relevant Libraries
import os
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from numpy import array

#Spark context needs to be created.
conf = SparkConf().setAppName("NaiveBayes")
sc = SparkContext(conf=conf)

# Create dataset path & output path.
# These will be S3 bucket locations
datasets_path = "s3a://msbatutorial/data"
output_path = "s3a://msbatutorial/output"

# Boston file
filepath = os.path.join(datasets_path, 'boston50.txt')

# Make RDD
bostonRDD = sc.textFile(file)
bostonRDD.cache()
# End Step 1a

# Begin Step 1b
# Use Step 1b if you are setting up your local machine to run the model
# Import Relevant Libraries
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from numpy import array

#Spark context only needs to be called
sc

# Write the absolute path for the file you wish to use
filepath = "file:/home/training/training_materials/data/boston50.txt"
bostonRDD = sc.textFile(filepath)
bostonRDD.cache
# End Step 1b

# Replace '\t' with ',' to create a comma separated dataset
new_b = bostonRDD.map(lambda line: line.replace("\t", ","))

def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[13], values[0:13])

parsedData = new_b.map(parsePoint)

# Split data approximately into training and test
training_boston, test_boston = parsedData.randomSplit([0.6, 0.4])

# Use the decision tree algorithm to fit a model
model = DecisionTree.trainClassifier(training_boston, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)

# Use the model we got from last step, we do the perdiction on each records and get their classifications.
prediction = model.predict(training_boston.map(lambda x: x.features))

# Combine the original training data classification labels and the we predicted. 
labelsAndPredictions = training_boston.map(lambda lp: lp.label).zip(prediction)

# Comparing these two labels and geting the error rate.
train_Err = labelsAndPredictions.filter(lambda (v, p): v != p).count()/float(training_boston.count())

print('..........................................................................................................')
# Print Training Error
print('Training error = ' + str(train_Err))
print('..........................................................................................................')

# Tests the predictions from the model
prediction_test = model.predict(test_boston.map(lambda x: x.features))

# Tests labels & predictions
labelsAndPredictions_test = test_boston.map(lambda lp: lp.label).zip(prediction_test)

# Calculating the accuracy of the model on the observed data
accuracy_boston = labelsAndPredictions_test.filter(lambda (v, p): v == p).count()/float(test_boston.count())

print('..........................................................................................................')
print('model accuracy: {}'.format(accuracy_boston))
print('-----Finished Classifitation Tree Model-----')
print('..........................................................................................................')

```



## Fourth Tutorial: Guassian Mixture Modeling

> The purpose of this tutorial is to use Spark to identify "hidden" distributions that make up the overall distribution of data present. Gaussian Mixture Modeling is a process quite similar to k-means clustering in that it produces random parameter estimates of a given number of underlying normal distributions that comprise the distribution of the data observed.

> Gaussian Mixture Modeling relies on a method called Expectation Maximization. In effect, this algorithm assumes each data point is a linear combination of multivariate Guassian distributions. Like with k-means, it is an iterative algorithm that stops once it reaches a "best" solution, though that solution may not be globally the "best". The basic idea is we have M clusters and points in that cluster follow a normal distribution. 

> If you are familiar with k-means, this algorithm is not a huge step. Instead of using hard assignments to clusters, we now use soft assignment, where each point has a probability of belonging to a particular cluster. As with most algorithms, there are some key assumptions that must be met. First, the dataset should be Gaussian. What happens when this assumption is not met? If the number of clusters is known in advance, the algorithm is not guaranteed to find the "right" clusters. If the number of clusters is unknown, this actually is a bit better! If those clusters have a normal distribution then an information criterion is typically good enough to find the best solution. Second, the cluster sizes must be relatively close to each other. If the clusters are not the same size, a larger cluster will show up as more important than any smaller cluster.


## 4.1 Introduction to GMM Using Spark

> SparkMLLib contains a number of powerful methods for implementing algorithms that use Expectation Maximization, both for k-means and Gaussian Mixture Modeling. The following example builds on an example dataset provided by Spark and saved for you as "gmm_data.txt". 

It is important to know there are key parameters in control of the user. The SparkMLLib documentation identifies them as follows:

- **k** is the number of desired clusters.
- **convergenceTol** is the maximum change in log-likelihood at which we consider convergence achieved.
- **maxIterations** is the maximum number of iterations to perform without reaching convergence.
- **initialModel** is an optional starting point from which to start the EM algorithm. If this parameter is omitted, a random starting point will be constructed from the data.

> The following code assumes you are working on AWS. Please refer to the directions for AWS provided in this document for submission directions to AWS. Please keep in mind you will need to have the _gmm_data.txt_ file uploaded to your S3 bucket for this work.

We first import the necessary functions. These include a print function for displaying the Spark results in a reasonable format on AWS, an operating system import function, a Spark Context and the Guassian Mixture Model functions appropriate for pyspark. Please note that if you would like to complete these in Scala, the documentation is available on the SparkMLLib online reference guide.

```Python

from __future__ import print_function
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel

```

In the next step, we create a configuration for Spark which allows us to create an app name here. For more information on SparkConf you can reference https://spark.apache.org/docs/1.6.2/api/java/org/apache/spark/SparkConf.html. We also create a Spark context based on this Spark configuration which allows us to then use SparkMLLib which comes as part of the base build for Spark.


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

### 4.2 Old Faithful Eruptions Using Gaussian Mixture Modeling

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

> Note that the data here is comma separated which requires splitting on a different character than in the previous example. However, unlike our earlier example, we only want to retain the "waiting" observations not the eruption time in minutes. Here, we split by a different character, retain only one value from that split and ensure that value is in float format.

```Python

formattedFaithfulData = faithfulData.map(lambda line: [float(x) for x in line.strip().split(',')])

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

formattedFaithfulData = faithfulData.map(lambda line: [float(x) for x in line.strip().split(',')])

gmm_model_faithful = GaussianMixture.train(formattedFaithfulData, 2)

for i in range(2):
    print("weight = ", gmm_model_faithful.weights[i], "mu = ", gmm_model_faithful.gaussians[i].mu,
    "sigma = ", gmm_model_faithful.gaussians[i].sigma.toArray())

```

### 4.3 Rainfall Data Using Gaussian Mixture Modeling

> The following dataset called "snoq.csv" contains rainfall data for Snoqualmie Peak in Washington over a period of years up to 1983 and can be found at the [CMU website](http://www.stat.cmu.edu/~cshalizi/402/lectures/16-glm-practicals/snoqualmie.csv). Note that one issue with the data is that it contains missing values for leap years. The dataset provided to you here has missing values removed but consider the problem of what it would require for Spark to parse missing values correctly.

> Produce the two and three cluster solutions for this data. 

Standard setup.

```Python

from __future__ import print_function
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel

conf = SparkConf().setAppName("Gaussian Mixture Modeling Rainfall Data")
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

formattedRainfallData = rainfallData.map(lambda line: [float(x) for x in line.strip().split(',')])

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

Putting all of this code together we have:

```Python

from __future__ import print_function
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel

conf = SparkConf().setAppName("Gaussian Mixture Modeling Faithful Data")
sc = SparkContext(conf=conf)

datasets_path = "s3a://bigdataweberproject"
data_file = os.path.join(datasets_path,'snoq.csv')
rainfallData = sc.textFile(data_file)

formattedRainfallData = rainfallData.map(lambda line: [float(x) for x in line.strip().split(',')])

gmm_model_rainfall_2 = GaussianMixture.train(formattedRainfallData, 2)
gmm_model_rainfall_3 = GaussianMixture.train(formattedRainfallData, 3)

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
3. [Spark and Recommender Systems](https://www.codementor.io/spark/tutorial/building-a-recommender-with-apache-spark-python-example-app-part1)