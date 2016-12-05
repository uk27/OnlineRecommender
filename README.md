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

Description comes here.


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

Hosting the service on AWS might appear to be a complicated process if you are doing it for the first time. However, if you closely follow the steps below, you can be up and running in a matter of minutes! Let's see how this done.

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

## 3  Third Tutorial
> Problem Statement

Some Description

### 3.1 Subsection Topic

Description comes here.


###### Figure 5: Code for XX
```Python

#Python Code comes here
def doSomething():
  #line 1
  #line 2
  return 

```

Code Description


### 3.2 Subsection Topic

Description comes here.


###### Figure 6: Code for XX
```Python

#Python Code comes here
def doSomething():
  #line 1
  #line 2
  return 

```

Code Description

