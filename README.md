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

![Alt text](/images/EMR.png?raw=true "Optional title")