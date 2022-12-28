# RealEstatePricePredictor
This is a ML model which predicts the price of real estate by understanding the features provided to it.

## Dragon Real State PRICE PREDICTOR

Dragon real state price predictor is a Machine learning model trained to predict prices of real estate properties.

Our model is capable of taking features and predict the prices of houses.
- We give dataset of house prices with some features like no. of bathrooms, no. of bedrooms etc.
- Our Model predicts the prices of any house by looking at it's features.

1. What type of Machine Learning model does it follows? Whether it is Supervised, Unsupervised or Reinforcement Learning?
- Since we have the labelled dataset with ourselves so this is a Supervised Machine Learning Model.

2. Whether it is a Classification task or a Regression Task?
- It is a Regression task because we are here to predict and find the value rather than classifying them into categories.

3. Whether it is Batch Learning or Online Learning?
- It is a Batch learning because we already have the dataset with us. In online learning it is a run-time learning process. Here we use batch learning for our model.

4. What are your Selecting performance measure?
- Here we follow RMSE(root mean square error) method which is a typical performance measure for Regression problems.
Other performance measure include Mean Absolute error, Manhattan norm etc. but we will use RMSE for this model.

### Coming on to the project

OPEN POWERSHELL WINDOW in any folder:
pip install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
if it doesnt have the permission then we can use 
pip install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn --user to give administrator permission.

then python -m notebook

We have taken our data from https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
- we get housing data and housing names
- now we have to convert our downloaded files to csv type.
we select our data and paste it in excel file but we see that all our data is pasted in one column so we select our column and goto data menu and select text to column and perform accordingly.

Since we know that we should create our test data which should be different from our feeding data so that we can ensure that our model is not just remembering things.

TRAIN_TEST splitting:

To overcome the above problem we create a Train-test splitting function which creates a random test case dataset for us.
test_ratio is just a part of actual dataset because we want to train our model on 80 percent of dataset and take its test on 20 percent remaining dataset.

import numpy as np
def split_train_test(data, test_ratio):
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

but this function has a problem that it will generate random permutation of dataset everytime but in long run our machine will get to know all the dataset and it will remember that so to overcome this we do: np.random.seed(42) to fix our dataset. We can use any number inplace of 42.

But easy way: We can implement the above split function from sklearn and this is the efficient way of splitting our train and test dataset.

IMPORTANT POINT: we should evenly distribute data with sklearn. it may happen that while feeding we haven't tested our model for a particular dataset and in testcase we are passing a different data which is completely new for the model so accuracy error may arise.

EVENLY DISTRIBUTION OF DATA:

So to overcome the above problem we use stratified sampling(StratifiedShuffleSplit): For stratified sampling we choose the most important feature and on that basis we distribute our data because we can't ignore the most important feature of any data.
Here, we do StratifiedShuffleSplit on basis of 'CHAS'
and our data gets evenly distributed.

LOOKING FOR CORRELATION:

Correlation means proportionality. We check if the price of the house increases then what is it's correlation with the given features.
We find correlation of our dataset with 'MEDV' which is the price of the house and find which features are directly and indirectly related to the price of the house.
we see the scattergraph inplace of straight line. But is kind of relate to straigt line.

We should plot different scatter plots and remove the other scatter points which are far from the main line or regression line so that our machine understands better.
This is a way of preparing data to feed to our machine learning model.

TRYING OUT ATTRIBUTES COMBINATION: We try out different attributes combination to get a better correlation of the final attribute to the price. Say, tax per room is a kind of attribute combination. It can be a good optimisation for our ML model.

MISSING ATTRIBUTES:
To take care of missing values we have three options:
1. Get rid of missing data points
2. Get rid of whole attribute
3. Set the value to some value(0, mean or median)
Note this we are doing only for the train_data but we can also have missing data in test_data and also if we are given any new dataset. so we compute this median and keep it safe for future.

We can easily do this with imputer class in scikitlearn. This fills up the missing boxes.

Scikit-Learn Design:
Primarily three types of objects:
1. Estimators: It estimates some parameters based on a dataset. Eg: imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters.

2. Transformers: transform method takes input and returns output based on the learning from fit(). It also has a convenience function called fit_transform() which fits and then transforms.

3. Predictors: LinearRegression model is an example of predictor. fit(), predict() are two common functions. It also gives score() function which will evaluate the predictions.

Feature Scaling: Feature scaling is a process of converting all the data features on same scale. Means all should be in range of 0 to 50 and so on. The units may vary but this is only feature scaling.
Primarily two types of feature scaling methods:
1. Min-Max scaling(Normalization): (value-min)/(max-min) and lies between 0 to 1.
Sklearn provides a class called MinMaxScaler for this.
2. Standarization:(value-mean)/std i.e. standard deviation.
Sklearn provides a class called StandardScaler for this.

Among them standarization is good because if we change the any value by mistake then this will not create much impact on the result.

Creating pipeline:

Pipeline means basic structure of machine learning model. It is basically a blueprint and it is useful because with less changes we can also change our machine learning model. 

Selecting a desired model for Dragon Real Estate:
Evaluating the model:

Testing the model:

Voila!!! you are done...
