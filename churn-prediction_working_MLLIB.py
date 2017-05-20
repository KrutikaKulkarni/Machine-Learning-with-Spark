
# coding: utf-8

# # Churn Prediction with PySpark using MLlib and ML Packages
# 
# Churn prediction is big business. It minimizes customer defection by predicting which customers are likely to cancel a subscription to a service. Though originally used within the telecommunications industry, it has become common practice across banks, ISPs, insurance firms, and other verticals.
# 
# The prediction process is heavily data driven and often utilizes advanced machine learning techniques. In this post, we'll take a look at what types of customer data are typically used, do some preliminary analysis of the data, and generate churn prediction models - all with PySpark and its machine learning frameworks. We'll also discuss the differences between two Apache Spark version 1.6.0 frameworks, MLlib and ML.

# ## Install and Run Jupyter on Spark
# 
# To run this notebook tutorial, we'll need to install [Spark](http://spark.apache.org/) and [Jupyter/IPython](http://jupyter.org/), along with Python's [Pandas](http://pandas.pydata.org/) and [Matplotlib](http://matplotlib.org/) libraries.
# 
# For the sake of simplicity, let's run PySpark in local mode, using a single machine:
# >PYSPARK_DRIVER_PYTHON=ipython PYSPARK_DRIVER_PYTHON_OPTS=notebook /path/to/bin/pyspark --packages com.databricks:spark-csv_2.10:1.3.0 --master local[*]

# In[1]:

# Disable warnings, set Matplotlib inline plotting and load Pandas package
import warnings
warnings.filterwarnings('ignore')

get_ipython().magic(u'matplotlib inline')
import pandas as pd
pd.options.display.mpl_style = 'default'


# ## Fetching and Importing Churn Data
# 
# For this example, we'll be using the Orange Telecoms Churn Dataset. It consists of cleaned customer activity data (features), along with a churn label specifying whether the customer canceled their subscription or not. The data can be fetched from BigML's S3 bucket, [churn-80](https://bml-data.s3.amazonaws.com/churn-bigml-80.csv) and [churn-20](https://bml-data.s3.amazonaws.com/churn-bigml-20.csv). The two sets are from the same batch, but have been split by an 80/20 ratio. We'll use the larger set for training and cross-validation purposes, and the smaller set for final testing and model performance evaluation. The two data sets have been included in this repository for convenience.
# 
# In order to read the CSV data and parse it into Spark [DataFrames](http://spark.apache.org/docs/latest/sql-programming-guide.html), we'll use the [CSV package](https://github.com/databricks/spark-csv). The library has already been loaded using the initial pyspark bin command call, so we're ready to go.
# 
# Let's load the two CSV data sets into DataFrames, keeping the header information and caching them into memory for quick, repeated access. We'll also print the schema of the sets.

# In[2]:

CV_data = sqlContext.read.load('/Users/kruti/churn-bigml-80.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')

final_test_data = sqlContext.read.load('/Users/kruti/churn-bigml-20.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')
CV_data.cache()
CV_data.printSchema()
#CV_data.dtypes


# By taking 5 rows of the CV_data variable and generating a Pandas DataFrame with them, we can get a display of what the rows look like. We're using Pandas instead of the Spark _DataFrame.show()_ function because it creates a prettier print.

# In[3]:

pd.DataFrame(CV_data.take(5), columns=CV_data.columns)


# ## Summary Statistics
# 
# Spark DataFrames include some [built-in functions](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame) for statistical processing. The _describe()_ function performs summary statistics calculations on all numeric columns, and returns them as a DataFrame. 

# In[4]:

CV_data.describe().toPandas().transpose()


# ## Correlations and Data Preparation
# 
# We can also perform our own statistical analyses, using the [MLlib statistics package](http://spark.apache.org/docs/latest/mllib-statistics.html) or other python packages. Here, we're use the Pandas library to examine correlations between the numeric columns by generating scatter plots of them.
# 
# For the Pandas workload, we don't want to pull the entire data set into the Spark driver, as that might exhaust the available RAM and throw an out-of-memory exception. Instead, we'll randomly sample a portion of the data (say 10%) to get a rough idea of how it looks. 

# In[5]:

numeric_features = [t[0] for t in CV_data.dtypes if t[1] == 'int' or t[1] == 'double']

import seaborn as sns
sampled_data = CV_data.select(numeric_features).sample(False, 0.10).toPandas()
corr = sampled_data.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# It's obvious that there are several highly correlated fields, ie _Total day minutes_ and _Total day charge_. Such correlated data won't be very beneficial for our model training runs, so we're going to remove them. We'll do so by dropping one column of each pair of correlated fields, along with the _State_ and _Area code_ columns.
# 
# While we're in the process of manipulating the data sets, let's transform the categorical data into numeric as required by the machine learning routines, using a simple user-defined function that maps Yes/True and No/False to 1 and 0, respectively.

# In[6]:

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import UserDefinedFunction

temp_data = CV_data
binary_map = {'Yes':1.0, 'No':0.0, True:1.0, False:0.0}
toNum = UserDefinedFunction(lambda k: binary_map[k], DoubleType())
final_CV_data = CV_data.drop('State').drop('Area code')     .drop('Total day charge').drop('Total eve charge')     .drop('Total night charge').drop('Total intl charge')    .withColumn('Churn', toNum(CV_data['Churn']))     .withColumn('International plan', toNum(CV_data['International plan']))     .withColumn('Voice mail plan', toNum(CV_data['Voice mail plan'])).cache() 

final_test_data = final_test_data.drop('State').drop('Area code')     .drop('Total day charge').drop('Total eve charge')     .drop('Total night charge').drop('Total intl charge')     .withColumn('Churn', toNum(final_test_data['Churn']))     .withColumn('International plan', toNum(final_test_data['International plan']))     .withColumn('Voice mail plan', toNum(final_test_data['Voice mail plan'])).cache()


# Let's take a quick look at the resulting data set.

# In[7]:

final_CV_data.toPandas()[:5]


# ## Using the Spark MLlib Package
# 
# The [MLlib package](http://spark.apache.org/docs/latest/mllib-guide.html) provides a variety of machine learning algorithms for classification, regression, cluster and dimensionality reduction, as well as utilities for model evaluation. The decision tree is a popular classification algorithm, and we'll be using extensively here.  
# 
# ### Decision Tree Models
# 
# Decision trees have played a significant role in data mining and machine learning since the 1960's. They generate white-box classification and regression models which can be used for feature selection and sample prediction. The transparency of these models is a big advantage over black-box learners, because the models are easy to understand and interpret, and they can be readily extracted and implemented in any programming language (with nested if-else statements) for use in production environments. Furthermore, decision trees require almost no data preparation (ie normalization) and can handle both categorical and continuous data. To remedy over-fitting and improve prediction accuracy, decision trees can also be limited to a certain depth or complexity, or bundled into ensembles of trees (ie random forests).
# 
# A decision tree is a predictive model which maps observations (features) about an item to conclusions about the item's label or class. The model is generated using a top-down approach, where the source dataset is split into subsets using a statistical measure, often in the form of the Gini index or information gain via Shannon entropy. This process is applied recursively until a subset contains only samples with the same target class, or is halted by a predefined stopping criteria.
# 
# ### Model Training
# 
# MLlib classifiers and regressors require data sets in a format of rows of type _LabeledPoint_, which separates row labels and feature lists, and names them accordingly. The custom _labelData()_ function shown below performs the row parsing. We'll pass it the prepared data set (CV_data) and split it further into training and testing sets. A decision tree classifier model is then generated using the training data, using a maxDepth of 2, to build a "shallow" tree. The tree depth can be regarded as an indicator of model complexity. 

# In[8]:

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree

def labelData(data):
    # label: row[end], features: row[0:end-1]
    return data.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))


training_data, testing_data = labelData(final_CV_data).randomSplit([0.8, 0.2])

model = DecisionTree.trainClassifier(training_data, numClasses=2, maxDepth=2,
                                     categoricalFeaturesInfo={1:2, 2:2},
                                     impurity='gini', maxBins=32)

print model.toDebugString()


# The _toDebugString()_ function provides a print of the tree's decision nodes and final prediction outcomes at the end leafs. We can see that features 12 and 4 are used for decision making and should thus be considered as having high predictive power to determine a customer's likeliness to churn. It's not surprising that these feature numbers map to the fields _Customer service calls_ and _Total day minutes_. Decision trees are often used for feature selection because they provide an automated mechanism for determining the most important features (those closest to the tree root).

# In[9]:

print 'Feature 12:', CV_data.columns[12]
print 'Feature 4: ', CV_data.columns[4]


# ### Model Evaluation
# 
# Predictions of the testing data's churn outcome are made with the model's _predict()_ function and grouped together with the actual churn label of each customer data using _getPredictionsLabels()_.
# 
# We'll use MLlib's _MulticlassMetrics()_ for the model evaluation, which takes rows of (prediction, label) tuples as input. It provides metrics such as precision, recall, F1 score and confusion matrix, which have been bundled for printing with the custom _printMetrics()_ function.

# In[10]:

from pyspark.mllib.evaluation import MulticlassMetrics

def getPredictionsLabels(model, test_data):
    predictions = model.predict(test_data.map(lambda r: r.features))
    return predictions.zip(test_data.map(lambda r: r.label))

def printMetrics(predictions_and_labels):
    metrics = MulticlassMetrics(predictions_and_labels)
    print 'Precision of True ', metrics.precision(1)
    print 'Precision of False', metrics.precision(0)
    print 'Recall of True    ', metrics.recall(1)
    print 'Recall of False   ', metrics.recall(0)
    print 'F-1 Score         ', metrics.fMeasure()
    print 'Confusion Matrix\n', metrics.confusionMatrix().toArray()

predictions_and_labels = getPredictionsLabels(model, testing_data)

printMetrics(predictions_and_labels)


# The overall accuracy, ie F-1 score, seems quite good, but one troubling issue is the discrepancy between the recall measures. The recall (aka sensitivity) for the Churn=False samples is high, while the recall for the Churn=True examples is relatively low. Business decisions made using these predictions will be used to retain the customers most likely to leave, not those who are likely to stay. Thus, we need to ensure that our model is sensitive to the Churn=True samples.
# 
# Perhaps the model's sensitivity bias toward Churn=False samples is due to a skewed distribution of the two types of samples. Let's try grouping the CV_data DataFrame by the _Churn_ field and counting the number of instances in each group. 

# In[11]:

final_CV_data.groupby('Churn').count().toPandas()


# ### Stratified Sampling
# 
# There are roughly 6 times as many False churn samples as True churn samples. We can put the two sample types on the same footing using stratified sampling. The DataFrames _sampleBy()_ function does this when provided with fractions of each sample type to be returned.
# 
# Here we're keeping all instances of the Churn=True class, but downsampling the Churn=False class to a fraction of 388/2278.

# In[12]:

stratified_CV_data = final_CV_data.sampleBy('Churn', fractions={0: 388./2278, 1: 1.0}).cache()

stratified_CV_data.groupby('Churn').count().toPandas()


# Let's build a new model using the evenly distributed data set and see how it performs.

# In[13]:

training_data, testing_data = labelData(stratified_CV_data).randomSplit([0.8, 0.2])

model = DecisionTree.trainClassifier(training_data, numClasses=2, maxDepth=2,
                                     categoricalFeaturesInfo={1:2, 2:2},
                                     impurity='gini', maxBins=32)

predictions_and_labels = getPredictionsLabels(model, testing_data)
printMetrics(predictions_and_labels)


# With these new recall values, we can see that the stratified data was helpful in building a less biased model, which will ultimately provide more generalized and robust predictions.
