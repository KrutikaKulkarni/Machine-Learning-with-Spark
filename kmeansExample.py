
# coding: utf-8

# In[1]:

from numpy import array
from math import sqrt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.regression import LabeledPoint


# In[5]:

# Load and parse the data
def parsePoint(line):
    aList = line.split(",")
    features = [float(x) for x in aList[:-1]]
    
    return features

data = sc.textFile("/Users/snerur/sparkExamples/iris.csv")
filteredData = data.filter(lambda x: "sepallength" not in x)
parsedData = filteredData.map(parsePoint)
parsedData.take(3)


# In[12]:

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 3, maxIterations=10, initializationMode="random")
# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))


# In[14]:

c = clusters.centers[clusters.predict([5.1,3.5,1.4,0.2])]
c
clusters.computeCost(parsedData)
clusters.clusterCenters
clusters.predict([5.1,3.5,1.4,0.2])

