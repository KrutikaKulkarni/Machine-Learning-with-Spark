
# coding: utf-8

# In[1]:

import warnings 
warnings.filterwarnings('ignore')


# In[2]:

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

import pandas as pd
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


winequality = sqlContext.read.load('winequality/winequality-white.csv',
                          format='com.databricks.spark.csv',
                          header='true',
                          delimiter=';',
                          inferschema='true')
   
#winequality.take(5)
winequality.cache()
winequality.printSchema()


# In[5]:

import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')
import pandas as pd
pd.options.display.mp1_style = 'default'


# In[3]:

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel


# In[4]:

from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint


# In[5]:


pd.DataFrame(winequality.take(5),columns=winequality.columns)


# In[9]:

winequality.describe().toPandas().transpose()


# In[10]:

NumericalFeatures = [t[0] for t in winequality.dtypes if t[1] == 'double']
import seaborn as sns
Sample = winequality.select(NumericFeatures).sample(False, 0.10).toPandas()
corr = Sample.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)


# In[11]:

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import UserDefinedFunction
temp_data = winequality
final_df_data= winequality
final_df_data.toPandas()[:5]


# In[12]:

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
stages = [] 
label_stringIdx = StringIndexer(inputCol = "quality", outputCol = "label")
stages += [label_stringIdx]
numericCols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
assemblerInputs = numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
cols = winequality.columns
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(winequality)
dataset = pipelineModel.transform(winequality)
selectedcols = ["label", "features"] 
dataset = dataset.select(selectedcols)
dataset.toPandas()



# In[13]:

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
print trainingData.count()
print testData.count()


# In[15]:

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(maxIter=3, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(trainingData)
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))




# In[16]:

from pyspark.ml.classification import DecisionTreeClassifier

# Create initial Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=3)

# Train model with Training Data
dtModel = dt.fit(trainingData)
print "numNodes = ", dtModel.numNodes
print "depth = ", dtModel.depth


# In[17]:

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree

def labelData(data):
    # label: row[end], features: row[0:end-1]
    return data.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))


training_data, testing_data = labelData(final_df_data).randomSplit([0.8, 0.2])

model = DecisionTree.trainClassifier(training_data, numClasses=3, maxDepth=3,
                                     categoricalFeaturesInfo={},
                                     impurity='gini', maxBins=32)

print model.toDebugString()


# In[45]:

print 'Feature 10:', winequality.columns[10]
print 'Feature 1:', winequality.columns[1]
print 'Feature 5:', winequality.columns[5]


# In[46]:

from pyspark.mllib.evaluation import MulticlassMetrics

def getPredictionsLabels(model, testData):
    predictions = model.predict(testData.map(lambda r: r.features))
    return predictions.zip(testData.map(lambda r: r.label))

def printMetrics(predictions_and_labels):
    metrics = MulticlassMetrics(predictions_and_labels)
    print 'Precision of True ', metrics.precision()
    print 'Precision of False', metrics.precision()
    print 'Recall of True    ', metrics.recall()
    print 'Recall of False   ', metrics.recall()
    print 'F-1 Score         ', metrics.fMeasure()
    print 'Confusion Matrix\n', metrics.confusionMatrix().toArray()

predictions_and_labels = getPredictionsLabels(model, testData)

printMetrics(predictions_and_labels)






