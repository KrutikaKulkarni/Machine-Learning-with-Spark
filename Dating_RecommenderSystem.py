
# coding: utf-8

# In[2]:

import random
def parse_rating(line, sep=','):
    fields = line.strip().split(sep)
    user_id = int(fields[0])
    profile_id = int(fields[1])
    rating = float(fields[2])
    return random.randint(1,10), (user_id, profile_id, rating)


# In[4]:

ratings = sc.textFile("ratings.dat").map(parse_rating)
ratings.take(2)


# In[5]:

def parse_user(line, sep=','):
    fields = line.strip().split(sep)
    user_id = int(fields[0])
    gender = fields[1]
    return user_id, gender


# In[15]:

users = dict(sc.textFile("gender.dat").map(parse_user).collect())


# In[19]:

# Create the training (60%) and validation (40%) set, based on last digit
    # of timestamp
num_partitions = 4
training = ratings.filter(lambda x: x[0] < 7).values().repartition(num_partitions).cache()

validation = ratings.filter(lambda x: x[0] >= 7).values().repartition(num_partitions).cache()

num_training = training.count()
num_validation = validation.count()

print "Training: %d and validation: %d\n" % (num_training, num_validation)


# In[22]:

from pyspark.mllib.recommendation import ALS

# rank is the number of latent factors in the model.
    # iterations is the number of iterations to run.
    # lambda specifies the regularization parameter in ALS
rank = 8
num_iterations = 8
lmbda = 0.1

# Train model with training data and configured rank and iterations
model = ALS.train(training, rank, num_iterations, lmbda)


# In[24]:

from math import sqrt
from operator import add
def compute_rmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error), or square root of the average value
        of (actual rating - predicted rating)^2
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictions_ratings = predictions.map(lambda x: ((x[0], x[1]), x[2]))       .join(data.map(lambda x: ((x[0], x[1]), x[2])))       .values()
    return sqrt(predictions_ratings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))


# In[26]:

print "The model was trained with rank = %d, lambda = %.1f, and %d iterations.\n" %         (rank, lmbda, num_iterations)
# Print RMSE of model
validation_rmse = compute_rmse(model, validation, num_validation)
print "Its RMSE on the validation set is %f.\n" % validation_rmse


# In[34]:

gender_filter = 'F'
matchseeker = 1
 # Filter on preferred gender
partners = sc.parallelize([u[0] for u in filter(lambda u: u[1] == gender_filter, users.items())])



# In[35]:

# run predictions with trained model
predictions = model.predictAll(partners.map(lambda x: (matchseeker, x))).collect()
# sort the recommedations
recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:10]

print "Eligible partners recommended for User ID: %d" % matchseeker
for i in xrange(len(recommendations)):
    print ("%2d: %s" % (i + 1, recommendations[i][1])).encode('ascii', 'ignore')

