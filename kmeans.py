import math
import numpy as np

def loadData(filename: str) -> np.array:
  """
  Brief : 
    Opens file "filename" and load dataset from file and output as an array
  
  Parameters :
    filename : Name of the file.
  
  Returns : 
    An array containing dataset in the file.
  """
  # List of outputs
  output = []

  # Open file as readonly
  dataFile = open(filename, "r")

  # Reads in all lines in the file
  lines = dataFile.readlines()

  id = 0
  # Read in each line and store it as a tuple of data into output along with an id
  for line in lines:
    output.append([])
    words = line.split('\t')
    for word in words:
      output[id].append(float(word))
    output[id].append(0)
    id = id + 1
  
  # Close the file
  dataFile.close()
  return np.asarray(output)

def errCompute(X: np.array, M: np.array):
  """
  Brief : 
    Evaluates quality of clustering

  Parameters : 
    X : Dataset.

    M : Mean value for clustering.
  
  Returns : 
    The value of objective function for clustering.
  """
  error = 0.0
  
  # Compute error based on distance with respective cluster 
  # and average the total sum of errors by the number of points
  for x in X:
    print(np.linalg.norm(x[:-1] - M[int(x[-1])]))
    error = error + np.linalg.norm(x[:-1] - M[int(x[-1])])
  
  return (1.0 / X.shape[0]) * error

def calcMean(X: np.array, M: np.array):
  """
  Brief : 
    Compute the new centroid based on current cluster
  
  Parameters :
    X : datasets.

    M : List of centroids.

  Returns :
    M with updated centroids.
  """
  newM = np.zeros(M.shape)
  counter = [0 for i in range(M.shape[0])]

  # separate all data into respective set
  for x in X:
    cid = int(x[-1])
    newM[cid] += x[:-1]
    counter[cid] += 1
  
  # Compute new centroid distance 
  for i in range(M.shape[0]):
    newM[i] = newM[i] / counter[i]
  
  return newM

def Group(X: np.array, M: np.array):
  """
  Brief :
    Groups each data in dataset with a cluster id based on euclidean distance.
  
  Parameters : 
    X : datasets.

    M : Set of centroids for each cluster.
  
  Returns : 
    X with updated cluster ids.
  """
  # for each data in dataset, calculate distance to each cluster,
  # group them in the shortest distance cluster
  for i in range(X.shape[0]):
    p = np.asarray([n for n in X[i][:-1]])
    shortest = float("inf")
    cid = 0
    for c in M:
      dist = np.linalg.norm(p - c)
      if dist < shortest:
        shortest = dist
        X[i][-1] = cid
      cid = cid + 1
  
  return X