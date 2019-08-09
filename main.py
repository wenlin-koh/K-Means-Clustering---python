import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

from kmeans import *

colorCodes = ("#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
              "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
              "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
              "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
              "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
              "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
              "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
              "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",

              "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
              "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
              "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
              "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
              "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
              "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
              "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
              "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58" )

def GetPlotPoints(X: np.array):
  """
  Brief : 
    Retrieves a list of x values and a list of y values from the dataset.

  Parameters : 
    X : Array of dataset to retrieve plot points
  """
  x = [n[0] for n in X]
  y = [n[1] for n in X]

  return x, y

def PlotScatter(x: list, y: list, title: str = "", color = "red", figure_size: tuple=(8, 4.5), scatter_size: float=5, edge_color = "red", marker: str="o"):
  """
  Brief : 
    Plot a scatter graph.
  """
  plt.figure(figsize=figure_size)
  plt.scatter(x, y, s=scatter_size, c=color, edgecolors=edge_color, marker=marker)
  plt.title(title)
  plt.xlabel("X position (converted from latitude)")
  plt.ylabel("Y position (converted from longitude)")

def Cluster(X: np.array, k: int):
  """
  Brief : 
    Cluster into set of cluster based on groups

  Parameters : 
    X : Dataset.
  """
  cluster = [[] for i in range(k)]
  for i in range(X.shape[0]):
    cluster[int(X[i][-1])].append(X[i][:-1])
  
  return cluster

def main():
  """
  Brief : 
    Main entry point of the program.
  """
  # Load dataset from file
  X = loadData("cudaTest1.txt")
  
  # print(X.shape)
  
  # Plot scatter plot
  px, py = GetPlotPoints(X)
  
  # PlotScatter(px, py, marker=".", title="Lightning data")
  # plt.show()

  # Test errCompute()
  # J = errCompute(X, np.array([[0,0]]))
  
  # print(J)
  
  # Test Group()
  # M = np.copy(X[0:5, 0:X.shape[1]-1])
  # X = Group(X, M)
  # J = errCompute(X, M)
  
  # print(J)

  # Test calcMean()
  print("Clustering")
  J = 0.0
  M = np.copy(X[0:2, 0:X.shape[1]-1])
  while True:
    X = Group(X,M)
    M = calcMean(X,M)
    newJ = errCompute(X,M)

    print(newJ)
    if newJ == J:
      break
    
    J = newJ

  print(J)

  cluster = Cluster(X, M.shape[0])

  print("Plotting")
  plt.figure(figsize=(8, 4.5))
  for i in range(M.shape[0]):
    plt.scatter([x[0] for x in cluster[i]], [y[1] for y in cluster[i]], s=15, c=colorCodes[i], edgecolors=colorCodes[i], marker=".")
  
  plt.scatter([x[0] for x in M], [y[1] for y in M], s=25, c="red", edgecolors="black", marker="P")

  plt.title("Lightning data")
  plt.xlabel("X position (converted from latitude)")
  plt.ylabel("Y position (converted from longitude)")
  plt.show()


if __name__ == "__main__":
  main()