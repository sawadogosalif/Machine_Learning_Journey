def calculate_wcss(data, feature_col, range_max):
  """This function calculates the within cluster sum of squares over a series of cluster numbers.
 
  Arguments:
      data {pyspark.sql.DataFrame or pandas dataframe} -- Data prepped for clustering - with an id column and vectorized features column to profile on
      feature_col {str} -- Name of vactorized features columns 
      range_max {int} -- Maximum number of clusters to calculate wcss for 
     
  Returns:
      list -- Contains the WCSS calculated @ each cluster number between 2 and the maximum provided.
  """    
 
  wcss = []
  
  for n in range(2, range_max+1):
      kmeans = KMeans().setK(n).setSeed(1).setFeaturesCol(feature_col)  # Udapte this line if you use sklearn API
      model = kmeans.fit(data) 
      ksummary = model.summary
      wcss.append(ksummary.trainingCost)
 
  return wcss
def optimal_number_of_clusters(wcss):
  """This function calculates the optimal number of clusters based on a list of previously calculated wcss - for each number of clusters; 
  establishes the point in an elbow curve, which is most distant from a line drawn between points corresponding to [wcss @ min k] and [wcss @ max k]
 
  Arguments:
      wcss {list} -- Contains wcss calculated for a desired number of clusters in a prior step
     
  Returns:
      int -- The optimal number of clusters.
  """  
  
  x1, y1 = 2, wcss[0]
  x2, y2 = 20, wcss[len(wcss)-1]
 
  distances = []
  for i in range(len(wcss)):
      x0 = i+2
      y0 = wcss[i]
      numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
      denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
      distances.append(numerator/denominator)
 
  return distances.index(max(distances)) + 2
