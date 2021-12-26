# funtion to customizer correlation

my_corr<-function (input_data) 
{
  
  library(ggcorrplot)
  ggcorrplot(cor(input_data), 
             hc.order = TRUE, 
             type = "lower",
             lab = TRUE,
             outline.color = "white") +
    theme(
      # Hide panel borders and remove grid lines
      panel.border = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      # Change axis line
      axis.line = element_line(colour = "black")
    )
}


#  function to find optimal k 

set.seed(142)

optimal_number_of_clusters <- function (wcss)
{
  x1=2
  y1 = wcss[1]
  x2 = 20
  y2 =wcss[length(wcss)]
  
  
  distances = list()
  for (i in 1: length(wcss))
  {
    x0 = i+1
    y0 = wcss[i]
    numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denominator = sqrt((y2 - y1)^2 + (x2 - x1)^2)
    distances=append(distances, numerator/denominator)
  }
  
  return (which.max(distances) + 1)
}



# function to compute wcss

calculate_wcss<-function(data)
{
  
  elb_wss=rep(0, times=20)
  for (k in 1:20){
    output=kmeans(data, centers=k)
    elb_wss[k]=output$tot.withinss
  } 
  return(elb_wss)
  
}  