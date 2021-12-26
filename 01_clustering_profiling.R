


######### LIBRARIES

library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(dendextend) # for comparing two dendrograms--> Hierarchical Clustering Algorithms
library(Factoshiny) # automate analysis
library(FactoMineR) #unsupervised algorithm
library(plotly)     #dynamic plots


######### My functions

source("C:/Users/u32118508/OneDrive - UPEC/Bureau/scoring/00_functions_clustering.R")



##########  ENvironment
setwd("C:/Users/u32118508/OneDrive - UPEC/Bureau/scoring/seance1/INPUT")


data=read.table("student-por.csv",sep=";",header=TRUE)
print(nrow(data)) # 649 student

View(data)


# Remove missing values
data <- na.omit(data)
sapply(data, class)

# ====> Kmean is not applicable for categorial features
  
#====> 1st solution , run the algorithm with just continous fetaure

#====> 2st solution ,transform  categorial features to continuous

#
#########################################################################
######  Transform categorial features to continuous features ############
#########################################################################

#list of non numeric columns
colnames(data)[grepl('factor|logical|character',sapply(data,class))]


### MCA analysis
res.MCA<-MCA(data,quanti.sup=c(3,7,8,13,14,15,24,25,26,27,28,29,30,31,32,33),graph=FALSE,ncp=Inf)
cat=res.MCA$ind$coord
cat=as.data.frame(cat)
cat$Id  <- 1:nrow(cat)

numeric_feature<-c("age","absences", "Fedu" , "Medu", "Fedu", "freetime", "G1", 
                   "G2","G3" ,"goout" , "health" ,"studytime" ,"traveltime","Walc")
cont<-data[numeric_feature]

### scale features
#cont<-as.data.frame(scale(cont))

###correlation
my_corr(cont)

cont$Id  <- 1:nrow(cont)
input_data<-cat %>%
            inner_join(cont,by='Id') %>%
            dplyr::select(-Id)

#input_data<-as.data.frame(scale(input_data))

####################################################################
#####################   kmeans                    ##################
####################################################################


###########---------------- Find optimal k---------------#

set.seed(10)

# Elbow method
fviz_nbclust(input_data, kmeans, method = "wss") +
  labs(subtitle = "Elbow method")
  # calculating the within clusters sum-of-squares for 19 cluster amounts
    sum_of_squares = calculate_wcss(input_data)

  # calculating the optimal number of clusters
     n = optimal_number_of_clusters(sum_of_squares)

     message("!!!!!optimal number of cluster: ", n)

# Silhouette method
fviz_nbclust(input_data, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")
 



# Gap statistic
fviz_nbclust(input_data, kmeans,k.max = 20,  method = "gap_stat", nboot = 200)+
  labs(subtitle = "Gap statistic method")


#plot.new()          
#dev.new(width = 15,   height = 3,noRStudioGD = TRUE)
output1 <- kmeans(input_data, centers = 6, nstart = 25)
output1$size
output1
str(output1)  

####################################################################
#####################   kmeans++                    ################
####################################################################

library(LICORS)
output2=kmeanspp(input_data, k = 6 , start = "random", iter.max = 100, nstart = 25)
output2$size
output2

# Enhanced Kmean clustering
res.km <- eclust(x=input_data, 
                 FUN="kmeans",
                 hc_metric = "euclidean" ,
                 k=6)




#assess silhouette
fviz_silhouette(res.km) 

 #variance explained
message ("variance explained :", res.km$betweenss/res.km$totss)





#####################################################################
#####################   Hierarchical Clustering    ##################
#####################################################################
# Dissimilarity matrix
d = dist(input_data, method = "euclidean")

# Hierarchical clustering using Complete Linkage

library("factoextra")

# Ward's method
# Enhanced hierarchical clustering
res.hc <- eclust(x=input_data, 
                 FUN="hclust",
                 hc_metric = "euclidean" ,
                 hc_method = "ward.D2",
                 k=6)



# dendogram
fviz_dend(res.hc, rect = TRUE) 

#assess silhouette
fviz_silhouette(res.hc) 




K <- n
ntotal<-dim(input_data)[1]
T <- sum(res.hc$height)
W <- sum(res.hc$height[1:(ntotal-K)])
# E
message ("variance explained :", (1-W/T))

#+++++++++ Descriptive statistics & profiling






data$romantic <- ifelse(data$romantic == "yes",1,0)
data$internet <- ifelse(data$internet == "yes",1,0)
data$activities <- ifelse(data$activities == "yes",1,0)
data$sex<- ifelse(data$sex == "M",1,0)


binary_feature<-c("romantic", "internet", "sex","activities")
output<-data[c(binary_feature,numeric_feature)]
output<-data.frame(scale(output))
output$cluster=res.km$cluster
# create dataset with the cluster number
output$cluster <- (res.km$cluster)

# Reshape the data
library(GGally)
temp <- gather(output, key="features", values,-cluster)
head(temp)

#compute size
size=output%>%
  group_by(cluster) %>%
  summarise(size=n())


#statistic
grp <- temp %>%
        group_by(cluster, features) %>%
         summarise(within_mean=mean(values),
                   sd=sd(values)) %>%
         ungroup() %>%
         data.frame()



grp=size%>%
   group_by(cluster) %>%
   right_join(grp, by='cluster') %>%
   mutate(w_mean= within_mean *  size / dim(input_data)[1]) %>%
   ungroup() %>%
   group_by(features) %>%
   mutate(between_mean= sum(w_mean)) %>%
  data.frame()

View(grp)

#####################################################################
#####################   Profiling    ##################
#####################################################################

grp$cluster_id=factor(grp$cluster)
ggplotly(ggplot(data=grp,aes(y=features, x=within_mean, fill=cluster_id))+
       facet_wrap(~cluster)+
       geom_bar(stat="identity") +
       geom_point(data=grp,aes(x=between_mean, y =features))
       
)


library("writexl")

output$romantic <- ifelse(output$romantic>0 ,1,0)
output$internet <- ifelse(output$internet >0,1,0)
output$activities <- ifelse(output$activities > 0,1,0)
output$sex<- ifelse(output$sex> 0,1,0)
write_xlsx(output,"../OUTPUT/output_tp1.xlsx")
