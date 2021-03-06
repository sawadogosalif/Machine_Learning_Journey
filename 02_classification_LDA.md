Linear Discriminant Analysis
================

-   [My functions](#my-functions)
-   [Libraries](#libraries)
-   [Data](#data)
-   [MODELS](#models)
    -   [First model](#first-model)
    -   [Nice to Know !!](#nice-to-know-)
    -   [Forwardselection](#forwardselection)
-   [Interpretable LDA](#interpretable-lda)
    -   [From LDA with PCA functions to easy to use
        scores](#from-lda-with-pca-functions-to-easy-to-use-scores)
    -   [Compute Wilks Lambda from scratch and derive its
        Pvalue](#compute-wilks-lambda-from-scratch-and-derive-its-pvalue)
    -   [Features importances by Wilk
        Lambda](#features-importances-by-wilk-lambda)

The purpose of this notebook is to measure the quality of the work
carried out in the clustering part. The more stable our clustering is,
the more we will be able to find good ranking. Moreover, an LDA will
also allow focusing on the variables to be explained.

# My functions

``` r
source("C:/Users/u32118508/OneDrive - UPEC/Bureau/Machine_learning_journey/A_journey_in_Machine_Learning/00_functions_multiclass.R")
```

    ## Le chargement a nécessité le package : ggplot2

    ## Le chargement a nécessité le package : lattice

# Libraries

``` r
library(tidyverse) #for easy data manipulation and visualization
library(caret)  #for easy machine learning workflow
library(MASS)   #for Modern Applied Statistic
library(readxl) # Load the data
library(klaR)   # lda features selection
```

# Data

``` r
data=read_excel("C:/Users/u32118508/OneDrive - UPEC/Bureau/Machine_learning_journey/A_journey_in_Machine_Learning/OUTPUT/output_tp1.xlsx")
head(data)
```

    ## # A tibble: 6 x 20
    ##   romantic internet   sex activities  paid schoolsup   age absences  Medu  Fedu
    ##      <dbl>    <dbl> <dbl>      <dbl> <dbl>     <dbl> <dbl>    <dbl> <dbl> <dbl>
    ## 1        0        0     0          0     0         1    18        4     4     4
    ## 2        0        1     0          0     0         0    17        2     1     1
    ## 3        0        1     0          0     0         1    15        6     1     1
    ## 4        1        1     0          1     0         0    15        0     4     2
    ## 5        0        0     0          0     0         0    16        0     3     3
    ## 6        0        1     1          1     0         0    16        6     4     3
    ## # ... with 10 more variables: freetime <dbl>, G1 <dbl>, G2 <dbl>, G3 <dbl>,
    ## #   goout <dbl>, health <dbl>, studytime <dbl>, traveltime <dbl>, Walc <dbl>,
    ## #   cluster <dbl>

LDA is compatible with categorial feeature. This is why we transform
features in binaries features. We don’t use hot Encoding in order to
avoid multicolinearity.

# MODELS

Split the data into training (80%) and test set (20%)

``` r
set.seed(123)
train.size=0.8
train.index<- sample.int(dim(data)[1],round(dim(data)[1] * train.size ))
train.sample=data[train.index, ]
test.sample=data[-train.index, ]
```

## First model

Fit the model

``` r
model <- lda(cluster~.,  data = train.sample)
str(model)
```

    ## List of 10
    ##  $ prior  : Named num [1:6] 0.0424 0.395 0.052 0.2197 0.1503 ...
    ##   ..- attr(*, "names")= chr [1:6] "1" "2" "3" "4" ...
    ##  $ counts : Named int [1:6] 22 205 27 114 78 73
    ##   ..- attr(*, "names")= chr [1:6] "1" "2" "3" "4" ...
    ##  $ means  : num [1:6, 1:19] 0.5 0.337 0.407 0.351 0.449 ...
    ##   ..- attr(*, "dimnames")=List of 2
    ##   .. ..$ : chr [1:6] "1" "2" "3" "4" ...
    ##   .. ..$ : chr [1:19] "romantic" "internet" "sex" "activities" ...
    ##  $ scaling: num [1:19, 1:5] 0.14084 -0.04457 0.00281 0.06444 0.03368 ...
    ##   ..- attr(*, "dimnames")=List of 2
    ##   .. ..$ : chr [1:19] "romantic" "internet" "sex" "activities" ...
    ##   .. ..$ : chr [1:5] "LD1" "LD2" "LD3" "LD4" ...
    ##  $ lev    : chr [1:6] "1" "2" "3" "4" ...
    ##  $ svd    : num [1:5] 22.934 16.821 4.58 2.634 0.984
    ##  $ N      : int 519
    ##  $ call   : language lda(formula = cluster ~ ., data = train.sample)
    ##  $ terms  :Classes 'terms', 'formula'  language cluster ~ romantic + internet + sex + activities + paid + schoolsup + age +      absences + Medu + Fedu + freetim| __truncated__ ...
    ##   .. ..- attr(*, "variables")= language list(cluster, romantic, internet, sex, activities, paid, schoolsup, age,      absences, Medu, Fedu, freetime, G1,| __truncated__ ...
    ##   .. ..- attr(*, "factors")= int [1:20, 1:19] 0 1 0 0 0 0 0 0 0 0 ...
    ##   .. .. ..- attr(*, "dimnames")=List of 2
    ##   .. .. .. ..$ : chr [1:20] "cluster" "romantic" "internet" "sex" ...
    ##   .. .. .. ..$ : chr [1:19] "romantic" "internet" "sex" "activities" ...
    ##   .. ..- attr(*, "term.labels")= chr [1:19] "romantic" "internet" "sex" "activities" ...
    ##   .. ..- attr(*, "order")= int [1:19] 1 1 1 1 1 1 1 1 1 1 ...
    ##   .. ..- attr(*, "intercept")= int 1
    ##   .. ..- attr(*, "response")= int 1
    ##   .. ..- attr(*, ".Environment")=<environment: R_GlobalEnv> 
    ##   .. ..- attr(*, "predvars")= language list(cluster, romantic, internet, sex, activities, paid, schoolsup, age,      absences, Medu, Fedu, freetime, G1,| __truncated__ ...
    ##   .. ..- attr(*, "dataClasses")= Named chr [1:20] "numeric" "numeric" "numeric" "numeric" ...
    ##   .. .. ..- attr(*, "names")= chr [1:20] "cluster" "romantic" "internet" "sex" ...
    ##  $ xlevels: Named list()
    ##  - attr(*, "class")= chr "lda"

Make predictions

``` r
train.pred<- model %>% predict(train.sample)
test.pred <- model %>% predict(test.sample)
```

How to evaluate the model? Let’s see the accuracy\| weigt accuracy and
micro\|macro F1 as we have multiclassification

For more detail
<https://www.cse.iitk.ac.in/users/purushot/papers/macrof1.pdf>

TRAIN SET

``` r
accuracy_rate(factor(train.pred$class),factor(train.sample$cluster))
```

    ## [1] "Accuracy is: 0.92"
    ## [1] "Weighted accuracy is: 0.86"

``` r
F1_rate(factor(train.pred$class),factor(train.sample$cluster))
```

    ## [1] "Macro F1 is: 0.9"
    ## [1] "Micro F1 is: 0.96"

TEST SET

``` r
accuracy_rate(factor(test.pred$class),factor(test.sample$cluster))
```

    ## [1] "Accuracy is: 0.88"
    ## [1] "Weighted accuracy is: 0.82"

``` r
F1_rate(factor(test.pred$class),factor(test.sample$cluster))
```

    ## [1] "Macro F1 is: 0.88"
    ## [1] "Micro F1 is: 0.93"

===> The model seems quite complex, the performance is low in the test
data Pour plus de détail sur les performance :

``` r
confusionMatrix(factor(train.pred$class),factor(train.sample$cluster))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   1   2   3   4   5   6
    ##          1  22   0   0   0   0   0
    ##          2   0 202  10   9   5   4
    ##          3   0   0  14   0   0   0
    ##          4   0   3   0 104   0   5
    ##          5   0   0   3   0  71   2
    ##          6   0   0   0   1   2  62
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9152          
    ##                  95% CI : (0.8879, 0.9377)
    ##     No Information Rate : 0.395           
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.8846          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity           1.00000   0.9854  0.51852   0.9123   0.9103   0.8493
    ## Specificity           1.00000   0.9108  1.00000   0.9802   0.9887   0.9933
    ## Pos Pred Value        1.00000   0.8783  1.00000   0.9286   0.9342   0.9538
    ## Neg Pred Value        1.00000   0.9896  0.97426   0.9754   0.9842   0.9758
    ## Prevalence            0.04239   0.3950  0.05202   0.2197   0.1503   0.1407
    ## Detection Rate        0.04239   0.3892  0.02697   0.2004   0.1368   0.1195
    ## Detection Prevalence  0.04239   0.4432  0.02697   0.2158   0.1464   0.1252
    ## Balanced Accuracy     1.00000   0.9481  0.75926   0.9463   0.9495   0.9213

``` r
confusionMatrix(factor(test.pred$class),factor(test.sample$cluster))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  1  2  3  4  5  6
    ##          1  8  0  0  0  0  0
    ##          2  0 49  6  2  2  5
    ##          3  0  0  7  0  0  0
    ##          4  0  0  0 19  0  1
    ##          5  0  0  0  0 24  0
    ##          6  0  0  0  0  0  7
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.8769         
    ##                  95% CI : (0.8078, 0.928)
    ##     No Information Rate : 0.3769         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.8333         
    ##                                          
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity           1.00000   1.0000  0.53846   0.9048   0.9231  0.53846
    ## Specificity           1.00000   0.8148  1.00000   0.9908   1.0000  1.00000
    ## Pos Pred Value        1.00000   0.7656  1.00000   0.9500   1.0000  1.00000
    ## Neg Pred Value        1.00000   1.0000  0.95122   0.9818   0.9811  0.95122
    ## Prevalence            0.06154   0.3769  0.10000   0.1615   0.2000  0.10000
    ## Detection Rate        0.06154   0.3769  0.05385   0.1462   0.1846  0.05385
    ## Detection Prevalence  0.06154   0.4923  0.05385   0.1538   0.1846  0.05385
    ## Balanced Accuracy     1.00000   0.9074  0.76923   0.9478   0.9615  0.76923

## Nice to Know !!

1.  Wilks lambda: test manova , diffcult to compute the pvalue direcly:
    A solution is ROA’s transformation The Wilks lambda is the preferred
    indicator for the statistical evaluation of the model. Itindicates
    to what extent the class centers are distinct from each other in the
    space of representation. It varies between 0 and 1: towards 0, the
    model will be good.

2.  We can use **Wilks lambda**’ to derive model selections because this
    statisitics show what will happen if the removed the features X

3.  R combines 2 approches of LDA:

The posterio is based on predictive approach

But the function score is compute on the geometric approach (PCA)

## Forwardselection

Features selection

``` r
model.forward <- greedy.wilks(cluster ~ .,data=train.sample, niveau = 0.1)
print(model.forward)
```

    ## Formula containing included variables: 
    ## 
    ## cluster ~ absences + G3 + G1 + G2 + age
    ## <environment: 0x000000002728d868>
    ## 
    ## 
    ## Values calculated in each step of the selection procedure: 
    ## 
    ##       vars Wilks.lambda F.statistics.overall p.value.overall F.statistics.diff
    ## 1 absences   0.18794867             443.2937   1.409900e-183        443.293742
    ## 2       G3   0.05155497             348.5877   2.292465e-321        270.909157
    ## 3       G1   0.04065768             206.0413    0.000000e+00         27.392213
    ## 4       G2   0.03908155             140.2928    0.000000e+00          4.113585
    ## 5      age   0.03778344             107.1326    0.000000e+00          3.497501
    ##    p.value.diff
    ## 1 1.409900e-183
    ## 2  0.000000e+00
    ## 3  0.000000e+00
    ## 4  1.134499e-03
    ## 5  4.058279e-03

Absences , G3 , G1 ,G2 and age are the most importants to split classes;

Let’s fit the model

``` r
model.best <- lda(model.forward$formula,  data = train.sample)
```

Make predictions

``` r
train.pred2<- predict(model.best, train.sample)
test.pred2<-  predict(model.best, test.sample)
```

How to evaluate the model?

``` r
#TRAIN SET
accuracy_rate(factor(train.pred2$class),factor(train.sample$cluster))
```

    ## [1] "Accuracy is: 0.93"
    ## [1] "Weighted accuracy is: 0.87"

``` r
F1_rate(factor(train.pred2$class),factor(train.sample$cluster))
```

    ## [1] "Macro F1 is: 0.92"
    ## [1] "Micro F1 is: 0.96"

``` r
#TEST SET
accuracy_rate(factor(test.pred2$class),factor(test.sample$cluster))
```

    ## [1] "Accuracy is: 0.89"
    ## [1] "Weighted accuracy is: 0.84"

``` r
F1_rate(factor(test.pred2$class),factor(test.sample$cluster))
```

    ## [1] "Macro F1 is: 0.89"
    ## [1] "Micro F1 is: 0.94"

====> this model is parsimonious ===== \>the metrics are efficients in
the two data sets compared to the first model

# Interpretable LDA

## From LDA with PCA functions to easy to use scores

Now the going to Construct our linear discriminant function Indeed, From
the raw values of the features, we construct a linear and easy to
implement score function in order to decide on the assignment of
classes.

R give us functions discriminant with PCA (these are not easy to use)
.For a new comer student who arrives, firsly it is necessary to have his
coordinates on PCA axes before being able to use the scores. This is
why, I am doing myself in this step the linear functions which do not
require the coordinates pca

Let see the functions with PCA

``` r
print(model.best$scaling)
```

    ##                  LD1         LD2         LD3         LD4          LD5
    ## absences -0.44445935 0.236909308 -0.01695951  0.03000607 -0.005182973
    ## G3        0.19606241 0.252341761  0.73855321 -0.02466343  0.443169886
    ## G1        0.12327031 0.184552184 -0.57088577  0.12582968  0.551953905
    ## G2        0.06277965 0.178893039 -0.31611667 -0.13280609 -0.974368175
    ## age      -0.01216889 0.004669037 -0.16720029 -0.81125632  0.197250120

R affiche bel et bien les coefficients des fonctions canoniques à partir
partir d’ une projection dans l’espace factoriel,

with R, we don’t have the intercept, we need to compute it ourselves

Overall means by features

``` r
xb <- colMeans(data.frame(train.sample[,-which(names(train.sample) == "cluster")])[,c('absences','G3','G2','G1','age')])
print(xb)
```

    ##  absences        G3        G2        G1       age 
    ##  3.552987 12.052023 11.687861 11.558767 16.709056

This is how to get intercept for lda functions

``` r
const_ <- apply(model.best$scaling,2,function(v){-sum(v*xb)})
print(const_)
```

    ##       LD1       LD2       LD3       LD4       LD5 
    ## -2.746882 -8.185783  4.279308 13.810355 -3.807208

Conditional means

``` r
cond_Xb <- sapply(1:ncol(model.best$scaling),function(j)
  {sapply(model.best$lev,
        function(niveau){sum(model.best$scaling[,j]*model.best$means[niveau,]) + const_[j]})
   }
)
colnames(cond_Xb) <- paste("LD",1:ncol(model.best$scaling),sep="")
rownames(cond_Xb) <- model.best$lev
print(cond_Xb)
```

    ##          LD1        LD2         LD3         LD4        LD5
    ## 1 -7.0396697  2.6258814 -0.34674338  0.25077114 -0.1781889
    ## 2  0.7545951 -0.9813370  0.31804967  0.09216973 -0.1954021
    ## 3 -1.3149288 -4.2622721 -1.28759834 -0.02176205 -0.2021992
    ## 4  2.5249423  1.3675328 -0.26857348 -0.16058496 -0.1919567
    ## 5 -2.6268373 -0.5689681  0.35952406 -0.36055693 -0.1982584
    ## 6 -0.7030025  2.0080500 -0.04332387  0.07229239 -0.2120466

From pca features functions to linear features functions

``` r
coef_ <- sapply(model.best$lev,function(niveau)
{rowSums(sapply(1:ncol(model.best$scaling),
                function(j){model.best$scaling[,j]*cond_Xb[niveau,j]}))
})
print(coef_)
```

    ##                    1           2          3          4          5          6
    ## absences  3.76527159 -0.56949029 -0.4031075 -0.7975217  1.0168399  0.7921848
    ## G3       -1.05883634  0.04634251 -2.3733891  0.5606673 -0.4720404  0.2411299
    ## G1       -0.25201687 -0.36591398 -0.3279747  0.5907988 -0.7888608  0.2007202
    ## G2        0.27773292 -0.05056911 -0.2381037  0.6964208 -0.1392870  0.5257979
    ## age      -0.08268656 -0.18025884  0.1891581  0.1129770  0.2225942 -0.0752997

From pca intercepts to linear features intercepts

``` r
intercept_ <- sapply(model.best$lev,function(niveau)
     {sum(mapply(prod,const_,cond_Xb[niveau,]))-0.5*sum(cond_Xb[niveau,]^2)+log(model.best$prior[niveau])})
names(intercept_) <- levels(factor(train.sample$cluster))
print(intercept_)
```

    ##          1          2          3          4          5          6 
    ## -30.994334   7.569065  19.707528 -26.472098   3.530489 -17.136820

Below are the functions

**Cluster 1**:  
*F*<sub>1</sub> =  − 30, 99 + 3, 76 \* *a**b**s**e**n**c**e**s* − 1, 06 \* *G*3 + 0, 27 \* *G*1 − 0, 25 \* *G*2 − 0, 08 \* *a**g**e*

**Cluster 2**:  
*F*<sub>2</sub> = 7, 56 − 0, 57 \* *a**b**s**e**n**c**e**s* + 0, 04 \* *G*3 − 0, 05 \* *G*2 − 0, 36 \* *G*1 − 0, 18 \* *a**g*

**Cluster 3**:  
*F*<sub>3</sub> = 19, 7 − 0, 4 \* *a**b**s**e**n**c**e**s* − 2, 37 \* *G*3 − 0, 24 \* *G*2 − 0, 33 \* *G*1 + 0, 19 \* *a**g**e*

**Cluster 4**:  
*F*<sub>4</sub> =  − 26, 47 − 0, 8 \* *a**b**s**e**n**c**e**s* + 0, 56 \* *G*3 + 0, 7 \* *G*2 + 0, 59 \* *G*1 + 0, 11 \* *a**g**e*

**Cluster 5**:  
*F*<sub>5</sub> = 3, 53 + 1, 01 \* *a**b**s**e**n**c**e**s* − 0, 47 \* *G*3 − 0, 14 \* *G*2 − 0, 79 \* *G*1 + 0, 22 \* *a**g**e*

**Cluster 6**:  
*F*<sub>6</sub> =  − 17, 13 + 0, 8 \* *a**b**s**e**n**c**e**s* + 0, 24 \* *G*3 + 0, 52 \* *G*2 + 0, 2 \* *G*1 − 0, 07 \* *a**g**e*

. Pour chaque indivi donnée, il suffit de calculer directemnt la
fonction score et prendre le maximum des 6 scores . On peut également se
transformer chaque fonction en probabilité:   score1=F1 / (F1+F2+F3+F4+f5+F6)

## Compute Wilks Lambda from scratch and derive its Pvalue

``` r
Xtrain=data.frame(train.sample[,-which(names(train.sample) == "cluster")])
#number of
p <- ncol(Xtrain)
#dataset length
n <- nrow(Xtrain)
#number of cluster
K <- nlevels(factor(train.sample$cluster))
#Degrees of freedom ;numerator
ddlSuppNum <- K - 1
#Degrees of freedom : denominator
ddlSuppDenom <- n - K - p + 1

#matrice de convariance totale
TOT <- cov(Xtrain)

#WITHIN - covariance intra-classe

WIT <- (1.0/(n-K)) *Reduce("+",
                         lapply(levels(factor(train.sample$cluster)),function(niveau)
                                        {(sum(factor(train.sample$cluster)==niveau)-1)*cov(Xtrain[factor(train.sample$cluster)==niveau,])}))

#lambda de Wilks
#matrices interm?diaires
WITprim <- (n-K)/n*WIT
TOTprim <- (n-1)/n*TOT
#rapport des d?terminants -- Lambda de Wilks
LW <- det(WITprim)/det(TOTprim)
print(LW)
```

    ## [1] 0.03346139

``` r
## lower value ==> good


#compute pvale
stat=-log(LW) * (n -0.5 *( p-K+1))
pvalue=pchisq(stat, df=p, lower.tail=FALSE)
print (pvalue)
```

    ## [1] 0

## Features importances by Wilk Lambda

``` r
#vecteur des F
FTest <- numeric(p)
#p-value
pvalueFTest <- numeric(p)
#boucle
for (j in 1:p){
  #Lambda corresp.
  LWvar <- det(WITprim[-j,-j])/det(TOTprim[-j,-j])
  #print(LWvar)
  #F
  FTest[j] <- ddlSuppDenom / ddlSuppNum * (LWvar/LW - 1)
  #p-value
  pvalueFTest[j] <- pf(FTest[j],ddlSuppNum,ddlSuppDenom,lower.tail=FALSE)
}


temp <- data.frame(var=colnames(Xtrain),FValue=FTest,pvalue=round(pvalueFTest,6))
print(temp)
```

    ##           var      FValue   pvalue
    ## 1    romantic   0.5030586 0.774006
    ## 2    internet   1.3797111 0.230363
    ## 3         sex   0.4473971 0.815227
    ## 4  activities   0.2596665 0.934874
    ## 5        paid   0.2716775 0.928571
    ## 6   schoolsup   1.1084641 0.354865
    ## 7         age   2.8640812 0.014626
    ## 8    absences 405.0588779 0.000000
    ## 9        Medu   2.1264501 0.061035
    ## 10       Fedu   1.4814370 0.194242
    ## 11   freetime   0.6779326 0.640348
    ## 12         G1  13.2207342 0.000000
    ## 13         G2   3.7129957 0.002615
    ## 14         G3  24.0874576 0.000000
    ## 15      goout   0.9010864 0.480069
    ## 16     health   1.4525203 0.203975
    ## 17  studytime   0.4632619 0.803634
    ## 18 traveltime   0.7713878 0.570679
    ## 19       Walc   1.4039420 0.221275

``` r
#affichage
temp <- data.frame(var=colnames(Xtrain),FValue=FTest,pvalue=round(pvalueFTest,6)) %>% 
  arrange(FValue)
print(temp)
```

    ##           var      FValue   pvalue
    ## 1  activities   0.2596665 0.934874
    ## 2        paid   0.2716775 0.928571
    ## 3         sex   0.4473971 0.815227
    ## 4   studytime   0.4632619 0.803634
    ## 5    romantic   0.5030586 0.774006
    ## 6    freetime   0.6779326 0.640348
    ## 7  traveltime   0.7713878 0.570679
    ## 8       goout   0.9010864 0.480069
    ## 9   schoolsup   1.1084641 0.354865
    ## 10   internet   1.3797111 0.230363
    ## 11       Walc   1.4039420 0.221275
    ## 12     health   1.4525203 0.203975
    ## 13       Fedu   1.4814370 0.194242
    ## 14       Medu   2.1264501 0.061035
    ## 15        age   2.8640812 0.014626
    ## 16         G2   3.7129957 0.002615
    ## 17         G1  13.2207342 0.000000
    ## 18         G3  24.0874576 0.000000
    ## 19   absences 405.0588779 0.000000
=======
Linear Discriminant Analysis
================

-   [My functions](#my-functions)
-   [Libraries](#libraries)
-   [Data](#data)
-   [MODELS](#models)
    -   [First model](#first-model)
    -   [Nice to Know !!](#nice-to-know-)
    -   [Forwardselection](#forwardselection)
-   [Interpretable LDA](#interpretable-lda)
    -   [From LDA with PCA functions to easy to use
        scores](#from-lda-with-pca-functions-to-easy-to-use-scores)
    -   [Compute Wilks Lambda from scratch and derive its
        Pvalue](#compute-wilks-lambda-from-scratch-and-derive-its-pvalue)
    -   [Features importances by Wilk
        Lambda](#features-importances-by-wilk-lambda)

The purpose of this notebook is to measure the quality of the work
carried out in the clustering part. The more stable our clustering is,
the more we will be able to find good ranking. Moreover, an LDA will
also allow focusing on the variables to be explained.

# My functions

``` r
source("C:/Users/u32118508/OneDrive - UPEC/Bureau/Machine_learning_journey/A_journey_in_Machine_Learning/00_functions_multiclass.R")
```

# Libraries

``` r
library(tidyverse) #for easy data manipulation and visualization
library(caret)  #for easy machine learning workflow
library(MASS)   #for Modern Applied Statistics
library(readxl) # Load the data
library(klaR)   # lda features selection
```

# Data

``` r
data=read_excel("C:/Users/u32118508/OneDrive - UPEC/Bureau/Machine_learning_journey/A_journey_in_Machine_Learning/OUTPUT/output_tp1.xlsx")
head(data)
```

    ## # A tibble: 6 x 20
    ##   romantic internet   sex activities  paid schoolsup   age absences  Medu  Fedu
    ##      <dbl>    <dbl> <dbl>      <dbl> <dbl>     <dbl> <dbl>    <dbl> <dbl> <dbl>
    ## 1        0        0     0          0     0         1    18        4     4     4
    ## 2        0        1     0          0     0         0    17        2     1     1
    ## 3        0        1     0          0     0         1    15        6     1     1
    ## 4        1        1     0          1     0         0    15        0     4     2
    ## 5        0        0     0          0     0         0    16        0     3     3
    ## 6        0        1     1          1     0         0    16        6     4     3
    ## # ... with 10 more variables: freetime <dbl>, G1 <dbl>, G2 <dbl>, G3 <dbl>,
    ## #   goout <dbl>, health <dbl>, studytime <dbl>, traveltime <dbl>, Walc <dbl>,
    ## #   cluster <dbl>

LDA is compatible with categorial feeature. This is why we transform
features in binaries features. We don’t use hot Encoding in order to
avoid multicolinearity.

# MODELS

Split the data into training (80%) and test set (20%)

``` r
set.seed(123)
train.size=0.8
train.index<- sample.int(dim(data)[1],round(dim(data)[1] * train.size ))
train.sample=data[train.index, ]
test.sample=data[-train.index, ]
```

## First model

Fit the model

``` r
model <- lda(cluster~.,  data = train.sample)
str(model)
```

    ## List of 10
    ##  $ prior  : Named num [1:6] 0.0424 0.395 0.052 0.2197 0.1503 ...
    ##   ..- attr(*, "names")= chr [1:6] "1" "2" "3" "4" ...
    ##  $ counts : Named int [1:6] 22 205 27 114 78 73
    ##   ..- attr(*, "names")= chr [1:6] "1" "2" "3" "4" ...
    ##  $ means  : num [1:6, 1:19] 0.5 0.337 0.407 0.351 0.449 ...
    ##   ..- attr(*, "dimnames")=List of 2
    ##   .. ..$ : chr [1:6] "1" "2" "3" "4" ...
    ##   .. ..$ : chr [1:19] "romantic" "internet" "sex" "activities" ...
    ##  $ scaling: num [1:19, 1:5] 0.14084 -0.04457 0.00281 0.06444 0.03368 ...
    ##   ..- attr(*, "dimnames")=List of 2
    ##   .. ..$ : chr [1:19] "romantic" "internet" "sex" "activities" ...
    ##   .. ..$ : chr [1:5] "LD1" "LD2" "LD3" "LD4" ...
    ##  $ lev    : chr [1:6] "1" "2" "3" "4" ...
    ##  $ svd    : num [1:5] 22.934 16.821 4.58 2.634 0.984
    ##  $ N      : int 519
    ##  $ call   : language lda(formula = cluster ~ ., data = train.sample)
    ##  $ terms  :Classes 'terms', 'formula'  language cluster ~ romantic + internet + sex + activities + paid + schoolsup + age +      absences + Medu + Fedu + freetim| __truncated__ ...
    ##   .. ..- attr(*, "variables")= language list(cluster, romantic, internet, sex, activities, paid, schoolsup, age,      absences, Medu, Fedu, freetime, G1,| __truncated__ ...
    ##   .. ..- attr(*, "factors")= int [1:20, 1:19] 0 1 0 0 0 0 0 0 0 0 ...
    ##   .. .. ..- attr(*, "dimnames")=List of 2
    ##   .. .. .. ..$ : chr [1:20] "cluster" "romantic" "internet" "sex" ...
    ##   .. .. .. ..$ : chr [1:19] "romantic" "internet" "sex" "activities" ...
    ##   .. ..- attr(*, "term.labels")= chr [1:19] "romantic" "internet" "sex" "activities" ...
    ##   .. ..- attr(*, "order")= int [1:19] 1 1 1 1 1 1 1 1 1 1 ...
    ##   .. ..- attr(*, "intercept")= int 1
    ##   .. ..- attr(*, "response")= int 1
    ##   .. ..- attr(*, ".Environment")=<environment: R_GlobalEnv> 
    ##   .. ..- attr(*, "predvars")= language list(cluster, romantic, internet, sex, activities, paid, schoolsup, age,      absences, Medu, Fedu, freetime, G1,| __truncated__ ...
    ##   .. ..- attr(*, "dataClasses")= Named chr [1:20] "numeric" "numeric" "numeric" "numeric" ...
    ##   .. .. ..- attr(*, "names")= chr [1:20] "cluster" "romantic" "internet" "sex" ...
    ##  $ xlevels: Named list()
    ##  - attr(*, "class")= chr "lda"

Make predictions

``` r
train.pred<- model %>% predict(train.sample)
test.pred <- model %>% predict(test.sample)
```

How to evaluate the model? Let’s see the accuracy\| weigt accuracy and
micro\|macro F1 as we have multiclassification

For more detail
<https://www.cse.iitk.ac.in/users/purushot/papers/macrof1.pdf>

TRAIN SET

``` r
accuracy_rate(factor(train.pred$class),factor(train.sample$cluster))
```

    ## [1] "Accuracy is: 0.92"
    ## [1] "Weighted accuracy is: 0.86"

``` r
F1_rate(factor(train.pred$class),factor(train.sample$cluster))
```

    ## [1] "Macro F1 is: 0.9"
    ## [1] "Micro F1 is: 0.96"

TEST SET

``` r
accuracy_rate(factor(test.pred$class),factor(test.sample$cluster))
```

    ## [1] "Accuracy is: 0.88"
    ## [1] "Weighted accuracy is: 0.82"

``` r
F1_rate(factor(test.pred$class),factor(test.sample$cluster))
```

    ## [1] "Macro F1 is: 0.88"
    ## [1] "Micro F1 is: 0.93"

===> The model seems quite complex, the performance is low in the test
data Pour plus de détail sur les performance :

``` r
confusionMatrix(factor(train.pred$class),factor(train.sample$cluster))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   1   2   3   4   5   6
    ##          1  22   0   0   0   0   0
    ##          2   0 202  10   9   5   4
    ##          3   0   0  14   0   0   0
    ##          4   0   3   0 104   0   5
    ##          5   0   0   3   0  71   2
    ##          6   0   0   0   1   2  62
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9152          
    ##                  95% CI : (0.8879, 0.9377)
    ##     No Information Rate : 0.395           
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.8846          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity           1.00000   0.9854  0.51852   0.9123   0.9103   0.8493
    ## Specificity           1.00000   0.9108  1.00000   0.9802   0.9887   0.9933
    ## Pos Pred Value        1.00000   0.8783  1.00000   0.9286   0.9342   0.9538
    ## Neg Pred Value        1.00000   0.9896  0.97426   0.9754   0.9842   0.9758
    ## Prevalence            0.04239   0.3950  0.05202   0.2197   0.1503   0.1407
    ## Detection Rate        0.04239   0.3892  0.02697   0.2004   0.1368   0.1195
    ## Detection Prevalence  0.04239   0.4432  0.02697   0.2158   0.1464   0.1252
    ## Balanced Accuracy     1.00000   0.9481  0.75926   0.9463   0.9495   0.9213

``` r
confusionMatrix(factor(test.pred$class),factor(test.sample$cluster))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  1  2  3  4  5  6
    ##          1  8  0  0  0  0  0
    ##          2  0 49  6  2  2  5
    ##          3  0  0  7  0  0  0
    ##          4  0  0  0 19  0  1
    ##          5  0  0  0  0 24  0
    ##          6  0  0  0  0  0  7
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.8769         
    ##                  95% CI : (0.8078, 0.928)
    ##     No Information Rate : 0.3769         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.8333         
    ##                                          
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity           1.00000   1.0000  0.53846   0.9048   0.9231  0.53846
    ## Specificity           1.00000   0.8148  1.00000   0.9908   1.0000  1.00000
    ## Pos Pred Value        1.00000   0.7656  1.00000   0.9500   1.0000  1.00000
    ## Neg Pred Value        1.00000   1.0000  0.95122   0.9818   0.9811  0.95122
    ## Prevalence            0.06154   0.3769  0.10000   0.1615   0.2000  0.10000
    ## Detection Rate        0.06154   0.3769  0.05385   0.1462   0.1846  0.05385
    ## Detection Prevalence  0.06154   0.4923  0.05385   0.1538   0.1846  0.05385
    ## Balanced Accuracy     1.00000   0.9074  0.76923   0.9478   0.9615  0.76923

## Nice to Know !!

1.  Wilks lambda: test manova , diffcult to compute the pvalue direcly:
    A solution is ROA’s transformation The Wilks lambda is the preferred
    indicator for the statistical evaluation of the model. Itindicates
    to what extent the class centers are distinct from each other in the
    space of representation. It varies between 0 and 1: towards 0, the
    model will be good.

2.  We can use **Wilks lambda**’ to derive model selections because this
    statisitics show what will happen if the removed the features X

3.  R combines 2 approches of LDA:

The posterio is based on predictive approach

But the function score is compute on the geometric approach (PCA)

## Forwardselection

Features selection

``` r
model.forward <- greedy.wilks(cluster ~ .,data=train.sample, niveau = 0.1)
print(model.forward)
```

    ## Formula containing included variables: 
    ## 
    ## cluster ~ absences + G3 + G1 + G2 + age
    ## <environment: 0x000000002728d868>
    ## 
    ## 
    ## Values calculated in each step of the selection procedure: 
    ## 
    ##       vars Wilks.lambda F.statistics.overall p.value.overall F.statistics.diff
    ## 1 absences   0.18794867             443.2937   1.409900e-183        443.293742
    ## 2       G3   0.05155497             348.5877   2.292465e-321        270.909157
    ## 3       G1   0.04065768             206.0413    0.000000e+00         27.392213
    ## 4       G2   0.03908155             140.2928    0.000000e+00          4.113585
    ## 5      age   0.03778344             107.1326    0.000000e+00          3.497501
    ##    p.value.diff
    ## 1 1.409900e-183
    ## 2  0.000000e+00
    ## 3  0.000000e+00
    ## 4  1.134499e-03
    ## 5  4.058279e-03

Absences , G3 , G1 ,G2 and age are the most importants to split classes;

Let’s fit the model

``` r
model.best <- lda(model.forward$formula,  data = train.sample)
```

Make predictions

``` r
train.pred2<- predict(model.best, train.sample)
test.pred2<-  predict(model.best, test.sample)
```

How to evaluate the model?

``` r
#TRAIN SET
accuracy_rate(factor(train.pred2$class),factor(train.sample$cluster))
```

    ## [1] "Accuracy is: 0.93"
    ## [1] "Weighted accuracy is: 0.87"

``` r
F1_rate(factor(train.pred2$class),factor(train.sample$cluster))
```

    ## [1] "Macro F1 is: 0.92"
    ## [1] "Micro F1 is: 0.96"

``` r
#TEST SET
accuracy_rate(factor(test.pred2$class),factor(test.sample$cluster))
```

    ## [1] "Accuracy is: 0.89"
    ## [1] "Weighted accuracy is: 0.84"

``` r
F1_rate(factor(test.pred2$class),factor(test.sample$cluster))
```

    ## [1] "Macro F1 is: 0.89"
    ## [1] "Micro F1 is: 0.94"

====> this model is parsimonious ===== \>the metrics are efficients in
the two data sets compared to the first model

# Interpretable LDA

## From LDA with PCA functions to easy to use scores

Now the going to Construct our linear discriminant function Indeed, From
the raw values of the features, we construct a linear and easy to
implement score function in order to decide on the assignment of
classes.

R give us functions discriminant with PCA (these are not easy to use)
.For a new comer student who arrives, firsly it is necessary to have his
coordinates on PCA axes before being able to use the scores. This is
why, I am doing myself in this step the linear functions which do not
require the coordinates pca

Let see the functions with PCA

``` r
print(model.best$scaling)
```

    ##                  LD1         LD2         LD3         LD4          LD5
    ## absences -0.44445935 0.236909308 -0.01695951  0.03000607 -0.005182973
    ## G3        0.19606241 0.252341761  0.73855321 -0.02466343  0.443169886
    ## G1        0.12327031 0.184552184 -0.57088577  0.12582968  0.551953905
    ## G2        0.06277965 0.178893039 -0.31611667 -0.13280609 -0.974368175
    ## age      -0.01216889 0.004669037 -0.16720029 -0.81125632  0.197250120

R affiche bel et bien les coefficients des fonctions canoniques à partir
partir d’ une projection dans l’espace factoriel,

with R, we don’t have the intercept, we need to compute it ourselves

Overall means by features

``` r
xb <- colMeans(data.frame(train.sample[,-which(names(train.sample) == "cluster")])[,c('absences','G3','G2','G1','age')])
print(xb)
```

    ##  absences        G3        G2        G1       age 
    ##  3.552987 12.052023 11.687861 11.558767 16.709056

This is how to get intercept for lda functions

``` r
const_ <- apply(model.best$scaling,2,function(v){-sum(v*xb)})
print(const_)
```

    ##       LD1       LD2       LD3       LD4       LD5 
    ## -2.746882 -8.185783  4.279308 13.810355 -3.807208

Conditional means

``` r
cond_Xb <- sapply(1:ncol(model.best$scaling),function(j)
  {sapply(model.best$lev,
        function(niveau){sum(model.best$scaling[,j]*model.best$means[niveau,]) + const_[j]})
   }
)
colnames(cond_Xb) <- paste("LD",1:ncol(model.best$scaling),sep="")
rownames(cond_Xb) <- model.best$lev
print(cond_Xb)
```

    ##          LD1        LD2         LD3         LD4        LD5
    ## 1 -7.0396697  2.6258814 -0.34674338  0.25077114 -0.1781889
    ## 2  0.7545951 -0.9813370  0.31804967  0.09216973 -0.1954021
    ## 3 -1.3149288 -4.2622721 -1.28759834 -0.02176205 -0.2021992
    ## 4  2.5249423  1.3675328 -0.26857348 -0.16058496 -0.1919567
    ## 5 -2.6268373 -0.5689681  0.35952406 -0.36055693 -0.1982584
    ## 6 -0.7030025  2.0080500 -0.04332387  0.07229239 -0.2120466

From pca features functions to linear features functions

``` r
coef_ <- sapply(model.best$lev,function(niveau)
{rowSums(sapply(1:ncol(model.best$scaling),
                function(j){model.best$scaling[,j]*cond_Xb[niveau,j]}))
})
print(coef_)
```

    ##                    1           2          3          4          5          6
    ## absences  3.76527159 -0.56949029 -0.4031075 -0.7975217  1.0168399  0.7921848
    ## G3       -1.05883634  0.04634251 -2.3733891  0.5606673 -0.4720404  0.2411299
    ## G1       -0.25201687 -0.36591398 -0.3279747  0.5907988 -0.7888608  0.2007202
    ## G2        0.27773292 -0.05056911 -0.2381037  0.6964208 -0.1392870  0.5257979
    ## age      -0.08268656 -0.18025884  0.1891581  0.1129770  0.2225942 -0.0752997

From pca intercepts to linear features intercepts

``` r
intercept_ <- sapply(model.best$lev,function(niveau)
     {sum(mapply(prod,const_,cond_Xb[niveau,]))-0.5*sum(cond_Xb[niveau,]^2)+log(model.best$prior[niveau])})
names(intercept_) <- levels(factor(train.sample$cluster))
print(intercept_)
```

    ##          1          2          3          4          5          6 
    ## -30.994334   7.569065  19.707528 -26.472098   3.530489 -17.136820

Below are the functions

**Cluster 1**:  
*F*<sub>1</sub> =  − 30, 99 + 3, 76 \* *a**b**s**e**n**c**e**s* − 1, 06 \* *G*3 + 0, 27 \* *G*1 − 0, 25 \* *G*2 − 0, 08 \* *a**g**e*

**Cluster 2**:  
*F*<sub>2</sub> = 7, 56 − 0, 57 \* *a**b**s**e**n**c**e**s* + 0, 04 \* *G*3 − 0, 05 \* *G*2 − 0, 36 \* *G*1 − 0, 18 \* *a**g*

**Cluster 3**:  
*F*<sub>3</sub> = 19, 7 − 0, 4 \* *a**b**s**e**n**c**e**s* − 2, 37 \* *G*3 − 0, 24 \* *G*2 − 0, 33 \* *G*1 + 0, 19 \* *a**g**e*

**Cluster 4**:  
*F*<sub>4</sub> =  − 26, 47 − 0, 8 \* *a**b**s**e**n**c**e**s* + 0, 56 \* *G*3 + 0, 7 \* *G*2 + 0, 59 \* *G*1 + 0, 11 \* *a**g**e*

**Cluster 5**:  
*F*<sub>5</sub> = 3, 53 + 1, 01 \* *a**b**s**e**n**c**e**s* − 0, 47 \* *G*3 − 0, 14 \* *G*2 − 0, 79 \* *G*1 + 0, 22 \* *a**g**e*

**Cluster 6**:  
*F*<sub>6</sub> =  − 17, 13 + 0, 8 \* *a**b**s**e**n**c**e**s* + 0, 24 \* *G*3 + 0, 52 \* *G*2 + 0, 2 \* *G*1 − 0, 07 \* *a**g**e*

Pour chaque indivi donnée, il suffit de calculer directemnt la
fonction score et prendre le maximum des 6 scores .

## Compute Wilks Lambda from scratch and derive its Pvalue

``` r
Xtrain=data.frame(train.sample[,-which(names(train.sample) == "cluster")])
#number of
p <- ncol(Xtrain)
#dataset length
n <- nrow(Xtrain)
#number of cluster
K <- nlevels(factor(train.sample$cluster))
#Degrees of freedom ;numerator
ddlSuppNum <- K - 1
#Degrees of freedom : denominator
ddlSuppDenom <- n - K - p + 1

#matrice de convariance totale
TOT <- cov(Xtrain)

#WITHIN - covariance intra-classe

WIT <- (1.0/(n-K)) *Reduce("+",
                         lapply(levels(factor(train.sample$cluster)),function(niveau)
                                        {(sum(factor(train.sample$cluster)==niveau)-1)*cov(Xtrain[factor(train.sample$cluster)==niveau,])}))

#lambda de Wilks
#matrices interm?diaires
WITprim <- (n-K)/n*WIT
TOTprim <- (n-1)/n*TOT
#rapport des d?terminants -- Lambda de Wilks
LW <- det(WITprim)/det(TOTprim)
print(LW)
```

    ## [1] 0.03346139

Lower value, So good new!

``` r
#compute pvale
stat=-log(LW) * (n -0.5 *( p-K+1))
pvalue=pchisq(stat, df=p, lower.tail=FALSE)
print (pvalue)
```

    ## [1] 0

## Features importances by Wilk Lambda

``` r
#vecteur des F
FTest <- numeric(p)
#p-value
pvalueFTest <- numeric(p)
#boucle
for (j in 1:p){
  #Lambda corresp.
  LWvar <- det(WITprim[-j,-j])/det(TOTprim[-j,-j])
  #print(LWvar)
  #F
  FTest[j] <- ddlSuppDenom / ddlSuppNum * (LWvar/LW - 1)
  #p-value
  pvalueFTest[j] <- pf(FTest[j],ddlSuppNum,ddlSuppDenom,lower.tail=FALSE)
}


temp <- data.frame(var=colnames(Xtrain),FValue=FTest,pvalue=round(pvalueFTest,6))
print(temp)
```

    ##           var      FValue   pvalue
    ## 1    romantic   0.5030586 0.774006
    ## 2    internet   1.3797111 0.230363
    ## 3         sex   0.4473971 0.815227
    ## 4  activities   0.2596665 0.934874
    ## 5        paid   0.2716775 0.928571
    ## 6   schoolsup   1.1084641 0.354865
    ## 7         age   2.8640812 0.014626
    ## 8    absences 405.0588779 0.000000
    ## 9        Medu   2.1264501 0.061035
    ## 10       Fedu   1.4814370 0.194242
    ## 11   freetime   0.6779326 0.640348
    ## 12         G1  13.2207342 0.000000
    ## 13         G2   3.7129957 0.002615
    ## 14         G3  24.0874576 0.000000
    ## 15      goout   0.9010864 0.480069
    ## 16     health   1.4525203 0.203975
    ## 17  studytime   0.4632619 0.803634
    ## 18 traveltime   0.7713878 0.570679
    ## 19       Walc   1.4039420 0.221275

``` r
temp <- data.frame(var=colnames(Xtrain),FValue=FTest,pvalue=round(pvalueFTest,6)) %>% 
  arrange(FValue)
print(temp)
```

    ##           var      FValue   pvalue
    ## 1  activities   0.2596665 0.934874
    ## 2        paid   0.2716775 0.928571
    ## 3         sex   0.4473971 0.815227
    ## 4   studytime   0.4632619 0.803634
    ## 5    romantic   0.5030586 0.774006
    ## 6    freetime   0.6779326 0.640348
    ## 7  traveltime   0.7713878 0.570679
    ## 8       goout   0.9010864 0.480069
    ## 9   schoolsup   1.1084641 0.354865
    ## 10   internet   1.3797111 0.230363
    ## 11       Walc   1.4039420 0.221275
    ## 12     health   1.4525203 0.203975
    ## 13       Fedu   1.4814370 0.194242
    ## 14       Medu   2.1264501 0.061035
    ## 15        age   2.8640812 0.014626
    ## 16         G2   3.7129957 0.002615
    ## 17         G1  13.2207342 0.000000
    ## 18         G3  24.0874576 0.000000
    ## 19   absences 405.0588779 0.000000
