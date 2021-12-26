
# prediction   : model predictions
# cluster       :reference 

#=====USEFUL REFERENCES : https://www.datascienceblog.net/==================

### OVERALL ACCURACY
calculate.accuracy <- function(predictions, ref.labels) {
  return(length(which(predictions == ref.labels)) / length(ref.labels))
}

### OVERALL WEIGHT ACCURACY

calculate.w.accuracy <- function(predictions, ref.labels, weights) {
  lvls <- levels(ref.labels)
  if (length(weights) != length(lvls)) {
    stop("Number of weights should agree with the number of classes.")
  }
  if (sum(weights) != 1) {
    stop("Weights do not sum to 1")
  }
  accs <- lapply(lvls, function(x) {
    idx <- which(ref.labels == x)
    return(calculate.accuracy(predictions[idx], ref.labels[idx]))
  })
  acc <- mean(unlist(accs))
  return(acc)
}

## FUNCTIONS TO CUSTOMIZE the 2 previous functions

accuracy_rate<-function (prediction, cluster)
{
  acc <- calculate.accuracy(factor(prediction),factor(cluster))
  print(paste0("Accuracy is: ", round(acc, 2)))
  
  weights <- rep(1 / length(levels(cluster)), length(levels(cluster)))
  w.acc <- calculate.w.accuracy(factor(prediction), factor(cluster), weights)
  
  print(paste0("Weighted accuracy is: ", round(w.acc, 2)))
  
}     


##CONFUSION MATRIX FOR MULTICLASSIFICATION

get.conf.stats <- function(cm) {
  out <- vector("list", length(cm))
  for (i in seq_along(cm)) {
    x <- cm[[i]]
    tp <- x$table[x$positive, x$positive] 
    fp <- sum(x$table[x$positive, colnames(x$table) != x$positive])
    fn <- sum(x$table[colnames(x$table) != x$positie, x$positive])
    # TNs are not well-defined for one-vs-all approach
    elem <- c(tp = tp, fp = fp, fn = fn)
    out[[i]] <- elem
  }
  df <- do.call(rbind, out)
  rownames(df) <- unlist(lapply(cm, function(x) x$positive))
  return(as.data.frame(df))
}



##Compute F1
###Micro F1
get.micro.f1 <- function(cm) {
  cm.summary <- get.conf.stats(cm)
  tp <- sum(cm.summary$tp)
  fn <- sum(cm.summary$fn)
  fp <- sum(cm.summary$fp)
  pr <- tp / (tp + fp)
  re <- tp / (tp + fn)
  f1 <- 2 * ((pr * re) / (pr + re))
  return(f1)
}

###Macro F1
get.macro.f1 <- function(cm) {
  c <- cm[[1]]$byClass # a single matrix is sufficient
  re <- sum(c[, "Recall"]) / nrow(c)
  pr <- sum(c[, "Precision"]) / nrow(c)
  f1 <- 2 * ((re * pr) / (re + pr))
  return(f1)
}

library(caret)


#CUSTOMIZE micro F1 and macro F1
F1_rate<-function(prediction, cluster)


{
  cm <- vector("list", length(levels(factor(cluster))))
  for (i in seq_along(cm)) {
    positive.class <- levels(cluster)[i]
    # in the i-th iteration, use the i-th class as the positive class
    cm[[i]] <- confusionMatrix(factor(prediction), factor(cluster), 
                               positive = positive.class)
   }
  
  macro.f1 <- get.macro.f1(cm)
  micro.f1 <- get.micro.f1(cm)
  
  print(paste0("Macro F1 is: ", round(macro.f1, 2)))
  print(paste0("Micro F1 is: ", round(micro.f1, 2)))
  


}