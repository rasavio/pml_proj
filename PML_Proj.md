---
title: "Prediction Assignment"
author: "Ryan Savio"
date: "March 20, 2020"
output: 
  html_document:
    keep_md: true
---

# Overview
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

The goal of this project is to predict the manner in which they did the exercise.


# Report

## Data Preperation

First we setup the environment by attaching required packages and loading the data files.

```r
# seed for reproducability
set.seed(4444)

# libraries
library(tidyverse)
library(ggplot2)
library(caret)
library(parallel)
library(doParallel)

# configure parallel processing
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

# download data
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if (!file.exists("pml_training.csv")) download.file(train_url, "pml_training.csv")

if (!file.exists("pml_testing.csv")) download.file(test_url, "pml_testing.csv")

# read data
train_tbl <- read.csv("pml_training.csv")
test_tbl <- read.csv("pml_testing.csv") %>%
  mutate_if(is.logical, ~ as.factor(.))

# check training dimensions
dim(train_tbl)
sum(is.na(train_tbl))
mean(is.na(train_tbl))
```

Next we clean the  data by removing incomplete rows, as well as any columns that will not be useful for predictive modelling.

```r
# remove first seven colums as their indentifiers, not useful for modelling
train_tbl <- select(train_tbl, -(1:7))
test_tbl <- select(test_tbl, -(1:7))

# remove variables that are near-zero-variance
nzv <- nearZeroVar(train_tbl)

train_tbl <- select(train_tbl, -nzv)
test_tbl <- select(test_tbl, -nzv)

# remove columns where 95% of values are blank or NA
cols_to_remove <- sapply(train_tbl, function(x) mean(is.na(x))) > 0.95

train_tbl <- train_tbl[, cols_to_remove == FALSE]
test_tbl <- test_tbl[, cols_to_remove == FALSE]
```

Lastly we split this cleaned training data (`train_tbl`) set 70/30 in order to create `train_data` (70%) for model training and `test_data` (30%) for validation. Without this step we would not be able to assess the performance of our model and would be blindly assuming any predictions we create to `test_tbl` are valuable.

```r
in_train <- createDataPartition(train_tbl$classe, p = 0.7, list = FALSE)

train_data <- train_tbl[in_train, ]
test_data <- train_tbl[-in_train, ]
```

## Model Building
I've decided to try a Random Forest model due to their being so many features in this data set. This will be set to use 2-fold cross-validation to select optimal tuning parameters.

```r
# fit model
n <- 2
trees <- 250

fit_control <- trainControl(method = "cv", number = n, verboseIter = FALSE, allowParallel = TRUE)
#fit_control <- trainControl(method = "cv", number = n, verboseIter = FALSE)

fit <- train(classe ~ ., data = train_data, method = "rf", trControl = fit_control, ntree = trees)

# view parameters
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = ..1, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 250
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.82%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    2    1    0    1 0.001024066
## B   20 2624   14    0    0 0.012791573
## C    0   14 2371   11    0 0.010434057
## D    0    1   33 2216    2 0.015985790
## E    0    3    2    9 2511 0.005544554
```

## Model Evaluation
Now we can use the fitted model to predict the label `classe` for our validation data (`test_data`) and check performance via the confusion matrix.

```r
# prediction
preds <- predict(fit, newdata = test_data)

# confusion matrix
confusionMatrix(test_data$classe, preds)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    3    0    0    0
##          B    6 1126    6    1    0
##          C    0    6 1015    5    0
##          D    0    1   13  949    1
##          E    0    0    5    1 1076
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9918         
##                  95% CI : (0.9892, 0.994)
##     No Information Rate : 0.285          
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9897         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9912   0.9769   0.9927   0.9991
## Specificity            0.9993   0.9973   0.9977   0.9970   0.9988
## Pos Pred Value         0.9982   0.9886   0.9893   0.9844   0.9945
## Neg Pred Value         0.9986   0.9979   0.9951   0.9986   0.9998
## Prevalence             0.2850   0.1930   0.1766   0.1624   0.1830
## Detection Rate         0.2839   0.1913   0.1725   0.1613   0.1828
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9979   0.9942   0.9873   0.9948   0.9989
```

The accuracy for the model is 99.2%, thus my predicted accuracy for the out-of-sample error is 0.8%

## Re-training Model
Before we can run our final prediction on the test data set for this project (`test_tbl`) we will need to train the model on the complete training set (`train_tbl`).

```r
# fit model
fit_control <- trainControl(method = "cv", number = n, verboseIter = FALSE, allowParallel = TRUE)

fit <- train(classe ~ ., data = train_tbl, method = "rf", trControl = fit_control, ntree = trees)

# view parameters
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = ..1, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 250
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.44%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5575    2    2    0    1 0.0008960573
## B   20 3772    4    1    0 0.0065841454
## C    0   12 3399   11    0 0.0067212157
## D    0    0   20 3193    3 0.0071517413
## E    0    1    5    5 3596 0.0030496257
```

## Making Predictions
Now we can use the re-trained model to run predictions on the test set. The resuls of which will be exported to a flat file (`.csv`).

```r
# prediction
preds <- predict(fit, newdata = test_tbl)

# bind predictions to test data
output <- cbind(test_tbl, preds)

# write output
write.csv(output, "project_output.csv")
```

