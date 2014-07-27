---
title: "PML.md"
author: "evoletttt (Lovette Regner)"
date: "Monday, July 28, 2014"
output: html_document
---

## SUMMARY ##
This model uses machine learning methods to predict the movement type: sitting, standing,... of a person given measurements from a movement tracker. The original data set was taken from this website: http://groupware.les.inf.puc-rio.br/har.


## Load and clean data. ##
First, download the data from the link. There are 160 variables, but most of them have NA and blank values. Subset only the complete columns.
```{r pml}
library(caret)
library(rpart)
library(rattle)
library(knitr)
library(markdown)

url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url, "pml.csv")
pml <- read.csv("pml.csv")

inc<- {}
blank <- {}
for (i in 1:160) {
  inc <- c(inc,anyNA(pml[,i]))
  blank <- c(blank, any(pml[,i] == ""))
}
pml <- pml[,!(inc|blank)]
```


## Make data tidy. ##
1. Convert new_window factor variable to 0/1.
2. Create dummy variables for the names because they are unrelated categories, not related discrete values.
3. Remove unnecessary and repeated time variables and original name variable.
4. Combine dummy name variables and big pml dataset.
5. Interchange the 7th column (new_window: categorical) and 8th column (raw_timestamp_part_1) so all the categorical variables are together on the left side.
```{r tidy}
pml[,6] <- factor(pml[,6], labels = c(0,1))
pml[,6] <- as.numeric(as.character(pml[,6]))
pml[,2] <- as.character(pml[,1])

adelmo <- pml[,2] == "adelmo"
carlitos <- pml[,2] == "carlitos"
charles <- pml[,2] == "charles"
eurico <- pml[,2] == "eurico"
jeremy <- pml[,2] == "jeremy"
pedro <- pml[,2] == "pedro"

pml <- pml[,-c(1,2,4,5)]

pml <- cbind(adelmo, carlitos, charles, eurico, jeremy, pedro, pml)

temp <- pml[,7]
tempname <- names(pml)[7]
pml[,7] <- pml[,8]
names(pml)[7] <- names(pml)[8]
pml[,8] <- temp
names(pml)[8] <- tempname
```


## Create data partition. ##
Divide the dataset into training(70%) and testing(30%) sets.
```{r partition}
inTrain <- createDataPartition(pml$classe, p = 0.7, list = FALSE)
training <- pml[inTrain,]
testing <- pml[-inTrain,]
```


## Preprocessing ##
Use standardization to preprocess the data. This is to control the weird skewness caused by the differing magnitudes of the variables. Preprocess the processed data again using PCA with a 95% threshold. As seen below, PCA needs 25 components to capture 95% of the variance of the 54 variables.
```{r std}
prep <- preProcess(training[,-c(1:7,62)], method = c("center", "scale"))
trainPC <- predict(prep, training[,-c(1:7,62)])
testPC <- predict(prep, testing[,-c(1:7, 62)])
prep2 <- preProcess(trainPC, method = "pca", thresh = 0.95)
prep2
```


## Training and Predicting ##
Train the data and put back the categorical variables. Then use the rpart clustering method to model the data.
```{r pca}
trainPC2 <- predict(prep2, trainPC)
trainPC2 <- cbind(training$classe, training[,1:7], trainPC2)
names(trainPC2)[1] <- "classe"
testPC2 <- predict(prep2, testPC)
testPC2 <- cbind(testing$classe, testing[,1:7], testPC2)
names(testPC2)[1] <- "classe"

fit <- train(classe~., method = "rpart", data = trainPC2)
fancyRpartPlot(fit$finalModel)
print(fit$finalModel)
```


## Testing ##
Predict the result of the test set and compute for the accuracy.
```{r test}
results <- predict(fit, newdata = testPC2)
accuracy <- (results == testPC2$classe)*1/length(results)
sum(accuracy)
```
The computed accuracy is only 37.31%.


## Testing ##
```{r testtt}
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url, "test.csv")
test <- read.csv("test.csv")

inc<- {}
blank <- {}
for (i in 1:160) {
  inc <- c(inc,anyNA(test[,i]))
  blank <- c(blank, any(test[,i] == ""))
}
test <- test[,!(inc|blank)]

test[,6] <- factor(test[,6], labels = c(0,1))
test[,6] <- as.numeric(as.character(test[,6]))
test[,2] <- as.character(test[,1])

adelmo <- test[,2] == "adelmo"
carlitos <- test[,2] == "carlitos"
charles <- test[,2] == "charles"
eurico <- test[,2] == "eurico"
jeremy <- test[,2] == "jeremy"
pedro <- test[,2] == "pedro"

test <- test[,-c(1,2,4,5)]

test <- cbind(adelmo, carlitos, charles, eurico, jeremy, pedro, test)

temp <- test[,7]
tempname <- names(test)[7]
test[,7] <- test[,8]
names(test)[7] <- names(test)[8]
test[,8] <- temp
names(test)[8] <- tempname


testPC <- predict(prep, test[,-c(1:7, 62)])
testPC2 <- cbind(test$classe, test[,1:7], testPC2)
results <- predict(fit, newdata = testPC2)
```
