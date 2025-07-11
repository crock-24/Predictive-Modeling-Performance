# Week 5 Homework
# Cody Rorick

#Problem 7.1 Simulate a single predictor and a nonlinear relationship, such as a sin wave shown in Fig. 7.7, and
#investigate the relationship between the cost, ε, and kernel parameters for a support vector machine model
par(mfrow = c(1,3))
set.seed(100)
x <- runif(100, min = 2, max = 10)
y <- sin(x) + rnorm(length(x)) * .25
sinData <- data.frame(x = x, y = y)
plot(x, y, main = 'C = 1, epsilon = 0.1, kpar = 10')
## Create a grid of x values to use for prediction
dataGrid <- data.frame(x = seq(2, 10, length = 100))

# 7.1a Fit different models using a radial basis function and different values of the cost (the C parameter) and
# ε. Plot the fitted curve. For example
library(kernlab)
rbfSVM <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = list(sigma = 10), C = 1, epsilon = 0.1)
modelPrediction <- predict(rbfSVM, newdata = dataGrid)
## This is a matrix with one column. We can plot the
## model predictions by adding points to the previous plot
points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")
    
#7.2 Friedman (1991) introduced several benchmark data sets create by simulation. 
# One of these simulations used the following nonlinear equation to create data
# where the x values are random variables uniformly distributed between [0, 1] (there are also 5 other non-
# informative variables also created in the simulation). The package mlbench contains a 
# function called mlbench.friedman1 that simulates these data
library(mlbench)
library(caret)
set.seed(200)
trainingData <- mlbench.friedman1(200, sd = 1)
## We convert the 'x' data from a matrix to a data frame
## One reason is that this will give the columns names.
trainingData$x <- data.frame(trainingData$x)
## Look at the data using
featurePlot(trainingData$x, trainingData$y)
## or other methods.
## This creates a list with a vector 'y' and a matrix
## of predictors 'x'. Also simulate a large test set to
## estimate the true error rate with good precision:
testData <- mlbench.friedman1(5000, sd = 1)
testData$x <- data.frame(testData$x)

#7.2a Consider KNN and MARS, which model appears to give the best performance?
### MARS Model
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:15)  
# Fix the seed so that the results can be reproduced
set.seed(100)
marsTuned <- train(trainingData$x, trainingData$y, method = "earth",tuneGrid = marsGrid, trControl = trainControl(method = "cv"))
marsTuned
marsPred <- predict(marsTuned, testData$x)
postResample(pred = marsPred, obs = testData$y)
### KNN Model
#Tune several models on these data. For example:
set.seed(100)
knnModel <- train(x = trainingData$x,y = trainingData$y,method = "knn",preProc = c("center", "scale"),tuneLength = 10)
knnModel
knnPred <- predict(knnModel, newdata = testData$x)
## The function 'postResample' can be used to get the test set
## performance values
postResample(pred = knnPred, obs = testData$y)

#7.2b Does MARS select the informative predictors (those named X1-X5)?
plot(varImp(marsTuned), main = 'MARS Model Informative Predictors')

# 7.5 Exercise 6.3 describes data for a chemical manufacturing process. Use the same data imputation, data
# splitting, and pre-processing steps as before and train several nonlinear regression models.
library(AppliedPredictiveModeling)
library(VIM)
data(ChemicalManufacturingProcess)
chem <- kNN(ChemicalManufacturingProcess, imp_var = FALSE)
trainRows <- createDataPartition(chem$Yield, p = .80, list = FALSE)
chem.predictors <- chem[,-1]
train.pred <- chem.predictors[trainRows,]
test.pred <- chem.predictors[-trainRows,]
Yield <- chem[,1]
train.resp <- Yield[trainRows]
test.resp <- Yield[-trainRows]

# 7.5a Which nonlinear regression model gives the optimal resampling and test set performance? (Please put
# the table of the summary statistics for each tuning parameter and the figure for the tuning
# parameter, then use a table to summarize the RMSE and the best tuning parameter values for all models)
### MARS Model
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:10)
set.seed(100)
marsTuned <- train(train.pred, train.resp, method = "earth",tuneGrid = marsGrid, trControl = trainControl(method = "cv"))
marsTuned
plot(marsTuned)
marsPred <- predict(marsTuned, test.pred)
postResample(pred = marsPred, obs = test.resp)

### KNN Model
set.seed(100)
knnModel <- train(x = train.pred,y = train.resp,method = "knn",preProc = c("center", "scale"),tuneLength = 10)
knnModel
plot(knnModel)
knnPred <- predict(knnModel, newdata = test.pred)
postResample(pred = knnPred, obs = test.resp)

### Radial Support Vector Machine
set.seed(100)
svmRTuned <- train(train.pred, train.resp, method = "svmRadial", preProc = c("center", "scale"), tuneLength = 14,trControl = trainControl(method = "cv"))
svmRTuned
plot(svmRTuned)
svmRPred <- predict(svmRTuned, newdata = test.pred)
postResample(pred = svmRPred, obs = test.resp)

### Partial Least Squares Regression Model
set.seed(100)
plsTune <- train(train.pred, train.resp, method = "pls", tuneLength = 20, trControl = ctrl, preProc = c("center", "scale"))
plsTune
plot(plsTune)
plsPred <- predict(plsTune, newdata = test.pred)
postResample(pred = plsPred, obs = test.resp)

### Multiple Linear Regression
ctrl <- trainControl(method = "cv", number = 5)
set.seed(100)
lmRegFit <- train(train.pred, train.resp, method = "lm", trControl = ctrl, preProc = c("center", "scale"))
lmRegFit

# Lasso Penalized Model
lassoGrid=expand.grid(.fraction=seq(0.01,1,length = 20))
set.seed(100)
lassoTune=train(train.pred,train.resp,method="lasso",tuneGrid=lassoGrid,trControl=ctrl)
lassoTune
plot(lassoTune)

# 7.5b Which predictors are most important in the optimal nonlinear regression model?
plot(varImp(svmRTuned), main = 'SVM Radial Model Informative Predictors')

# 7.5c Do either the biological or process variables dominate the list? How do the top ten important predictors
# compare to the top ten predictors from the optimal linear model?
plot(varImp(lassoTune), main = 'Lasso Model Informative Predictors')








