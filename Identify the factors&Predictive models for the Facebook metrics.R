### 1.To identify the factors that can predict various Facebook metrics (especially different kinds of user engagement)
### 2.To develop predictive models for the Facebook metrics

# Principal Components Analysis
# Regression analysis
# Multilayer Perceptron
# Support vector regression
# Random Forest


## 1. Data preparation

# Load stringr for data preparation
# install.packages("stringr", dependencies = TRUE)
library(stringr)

# Load dplyr for data transformations
# install.packages("dplyr", dependencies = TRUE)
library(dplyr)

# Load caTools for sample.split
# install.packages("caTools", dependencies = TRUE)
library(caTools)

# Load fastDummies to derive dummy variables
# install.packages("fastDummies", dependencies = TRUE)
library(fastDummies)


# Read dataset
Dataset <- read.csv("Live.csv", header=TRUE, sep =",", colClasses=c("character","character", "factor", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric"))

#To split the status_id into company_id and record_id
Dataset$company_id <- str_split_fixed(Dataset[,1], "_", 2)[,1]
Dataset$record_id <- str_split_fixed(Dataset[,1], "_", 2)[,2]

#extract the and date from status_published
Dataset$year<-str_split_fixed(Dataset$status_published, "[ /:]", 5)[,3]
Dataset$date<-as.Date(Dataset$status_published, tryFormats = c("%m/%d/%Y"))

#Delete the useless column
#Dataset <- Dataset[,-1]
#Dataset <- Dataset[,-2]

#Transform the dataframe into numeric
#Dataset <-as.numeric(Dataset[,c(17:27)])
#Dataset$company_id <-as.numeric(Dataset$company_id)
#Dataset$record_id <-as.numeric(Dataset$record_id)

# Create dummy variables for factor and char columns, and add the dummy columns at the end
# https://www.rdocumentation.org/packages/fastDummies/versions/1.6.1/topics/dummy_cols
Dataset<- dummy_cols(Dataset,  select_columns = c("status_type","year"))

# Partition the dataset into 70% trainingSet and 30% testSet
# Set seed in order to get the same set of random nos. every time
set.seed(1234)

# Use sample.split from caTools
# https://www.rdocumentation.org/packages/caTools/versions/1.17.1/topics/sample.split
trainIndex <- sample.split( Dataset$num_shares, SplitRatio = 0.7, group = NULL )

# Create separate training and test set records:
trainingSet <- Dataset[trainIndex,]
testSet <- Dataset[!trainIndex,]



## 2. Perform Principal Components Analysis using psych package
# https://cran.r-project.org/web/packages/psych/psych.pdf

# install.packages("psych", dependencies = TRUE)
library(psych)

# Assess the components (based on eigenvalues)
fa.parallel(trainingSet[,c(5,7:12)], fa="pc", n.iter=100, show.legend=FALSE, main="Scree plot with parallel analysis")

# Perform PCA, and derive the rotated components
#nfactors means the factor to be kept
pc <- principal(trainingSet[,c(5,7:12)], nfactors=2, rotate="varimax", score=TRUE)
pc

# List the scoring formulas for the rotated components
# round(pc$weights, 2)  # rounds to 2 decimal places

# Glue the matrix of rotated component scores to tainingSet using cbind
# https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/cbind
# pc$scores are the PCA component scores
trainingSet <- cbind(trainingSet, pc$scores)

# Generate rotated component scores for testSet
testSet_RCscores <- predict(pc, testSet[,c(5,7:12)], trainingSet[,c(5,7:12)])
testSet <- cbind(testSet,testSet_RCscores)


## 3. Develop Multiple Linear Regression model

# Load MASS package for linear regression, and psych package for PCA
library(MASS)

# Perform linear regression:
fit <- lm(num_shares ~  status_type + date + year+ company_id, 
       data= trainingSet, na.action = na.exclude)
# summary(fit)

# Perform backward stepwise regression
model.regression <- stepAIC(fit, direction="backward")
# model.regression

# Apply regression model to testSet
testSet.predictRegression <- predict(model.regression, testSet)



## 4. Develop neural network model using RSNNS package

# - https://www.rdocumentation.org/packages/RSNNS/versions/0.4-12/topics/mlp
# - https://cran.r-project.org/web/packages/RSNNS/RSNNS.pdf

# install.packages("RSNNS", dependencies=TRUE)
# install.packages("Rcpp", dependencies=TRUE)
library(RSNNS)

# Select the independent variables (input variables)
trainingSet_IV <- trainingSet[,c(4,17:27)]
testSet_IV <- testSet[,c(4,17:27)]

# Create 30% testset within trainingSet
# Target is trainingSet$num_shares
trainingSet_mlp <- splitForTrainingAndTest(trainingSet_IV, trainingSet$num_shares, ratio=0.3)


# normalize the attributes in the dataset
# The normalization parameters used are attached as attributes to the output data
# The normalization parameters are needed to de-normalize the predicted values later
trainingSet_mlp.norm <- normTrainingAndTestSet(trainingSet_mlp, dontNormTargets = FALSE)
testSet_IV.norm <- normalizeData(testSet_IV)


# train the multilayer perceptron
model.mlp <- RSNNS::mlp(trainingSet_mlp.norm$inputsTrain, trainingSet_mlp.norm$targetsTrain, size=10, learnFuncParams=c(0.5), maxit=50, inputsTest=trainingSet_mlp.norm$inputsTest, targetsTest=trainingSet_mlp.norm$targetsTest)
# there are many parameters that can be explored


# Display details of the model:
# summary(model.mlp)
# model.mlp
# weightMatrix(model.mlp)
# extractNetInfo(model.mlp)

# Use the model to make predictions
mlp_pred.norm <- predict(model.mlp, testSet_IV.norm)

# Denormalize the predicted values
testSet.predictMLP <-denormalizeData(mlp_pred.norm, getNormParameters(trainingSet_mlp.norm$targetsTrain))


## 5. Support vector regression

# https://cran.r-project.org/web/packages/e1071/e1071.pdf
# install.packages("e1071", dependencies=TRUE)
library(e1071)

# Alternative package
# install.packages("liquidSVM", dependencies=TRUE)

# Two ways approaches of specifying model
# - using formula
# model_svm <- svm(num_shares ~ ., data = cbind(trainingSet_IV,trainingSet$num_shares), kernel="linear")

# - traditional method: using separate IV dataset and target vector
model_svm <- svm(trainingSet_IV, trainingSet$num_shares, kernel="linear")

# Notes:
# kernel = polynomial, radial basis, sigmoid, tanh
# degree = 2 (needed for kernel of type polynomial)
# Many parameters can be set

# For a linear kernel, the feature weights (coefficients) squared indicate the relative importance of the IVs
coef(model_svm)

# Make prediction
testSet.predictSVM <- predict(model_svm, testSet_IV)



## 6. Random forest (decision tree)
# https://cran.r-project.org/web/packages/randomForest/randomForest.pdf

# install.packages("randomForest", dependencies=TRUE)
library(randomForest)

# 2 methods:
# model_RF <- randomForest(num_shares ~ ., data = cbind(trainingSet_IV,trainingSet$Lifetime_Post_Consumers))
model_RF <- randomForest(trainingSet_IV, trainingSet$num_shares, ntree=50, mtry=15, importance=TRUE)

# Importance of features
round(importance(model_RF), 2)
varImpPlot(model_RF,  n.var=10)

testSet.predictRF <- predict(model_RF, testSet_IV)

## 7. Evaluation of predictive models (for numeric target)

## Metrics package
# https://cran.r-project.org/web/packages/Metrics/Metrics.pdf
# https://www.rdocumentation.org/packages/Metrics/versions/0.1.4
# install.packages("Metrics", dependencies=TRUE)
library(Metrics)

# Comparing predicted values with target values
rmse(testSet$num_shares, testSet.predictRegression)
mae(testSet$num_shares, testSet.predictRegression)


## MLmetrics package
# https://cran.r-project.org/web/packages/MLmetrics/MLmetrics.pdf
# install.packages("MLmetrics", dependencies=TRUE)
library(MLmetrics)


# Predicted values & Target values
# testSet.predictRegression
R2_Score(testSet.predictRegression, testSet$num_shares)
RMSE(testSet.predictRegression, testSet$num_shares)
MAE(testSet.predictRegression, testSet$num_shares)

# testSet.predictMLP
R2_Score(testSet.predictMLP, testSet$num_shares)
RMSE(testSet.predictMLP, testSet$num_shares)
MAE(testSet.predictMLP, testSet$num_shares)

# testSet.predictSVM
R2_Score(testSet.predictSVM, testSet$num_shares)
RMSE(testSet.predictSVM, testSet$num_shares)
MAE(testSet.predictSVM, testSet$num_shares)

# testSet.predictRF
R2_Score(testSet.predictRF, testSet$num_shares)
RMSE(testSet.predictRF, testSet$num_shares)
MAE(testSet.predictRF, testSet$num_shares)


## Graphs -- separate file


### THE END ###

