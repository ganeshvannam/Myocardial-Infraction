
#QDA Model
# Load required libraries
library(caret)

# Split data: Stratified random sampling (80-20 split)
set.seed(108)
train_index <- createDataPartition(my_target, p = 0.8, list = FALSE)
train_data <- my_inp[train_index, ]
train_target <- my_target[train_index]
test_data <- my_inp[-train_index, ]
test_target <- my_target[-train_index]

# Combine training data for caret processing
train_combined <- data.frame(train_data, Target = train_target)
test_combined <- data.frame(test_data, Target = test_target)

# Preprocess: PCA transformation
pre_proc <- preProcess(train_combined[, -ncol(train_combined)], method = c("center", "scale", "pca"))

# Apply PCA transformation to training and test data
train_pca <- predict(pre_proc, train_combined[, -ncol(train_combined)])
test_pca <- predict(pre_proc, test_combined[, -ncol(test_combined)])

# Combine PCA-transformed data with target variable
train_combined_pca <- data.frame(train_pca, Target = train_target)
test_combined_pca <- data.frame(test_pca, Target = test_target)

# Train QDA model
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = defaultSummary)
set.seed(108)
qda_model_pca <- train(
  Target ~ .,                  # Formula (predict all features for Target)
  data = train_combined_pca,    # Training data
  method = "qda",              # QDA method
  metric = "Kappa",            # Optimize based on Kappa
  trControl = train_control,   # Training control
  prior = c(0.2, 0.8)          # Class priors (adjust as needed)
)

# Display the model
print(qda_model_pca)

# Evaluate model on test data (PCA-transformed test data)
predictions_qda_pca <- predict(qda_model_pca, newdata = test_combined_pca)
confusion_matrix_qda_pca <- confusionMatrix(predictions_qda_pca, test_target)
print(confusion_matrix_qda_pca)

# Check the number of components retained after PCA
num_components <- ncol(pre_proc$rotation)  # Number of principal components
cat("Number of features (principal components) retained after PCA:", num_components, "\n")

###RDA MODEL
# Remove near-zero variance predictors from bio
nzv <- nearZeroVar(bio)
bio <- bio[, -nzv]

# Define training control
ctrl <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = defaultSummary
)

# Stratified train-test split
set.seed(123)
trainIndex <- createDataPartition(injury, p = 0.8, list = FALSE)

trainPredictors <- bio[trainIndex, ]
testPredictors <- bio[-trainIndex, ]
trainTarget <- injury[trainIndex]
testTarget <- injury[-trainIndex]

trainData <- data.frame(trainPredictors, injury = trainTarget)
testData <- data.frame(testPredictors, injury = testTarget)

set.seed(124)
rdaFit <- train(
  x = trainPredictors, 
  y = trainTarget, 
  method = "rda", 
  metric = "Kappa", 
  preProc = c("center", "scale"), 
  tuneGrid = expand.grid(.gamma = seq(0.1, 1, by = 0.2), .lambda = seq(0.1, 1, by = 0.2)),
  trControl = ctrl
)

print(rdaFit)

# Plot tuning results
plot(rdaFit, main = "RDA Hyperparameter Tuning")

# Evaluate on test data
rdaPred <- predict(rdaFit, newdata = testPredictors)
confusionMatrix(data = rdaPred, reference = testTarget)

### FDA MODEL
library(earth)

set.seed(124)
fdaFit <- train(
  x = trainPredictors, 
  y = trainTarget, 
  method = "fda", 
  metric = "Kappa", 
  preProc = c("center", "scale"), 
  trControl = ctrl
)

print(fdaFit)

# Plot tuning results
plot(fdaFit, main = "FDA Hyperparameter Tuning")

# Evaluate on test data
fdaPred <- predict(fdaFit, newdata = testPredictors)
confusionMatrix(data = fdaPred, reference = testTarget)


##MDA Model
# Load required libraries
library(caret)

# Ensure valid levels for the target variable
levels(my_target) <- c("no_complication", "complication")

# Split data: Stratified random sampling (80-20 split)
set.seed(211)
train_index <- createDataPartition(my_target, p = 0.8, list = FALSE)
train_data <- my_inp[train_index, ]
train_target <- my_target[train_index]
test_data <- my_inp[-train_index, ]
test_target <- my_target[-train_index]

# Combine training data for caret processing
train_combined <- data.frame(train_data, Target = train_target)
test_combined <- data.frame(test_data, Target = test_target)

# Train control: 10-fold cross-validation
train_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = defaultSummary
)

# Train Mixture Discriminant Analysis (MDA) model
set.seed(108)
mda_model_pca <- train(
  Target ~ .,                  # Formula (predict all features for Target)
  data = train_combined,        # Training data
  method = "mda",               # MDA method
  metric = "Kappa",             # Optimize based on Kappa
  trControl = train_control,    # Training control
  prior = c(0.2, 0.8)           # Class priors (adjust as needed)
)

# Display the model
print(mda_model_pca)

# Evaluate model on test data
predictions_mda_pca <- predict(mda_model_pca, newdata = test_combined)
confusion_matrix_mda_pca <- confusionMatrix(predictions_mda_pca, test_target)
print(confusion_matrix_mda_pca)

# Check the number of components retained after PCA
num_components <- ncol(pre_proc$rotation)  # Number of principal components
cat("Number of features (principal components) retained after PCA:", num_components, "\n")

## KNN Model
set.seed(127)
knnFit <- train(
  x = trainPredictors, 
  y = trainTarget, 
  method = "knn", 
  metric = "Kappa", 
  preProc = c("center", "scale"), 
  tuneGrid = data.frame(.k = 1:20), 
  trControl = ctrl
)

print(knnFit)

# Plot tuning results
plot(knnFit, main = "KNN Hyperparameter Tuning")

# Evaluate on test data
knnPred <- predict(knnFit, newdata = testPredictors)
confusionMatrix(data = knnPred, reference = testTarget)

### NEURAL NETWORK
library(nnet)

set.seed(126)
nnetFit <- train(
  x = trainPredictors, 
  y = trainTarget, 
  method = "nnet", 
  metric = "Kappa", 
  preProc = c("center", "scale"), 
  tuneGrid = expand.grid(.size = 1:10, .decay = c(0, 0.1, 1)), 
  trControl = ctrl,
  trace = FALSE
)

print(nnetFit)

# Plot tuning results
plot(nnetFit, main = "NN Hyperparameter Tuning")

# Evaluate on test data
nnetPred <- predict(nnetFit, newdata = testPredictors)
confusionMatrix(data = nnetPred, reference = testTarget)



## SVM Model
# Load required libraries
library(caret)

# Ensure valid levels for the target variable
levels(my_target) <- c("no_complication", "complication")

# Stratified random sampling (80-20 split)
set.seed(125)  # Set seed for reproducibility
train_index <- createDataPartition(my_target, p = 0.8, list = FALSE)

# Split the data into training and testing sets
train_data <- my_inp[train_index, ]
train_target <- my_target[train_index]
test_data <- my_inp[-train_index, ]
test_target <- my_target[-train_index]

# Combine training data for caret processing
train_combined <- data.frame(train_data, Target = train_target)
test_combined <- data.frame(test_data, Target = test_target)

# Define the training control with 10-fold cross-validation
train_control <- trainControl(
  method = "cv",        # Cross-validation
  number = 10,          # 10 folds
  classProbs = TRUE,    # Enable probabilities
  summaryFunction = defaultSummary  # Evaluate with ROC, Sensitivity, and Specificity
)

# Train the Support Vector Machine model
set.seed(130)  # For reproducibility
svm_model <- train(
  Target ~ .,                  # Formula (predict all features for Target)
  data = train_combined,        # Training data
  method = "svmRadial",         # Radial SVM method
  trControl = train_control,    # Training control
  metric = "Kappa",             # Optimize based on Kappa
  preProcess = c("center", "scale"), # Preprocessing: centering and scaling
  tuneLength = 5                # Number of parameter grid values to explore
)

# Display the model
print(svm_model)

# Evaluate model on test data
predictions_svm <- predict(svm_model, newdata = test_combined)
confusion_matrix_svm <- confusionMatrix(predictions_svm, test_target)
print(confusion_matrix_svm)


# Naive Bayes Model
levels(my_target) <- c("no_complication", "complication")
set.seed(125)  # Set seed for reproducibility
train_index <- createDataPartition(my_target, p = 0.8, list = FALSE)
# Split the data into training and testing sets
train_data <- my_inp[train_index, ]
train_target <- my_target[train_index]
test_data <- my_inp[-train_index, ]
test_target <- my_target[-train_index]

# Combine training data for caret processing
train_combined <- data.frame(train_data, Target = train_target)
# Define the training control with 10-fold cross-validation
train_control <- trainControl(
  method = "cv",        # Cross-validation
  number = 10,          # 10 folds
  classProbs = TRUE,    # Enable probabilities
  summaryFunction = defaultSummary  # Evaluate with ROC, Sensitivity, and Specificity
)
set.seed(132) 
nb_model <- train(
  Target ~ .,                   # Formula specifying the target variable
  data = train_combined,         method = "nb",
  trControl = train_control,     # Training control
  metric = "Kappa",              # Evaluation metric
  tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
  preProcess = c("center", "scale","BoxCox","spatialSign")  # Preprocessing: centering and scaling
)
print(nb_model)
plot(nb_model)

test_combined <- data.frame(test_data, Target = test_target)
predictions_nb <- predict(nb_model, newdata = test_combined)
confusion_matrix_nb <- confusionMatrix(predictions_nb , test_target)
# Print the confusion matrix
print(confusion_matrix_nb)
