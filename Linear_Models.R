# Load required libraries
library(ggplot2)
library(dplyr)
library(VIM)
library(caret)
library(naniar)
library(tidyr)

# Read the dataset
my <- read.csv("C:\\Users\\vanna\\Downloads\\Myocardial_infarction_complications_Database.csv")

# View dataset structure
head(my)
dim(my)
str(my)
colnames(my)

# Separating input columns and target column (121st column)
input_columns <- setdiff(1:123, 121)  # All columns except 121st
my_inp <- my[, input_columns]

# Target column (121st column)
excluded_column_name <- colnames(my)[121]
cat("The name of the excluded column (target):", excluded_column_name, "\n")
my_target <- my[, 121]
my_target <- as.factor(my_target)
cat("Levels of the target variable:", levels(my_target), "\n")

# Bar chart for target distribution
ggplot(data = data.frame(Target = my_target), aes(x = Target)) +
  geom_bar(fill = "lightblue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of the Target Variable", x = "Target Levels", y = "Frequency")

# Preprocessing: Converting binary variables to factors
binary_columns <- c("SEX", "FIBR_PREDS", "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", 
                    "A_V_BLOK", "OTEK_LANC", "RAZRIV", "DRESSLER", "REC_IM", 
                    "P_IM_STEN", "IBS_NASL", "SIM_GIPERT", "GIPO_K", "GIPER_NA", 
                    "NA_KB", "NITR_S", "IM_PG_P", "ritm_ecg_p_01", "ritm_ecg_p_02", 
                    "ritm_ecg_p_04", "ritm_ecg_p_06", "ritm_ecg_p_07", "ritm_ecg_p_08", 
                    "n_r_ecg_p_01", "n_r_ecg_p_02", "n_r_ecg_p_03", "n_r_ecg_p_04", 
                    "n_r_ecg_p_05", "n_r_ecg_p_06", "n_r_ecg_p_08", "n_r_ecg_p_09", 
                    "n_r_ecg_p_10", "n_p_ecg_p_01", "n_p_ecg_p_03", "n_p_ecg_p_04", 
                    "n_p_ecg_p_05", "n_p_ecg_p_06", "n_p_ecg_p_07", "n_p_ecg_p_08", 
                    "n_p_ecg_p_09", "n_p_ecg_p_10", "fibr_ter_01", "fibr_ter_02", 
                    "fibr_ter_03", "fibr_ter_05", "fibr_ter_06", "fibr_ter_07", 
                    "fibr_ter_08", "nr_11", "nr_01", "nr_02", "nr_03", "nr_04", 
                    "nr_07", "nr_08", "np_01", "np_04", "np_05", "np_07", "np_08", 
                    "np_09", "np_10", "endocr_01", "endocr_02", "endocr_03", 
                    "zab_leg_01", "zab_leg_02", "zab_leg_03", "zab_leg_04", 
                    "zab_leg_06", "O_L_POST", "K_SH_POST", "MP_TP_POST", "SVT_POST", 
                    "GT_POST", "FIB_G_POST", "n_p_ecg_p_11", "n_p_ecg_p_12", 
                    "NOT_NA_KB", "LID_KB", "LID_S_n", "B_BLOK_S_n", "ANT_CA_S_n", 
                    "GEPAR_S_n", "ASP_S_n", "TRENT_S_n")

# Convert binary columns to factors
my_inp[binary_columns] <- lapply(my_inp[binary_columns], as.factor)

# Convert ordinal variables to ordered factors
ordinal_columns <- list(
  "FK_STENOK" = c(0, 1, 2, 3, 4),
  "INF_ANAM" = c(0, 1, 2, 3),
  "STENOK_AN" = c(0, 1, 2, 3, 4, 5, 6),
  "IBS_POST" = c(0, 1, 2),
  "DLIT_AG" = c(0, 1, 2, 3, 4, 5, 6, 7),
  "ZSN_A" = c(0, 1, 2, 3, 4),
  "ant_im" = c(0, 1, 2, 3, 4),
  "lat_im" = c(0, 1, 2, 3, 4),
  "inf_im" = c(0, 1, 2, 3, 4),
  "post_im" = c(0, 1, 2, 3, 4),
  "TIME_B_S" = c(1, 2, 3, 4, 5, 6, 7, 8, 9),
  "R_AB_1_n" = c(0, 1, 2, 3),
  "R_AB_2_n" = c(0, 1, 2, 3),
  "R_AB_3_n" = c(0, 1, 2, 3),
  "NA_R_1_n" = c(0, 1, 2, 3, 4),
  "NA_R_2_n" = c(0, 1, 2, 3),
  "NA_R_3_n" = c(0, 1, 2),
  "NOT_NA_1_n" = c(0, 1, 2, 3, 4),
  "NOT_NA_2_n" = c(0, 1, 2, 3),
  "NOT_NA_3_n" = c(0, 1, 2),
  "GB" = c(0, 1, 2, 3)
)

# Convert ordinal columns to ordered factors
for (col in names(ordinal_columns)) {
  my_inp[[col]] <- factor(my_inp[[col]], levels = ordinal_columns[[col]], ordered = TRUE)
}

# Convert remaining numeric columns
remaining_cols <- setdiff(colnames(my_inp), c(binary_columns, names(ordinal_columns)))
my_inp[remaining_cols] <- lapply(my_inp[remaining_cols], as.numeric)

# Check missing values
cat("Proportion of missing values in the dataset:", sum(is.na(my_inp)) / prod(dim(my_inp)) * 100, "%\n")
colSums(is.na(my_inp)) / nrow(my_inp) * 100

# Visualizing missing values
gg_miss_fct(cbind(my_inp, class = my_target), fct = class)

# Remove irrelevant features (ID, IBS_NASL, KFK_BLOOD)
my_inp <- select(my_inp, -c("ID", "IBS_NASL", "KFK_BLOOD"))

# Handling missing values using KNN imputation
my_inp <- kNN(my_inp, k = 5, imp_var = FALSE)

# Check missing values after imputation
cat("Missing values after imputation:", sum(is.na(my_inp)), "\n")

# Remove near-zero variance features
nzv <- nearZeroVar(my_inp, saveMetrics = TRUE)
nzv_predictors <- rownames(nzv[nzv$nzv == TRUE, ])
cat("Number of zero or near-zero variance predictors:", length(nzv_predictors), "\n")
my_inp <- my_inp[, !(colnames(my_inp) %in% nzv_predictors)]

# Create dummy variables for categorical data
categorical_columns <- sapply(my_inp, is.factor)
categorical_data <- my_inp[, categorical_columns]
dummy_variables <- model.matrix(~ . - 1, data = categorical_data)

# Combine dummy variables with other data
my_inp <- my_inp %>%
  select(where(~ !is.factor(.))) %>%
  bind_cols(as.data.frame(dummy_variables))

# Check final data structure
cat("Final data dimensions:", dim(my_inp), "\n")
str(my_inp)

# Final checks
num_continuous <- sum(sapply(my_inp, is.numeric))
num_categorical <- sum(sapply(my_inp, is.factor))
cat("Number of continuous variables:", num_continuous, "\n")
cat("Number of categorical variables:", num_categorical, "\n")


# Load required libraries
library(caret)
library(MASS)      # For LDA
library(pls)       # For PLSDA
library(glmnet)    # For Penalized models
library(corrplot)  # For correlation plots (optional)

# Set target levels (ensure binary classification)
levels(my_target) <- c("no_complication", "complication")

# Stratified random sampling (80-20 split)
set.seed(124) # Set seed for reproducibility
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
  summaryFunction = defaultSummary  # Evaluate with ROC, Sensitivity, Specificity
)

# ========== Logistic Regression Model ==========
logistic_model <- train(
  Target ~ .,                  # Formula (predict all features for Target)
  data = train_combined,       # Training data
  method = "glm",              # Logistic regression
  trControl = train_control,   # Training control
  metric = "Kappa",            # Metric to evaluate model performance
  preProcess = c("center", "scale")  # Preprocessing: centering and scaling
)

# Print Logistic Regression model summary
print(logistic_model)
plot(logistic_model)

# Predict and evaluate Logistic Regression model
predictions_logistic <- predict(logistic_model, newdata = test_combined)
confusion_matrix_logistic <- confusionMatrix(predictions_logistic, test_target)
print(confusion_matrix_logistic)

# ========== LDA (Linear Discriminant Analysis) Model ==========
lda_model <- train(
  Target ~ .,                  # Formula (predict all features for Target)
  data = train_combined,       # Training data
  method = "lda",              # LDA model
  trControl = train_control,   # Training control
  metric = "Kappa",            # Metric to evaluate model performance
  preProcess = c("center", "scale")  # Preprocessing: centering and scaling
)

# Print LDA model summary
print(lda_model)
plot(lda_model)

# Predict and evaluate LDA model
predictions_lda <- predict(lda_model, newdata = test_combined)
confusion_matrix_lda <- confusionMatrix(predictions_lda, test_target)
print(confusion_matrix_lda)

# ========== PLSDA (Partial Least Squares Discriminant Analysis) Model ==========
plsdaGrid <- expand.grid(ncomp = seq(1, 30, by = 3))  # Grid for tuning ncomp

set.seed(123)  # For reproducibility
plsda_model <- train(
  Target ~ .,                  # Formula (predict all features for Target)
  data = train_combined,       # Training data
  method = "pls",              # PLSDA model
  tuneGrid = plsdaGrid,        # Grid for tuning PLS components
  trControl = train_control,   # Training control
  metric = "Kappa",            # Metric to evaluate model performance
  preProcess = c("center", "scale")  # Preprocessing: centering and scaling
)

# Print PLSDA model summary
print(plsda_model)
plot(plsda_model)

# Predict and evaluate PLSDA model
predictions_plsda <- predict(plsda_model, newdata = test_combined)
confusion_matrix_plsda <- confusionMatrix(predictions_plsda, test_target)
print(confusion_matrix_plsda)

# Variable Importance for PLSDA
varImp(plsda_model)

# ========== Penalized Model (Elastic Net - glmnet) ==========
set.seed(125)  # Set seed for reproducibility

pm_model <- train(
  Target ~ .,                  # Formula (predict all features for Target)
  data = train_combined,       # Training data
  method = "glmnet",           # Penalized regression (Elastic Net)
  trControl = train_control,   # Training control
  metric = "Kappa",            # Metric to evaluate model performance
  preProcess = c("center", "scale")  # Preprocessing: centering and scaling
)

# Print Penalized Model (Elastic Net) summary
print(pm_model)
plot(pm_model)

# Predict and evaluate Penalized model
predictions_pm <- predict(pm_model, newdata = test_combined)
confusion_matrix_pm <- confusionMatrix(predictions_pm, test_target)
print(confusion_matrix_pm)


