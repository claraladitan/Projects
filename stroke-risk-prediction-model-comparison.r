# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Project Objective
# 
# The goal of this project is to leverage machine learning to predict stroke risk. By exploring a comprehensive stroke dataset, I aimed to understand which clinical indicators and symptoms best predict whether an individual is at risk (as indicated by the binary target variable at_risk_binary). To achieve this, I:
# 
# * Explored and preprocessed the dataset,
# * Developed several models (including logistic regression, decision trees, and ensemble methods like XGBoost and regularized logistic regression with glmnet),
# * Evaluated and compared these models using cross-validation and various performance metrics,
# * Investigated potential data leakage to ensure reliable results.
#   
# This notebook demonstrates how advanced modeling techniques can be applied to healthcare data, potentially assisting in early identification of high-risk patients and informing better preventative strategies.

# %% [code] {"_execution_state":"idle","execution":{"iopub.status.busy":"2025-02-24T21:17:27.863454Z","iopub.execute_input":"2025-02-24T21:17:27.865501Z","iopub.status.idle":"2025-02-24T21:17:31.731870Z","shell.execute_reply":"2025-02-24T21:17:31.730116Z"},"jupyter":{"outputs_hidden":false}}
# Load libraries
library(tidyverse) #for data manipulation
library(janitor) #to clean up column names
library(corrplot) #for correlation matrix
library(caret) #for machine learning models
library(rpart) #for decison trees

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:31.734574Z","iopub.execute_input":"2025-02-24T21:17:31.765696Z","iopub.status.idle":"2025-02-24T21:17:32.186280Z","shell.execute_reply":"2025-02-24T21:17:32.184435Z"},"jupyter":{"outputs_hidden":false}}
#Load the dataset
stroke_data <- read.csv("/kaggle/input/stroke-risk-prediction-dataset/stroke_risk_dataset.csv")

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:32.189084Z","iopub.execute_input":"2025-02-24T21:17:32.190630Z","iopub.status.idle":"2025-02-24T21:17:32.330273Z","shell.execute_reply":"2025-02-24T21:17:32.328498Z"},"jupyter":{"outputs_hidden":false}}
#clean up column names
stroke_data <- clean_names(stroke_data)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Data Exploration

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:32.333299Z","iopub.execute_input":"2025-02-24T21:17:32.334839Z","iopub.status.idle":"2025-02-24T21:17:32.352557Z","shell.execute_reply":"2025-02-24T21:17:32.350743Z"},"jupyter":{"outputs_hidden":false}}
#Check the dimensions (rows, columns)
dim(stroke_data)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:32.355302Z","iopub.execute_input":"2025-02-24T21:17:32.356779Z","iopub.status.idle":"2025-02-24T21:17:32.372367Z","shell.execute_reply":"2025-02-24T21:17:32.370632Z"},"jupyter":{"outputs_hidden":false}}
#View column names
names(stroke_data)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:32.374955Z","iopub.execute_input":"2025-02-24T21:17:32.376442Z","iopub.status.idle":"2025-02-24T21:17:32.454188Z","shell.execute_reply":"2025-02-24T21:17:32.452398Z"},"jupyter":{"outputs_hidden":false}}
#Check summary statistics
summary(stroke_data)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:32.456837Z","iopub.execute_input":"2025-02-24T21:17:32.458283Z","iopub.status.idle":"2025-02-24T21:17:32.494253Z","shell.execute_reply":"2025-02-24T21:17:32.492431Z"},"jupyter":{"outputs_hidden":false}}
#View  first few rows
head(stroke_data)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:32.496844Z","iopub.execute_input":"2025-02-24T21:17:32.498295Z","iopub.status.idle":"2025-02-24T21:17:32.525644Z","shell.execute_reply":"2025-02-24T21:17:32.523764Z"},"jupyter":{"outputs_hidden":false}}
#Check data types
str(stroke_data)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:32.528294Z","iopub.execute_input":"2025-02-24T21:17:32.529791Z","iopub.status.idle":"2025-02-24T21:17:32.563787Z","shell.execute_reply":"2025-02-24T21:17:32.562068Z"},"jupyter":{"outputs_hidden":false}}
#Check for missing values
stroke_data %>% summarize(total_missing = sum(is.na(.)))

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:32.566560Z","iopub.execute_input":"2025-02-24T21:17:32.568044Z","iopub.status.idle":"2025-02-24T21:17:37.126992Z","shell.execute_reply":"2025-02-24T21:17:37.125011Z"},"jupyter":{"outputs_hidden":false}}
# Convert the dataset from wide to long format
data_long <- stroke_data %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

# Plot distributions for all variables
ggplot(data_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  facet_wrap(~ variable, scales = "free") +
  labs(title = "Distributions of Variables in the Stroke Dataset")

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:37.129633Z","iopub.execute_input":"2025-02-24T21:17:37.131713Z","iopub.status.idle":"2025-02-24T21:17:37.188571Z","shell.execute_reply":"2025-02-24T21:17:37.186572Z"},"jupyter":{"outputs_hidden":false}}
cor_matrix <- cor(stroke_data)

head(cor_matrix)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:37.191371Z","iopub.execute_input":"2025-02-24T21:17:37.193175Z","iopub.status.idle":"2025-02-24T21:17:37.365418Z","shell.execute_reply":"2025-02-24T21:17:37.363361Z"},"jupyter":{"outputs_hidden":false}}
# Visualize the correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", addCoef.col = "black",
         tl.cex = 0.7, number.cex = 0.7)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Building the model

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:37.368212Z","iopub.execute_input":"2025-02-24T21:17:37.369950Z","iopub.status.idle":"2025-02-24T21:17:37.528720Z","shell.execute_reply":"2025-02-24T21:17:37.526918Z"},"jupyter":{"outputs_hidden":false}}
#drop target variable we are not intrested in from data set
stroke_data1 <- stroke_data %>% select(- stroke_risk)

stroke_data1

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:37.531497Z","iopub.execute_input":"2025-02-24T21:17:37.533054Z","iopub.status.idle":"2025-02-24T21:17:37.588434Z","shell.execute_reply":"2025-02-24T21:17:37.586734Z"},"jupyter":{"outputs_hidden":false}}
#Splitting the data (70% for training and 30% for testing)

set.seed(123)  # for reproducibility
#shuffling the data
train_index <- sample(seq_len(nrow(stroke_data1)), size = 0.7 * nrow(stroke_data1))
#split into train and test sets 
train_data <- stroke_data1[train_index, ]
test_data  <- stroke_data1[-train_index, ]

nrow(train_data)
nrow(test_data)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Training Models

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:37.590995Z","iopub.execute_input":"2025-02-24T21:17:37.592396Z","iopub.status.idle":"2025-02-24T21:17:39.379205Z","shell.execute_reply":"2025-02-24T21:17:39.374757Z"},"jupyter":{"outputs_hidden":false}}
#Train logistic regression model
model_logit <- glm(at_risk_binary ~ ., data = train_data, family = "binomial")
summary(model_logit)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:39.385797Z","iopub.execute_input":"2025-02-24T21:17:39.389588Z","iopub.status.idle":"2025-02-24T21:17:40.197829Z","shell.execute_reply":"2025-02-24T21:17:40.194832Z"},"jupyter":{"outputs_hidden":false}}
# Predict probabilities on test data
pred_prob_logit <- predict(model_logit, newdata = test_data, type = "response")
pred_prob_logit #shows probability of each person suffering stroke

# Convert probabilities to binary predictions
pred_class_logit <- ifelse(pred_prob_logit > 0.5, 1, 0) #1=yes, 0=no
pred_class_logit

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:40.202096Z","iopub.execute_input":"2025-02-24T21:17:40.204439Z","iopub.status.idle":"2025-02-24T21:17:42.309754Z","shell.execute_reply":"2025-02-24T21:17:42.307881Z"},"jupyter":{"outputs_hidden":false}}
# Train decision tree model
model_tree <- rpart(at_risk_binary ~ ., data = train_data, method = "class")
plot(model_tree); text(model_tree)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:42.312432Z","iopub.execute_input":"2025-02-24T21:17:42.314300Z","iopub.status.idle":"2025-02-24T21:17:42.507376Z","shell.execute_reply":"2025-02-24T21:17:42.505508Z"},"jupyter":{"outputs_hidden":false}}
# Make predictions on the test set
pred_class_tree <- predict(model_tree, newdata = test_data, type = "class")
pred_class_tree

# Predict class probabilities; type="class" gives direct class predictions

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:42.510103Z","iopub.execute_input":"2025-02-24T21:17:42.511551Z","iopub.status.idle":"2025-02-24T21:17:42.546442Z","shell.execute_reply":"2025-02-24T21:17:42.544602Z"},"jupyter":{"outputs_hidden":false}}
# Convert predictions and actuals to factors
pred_logit_factor <- factor(pred_class_logit, levels = c(0,1)) #logit
pred_tree_factor <- factor(pred_class_tree, levels = c(0,1)) #decison tree
actual_factor <- factor(test_data$at_risk_binary, levels = c(0,1))

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:42.549154Z","iopub.execute_input":"2025-02-24T21:17:42.550574Z","iopub.status.idle":"2025-02-24T21:17:42.625839Z","shell.execute_reply":"2025-02-24T21:17:42.624072Z"},"jupyter":{"outputs_hidden":false}}
# Logistic Regression Confusion Matrix
cm_logit <- confusionMatrix(data = pred_logit_factor, reference = actual_factor, positive = '1')
print(cm_logit)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# The logit model appears to have achieved 100% accuracy, sensitivity and specificity. That means that the model correctly classified ALL those who at risk of stroke and those who are not. This is concerning to me, because an 100% performance is quite rare in real-world scenarios. So this raises a red flag. But I will be running the decision tree model to see if I encounter similar issues.

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:42.628663Z","iopub.execute_input":"2025-02-24T21:17:42.630134Z","iopub.status.idle":"2025-02-24T21:17:42.651857Z","shell.execute_reply":"2025-02-24T21:17:42.650076Z"},"jupyter":{"outputs_hidden":false}}
# Decision Tree Confusion Matrix
cm_tree <- confusionMatrix(data = pred_tree_factor, reference = actual_factor, positive = '1')
print(cm_tree)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# Here, we can see that this model's performance is no longer perfect (compared to the previous result). It correctly classifies about 80.9% of cases overall (accuracy), correctly identifies 86% of those who are at risk of stroke (sensitivity) and 72% of those not at risk (specificity). Overall, the model does fairly well at catching positive and negative cases because high rates of false positives and false negatives could be problematic, posing medical risks and increased costs.
# 
# Since the results of the decison tree are more realistic, I reviewed my processes by ensuring that no features used in my model were similar to the target variable (I had already dropped the second stroke variable) to ensure there was no data leakage. Also, cross-checked my data splitting process to make sure the train and test datasets were properly split without any overlap. And everything looks okay. 
# 
# What else should I do? If you know what the issue could be or any area I could improve, please let me know! I appreciate your comments.
# 
# Anyway, I decided to cross-validate the training data and try other models, maybe this could provide more insights.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Cross Validation

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:42.654532Z","iopub.execute_input":"2025-02-24T21:17:42.656078Z","iopub.status.idle":"2025-02-24T21:17:42.676623Z","shell.execute_reply":"2025-02-24T21:17:42.674704Z"},"jupyter":{"outputs_hidden":false}}
#set target variable as factor
train_data$at_risk_binary <- factor(train_data$at_risk_binary, levels = c(0,1), labels = c('N','Y'))
test_data$at_risk_binary <- factor(test_data$at_risk_binary, levels = c(0,1), labels = c('N','Y'))

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:42.679766Z","iopub.execute_input":"2025-02-24T21:17:42.681319Z","iopub.status.idle":"2025-02-24T21:17:42.697345Z","shell.execute_reply":"2025-02-24T21:17:42.695471Z"},"jupyter":{"outputs_hidden":false}}
# 5-fold cross-validation
set.seed(123)  # for reproducibility
cv_control <- trainControl(
  method = "cv",       # cross-validation
  number = 5,          # number of folds
  classProbs = TRUE,   # if you want to compute class probabilities
  summaryFunction = twoClassSummary, # for extended metrics (AUC, etc.)
  savePredictions = TRUE
)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:42.700075Z","iopub.execute_input":"2025-02-24T21:17:42.701577Z","iopub.status.idle":"2025-02-24T21:17:51.028348Z","shell.execute_reply":"2025-02-24T21:17:51.026533Z"},"jupyter":{"outputs_hidden":false}}
#Cross validation on decision tree
set.seed(123)
model_tree_cv <- train(
  at_risk_binary  ~ .,
  data = train_data,
  method = "rpart",
  trControl = cv_control,
  metric = "ROC"
)

model_tree_cv

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:51.031003Z","iopub.execute_input":"2025-02-24T21:17:51.032466Z","iopub.status.idle":"2025-02-24T21:17:51.149513Z","shell.execute_reply":"2025-02-24T21:17:51.147624Z"},"jupyter":{"outputs_hidden":false}}
#make predictions
pred_dtree_cv <- predict(model_tree_cv, new_data = test_data, type = 'raw')
pred_dtree_cv

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:51.152237Z","iopub.execute_input":"2025-02-24T21:17:51.153662Z","iopub.status.idle":"2025-02-24T21:17:51.179919Z","shell.execute_reply":"2025-02-24T21:17:51.177832Z"},"jupyter":{"outputs_hidden":false}}
cm_dtree <- confusionMatrix(data = pred_dtree_cv, reference = train_data$at_risk_binary, positive = 'Y')

cm_dtree

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# After cross-validation, the decision tree gives the same result This shows that the model is consistent.

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:17:51.182719Z","iopub.execute_input":"2025-02-24T21:17:51.184246Z","iopub.status.idle":"2025-02-24T21:19:18.747807Z","shell.execute_reply":"2025-02-24T21:19:18.745963Z"},"jupyter":{"outputs_hidden":false}}
#Generalized Logistic regression
set.seed(123)
model_glmnet <- train(
  at_risk_binary ~ ., 
  data = train_data,
  method = "glmnet",
  trControl = cv_control,
  metric = "ROC"  # or "Accuracy"
)

model_glmnet

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:19:18.750456Z","iopub.execute_input":"2025-02-24T21:19:18.751886Z","iopub.status.idle":"2025-02-24T21:19:18.852514Z","shell.execute_reply":"2025-02-24T21:19:18.850704Z"},"jupyter":{"outputs_hidden":false}}
pred_glmnet <- predict(model_glmnet, new_data = train_data, type = 'raw')

pred_glmnet

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:19:18.855223Z","iopub.execute_input":"2025-02-24T21:19:18.856640Z","iopub.status.idle":"2025-02-24T21:19:18.882058Z","shell.execute_reply":"2025-02-24T21:19:18.880184Z"},"jupyter":{"outputs_hidden":false}}
cm_glmnet <- confusionMatrix(data = pred_glmnet, reference = train_data$at_risk_binary, positive = 'Y')

cm_glmnet

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# At this point, do I conclude that the logistic regression model is an incredibly powerful model in this scenario? This is the regularized logistic regression and it correctly identifies 99% of cases with 99% of those not at risk of stroke and 100% of those at risk.

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:19:18.884748Z","iopub.execute_input":"2025-02-24T21:19:18.886248Z","iopub.status.idle":"2025-02-24T21:23:59.964675Z","shell.execute_reply":"2025-02-24T21:23:59.962858Z"},"jupyter":{"outputs_hidden":false}}
# Gradient Boosting model
set.seed(123)
model_xgb <- train(
 at_risk_binary ~ ., 
  data = train_data,
  method = "xgbTree",
  trControl = cv_control,
  metric = "ROC"
)

model_xgb

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:23:59.967332Z","iopub.execute_input":"2025-02-24T21:23:59.968705Z","iopub.status.idle":"2025-02-24T21:24:00.052701Z","shell.execute_reply":"2025-02-24T21:24:00.050913Z"},"jupyter":{"outputs_hidden":false}}
pred_xgb <- predict(model_xgb, newdata = test_data, type = 'raw')

pred_xgb

# %% [code] {"execution":{"iopub.status.busy":"2025-02-24T21:24:00.055407Z","iopub.execute_input":"2025-02-24T21:24:00.056846Z","iopub.status.idle":"2025-02-24T21:24:00.080160Z","shell.execute_reply":"2025-02-24T21:24:00.078345Z"},"jupyter":{"outputs_hidden":false}}
cm_xgb <- confusionMatrix(data = pred_xgb, reference = test_data$at_risk_binary, positive = 'Y')

cm_xgb

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# The xgboost results are similar to that of the glmnet model.
# 
# Overall, glmnet has the highest accuracy numerically, making it the 'best'. But both xgboost and glmnet have perfect sensitivity and very high specificity suggesting that they are almost never wrong.