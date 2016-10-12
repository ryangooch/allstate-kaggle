# This is an attempt at feature engineering in R. I have worked in R in 
# class settings and in tutorials, but never have used it as a primary
# tool for data science. I want to use this as an opportunity to learn
# where R may be better than python, where they can work together, and
# what the drawbacks are.
#
# I'm using a few scripts from the kernels for the Kaggle competition,
# all credit be unto Laurae and dmi3kno, etc etc.
#
# By Ryan Gooch, Oct 2016

setwd('~/Documents/kaggle/2016/allstate-kaggle/')
# From dmi3kno
library(data.table)
library(gridExtra)
library(corrplot)
library(GGally)
library(ggplot2)
library(e1071)

# from Laurae
library(Matrix)
library(xgboost)
library(DT)
library(plotluck) 
library(FSelector)

# from my own work
library(plyr)
library(caret)

dt_train <- read.csv("data/train.csv")
dt_test <- read.csv("data/test.csv")

names(dt_train)

# We can generate a holdout set from train to treat it like a test set
smp_size <- floor(0.75 * nrow(dt_train))

set.seed(2727)
train_ind <- sample(seq_len(nrow(dt_train)), size = smp_size)

train_sub <- dt_train[train_ind,]
holdout <- dt_train[-train_ind,]

dt_train <- train_sub

# at this point, dt_train is a subset of the given labeled instances, holdout is
# a validation set, also a subset of the given labeled instances, and dt_test is
# the original test set

# Pretty print from Laurae
pprint <- function(data) {
  cat(pprint_helper(data), sep = "\n")
}

pprint_helper <- function(data) {
  out <- paste(names(data), collapse = " | ")
  out <- c(out, paste(rep("---", ncol(data)), collapse = " | "))
  invisible(apply(data, 1, function(x) {
    out <<- c(out, paste(x, collapse = " | "))
  }))
  return(out)
}

pprint(dt_train[1:25, ])
head(dt_train,25)
# pprint must help the .Rmd files, head and rstudio printing 
# looks way better here.

#Missing values
colSums(sapply(dt_train, is.na))

# Check for duplicated rows.
cat("The number of duplicated rows are", nrow(dt_train) - 
      nrow(unique(dt_train)))

hist(log(dt_train$loss))
# this shows the distribution for log loss, pmuch more informative than non 
# log values

# Do correlation analysis, make cat feats numeric? idk if that makes a ton of 
# sense but we'll see. ok that didn't work with fread, it's fine with read.csv
dt_train_num <- colwise(as.numeric)(dt_train)
holdout_num <- colwise(as.numeric)(holdout)
dt_test_num <- colwise(as.numeric)(dt_test)

correlations <- cor(dt_train_num)
corrplot(correlations, method="square", order="hclust")

num_var <- names(dt_train)[which(sapply(dt_train, is.numeric))]
correlations <- cor(dt_train[num_var])
corrplot(correlations, method="square", order="hclust")
# definitely hard to read that one, but there are clearly correlated features

### Feature engineering
# now for the meat. Taking notes from Applied Predictive Modeling

# Look at ratio of second-most frequent categorical value to most frequent 
# in each variable. If that is a small number, consider dropping due to 
# imbalance. Related: Near-zero variance for numericals

# Collinearity: As above, remove feats with high inter-feature correlation

# Algorithm in Section 3.5, p.47, to remove collinear features

# Check for skewness
skewValues <- apply(dt_train_num, 2, skewness)
head(skewValues)

# BoxCox Transformation. Not sure how much sense this makes with factors
# of few levels. Cat78 has a very high skewness. Let's look at that
skewValues[79]
cat78bc <- BoxCoxTrans(dt_train_num$cat78)
cat78bc
head(dt_train_num$cat78)
predict(cat78bc,head(dt_train_num$cat78))

# can use preProcess to perform box cox transform on all features, then center
# and scale result. Might need to only do this for the higher skewness feats?
trans <- preProcess(dt_train_num,
                    method = c('BoxCox', 'center', 'scale'))
trans
# trans
# Created from 141238 samples and 132 variables
# 
# Pre-processing:
#   - Box-Cox transformation (59)
# - centered (132)
# - ignored (0)
# - scaled (132)
# 
# Lambda estimates for Box-Cox transformation:
#   Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# -2.00000 -0.65000  0.20000  0.04068  0.75000  2.00000 

# Apply the above transformations
transformed <- predict(trans, dt_train_num)

# transform holdout set as well
transformed_holdout <- predict(trans, holdout_num)

# CAREFUL: The loss column was transformed as well. This needs to be untouched
transformed$loss <- dt_train_num$loss
transformed_holdout$loss <- holdout_num$loss

# look for near-zero variance predictors
zero_var_variables <- nearZeroVar(dt_train_num) 

# Get correlation matrix for all variables
correlations <- cor(dt_train_num)
highCorr <- findCorrelation(correlations, cutoff = 0.80)
# make sure the loss column isn't one that gets removed, but high correlated
# loss to a predictor would be a good thing!
highCorr

vars_to_remove <- unique(c(highCorr,zero_var_variables))

# filter out near zero variance and highly correlated values from transformed
transformed_train_vars_removed <- transformed[, -vars_to_remove]
transformed_holdout_vars_removed <- transformed[, -vars_to_remove]

### Time to get crazy! xgboost time
target <- dt_train$loss
# transformed_train_vars_removed <- transformed_train_vars_removed[, -ncol(transformed_train_vars_removed)]
# transformed_holdout_vars_removed <- transformed_holdout_vars_removed[, -ncol(transformed_holdout_vars_removed)]
# data <- rbind(transformed_train_vars_removed, transformed_holdout_vars_removed)
dt_train <- dt_train[, -ncol(dt_train)]
holdout <- holdout[, -ncol(holdout)]
data <- rbind(dt_train, holdout)
data <- data[, -1] # remove ids
gc(verbose = FALSE)
data_sparse <- sparse.model.matrix(~.-1, data = as.data.frame(data))
cat("Data size: ", data_sparse@Dim[1], " x ", data_sparse@Dim[2], "  \n", sep = "")
gc(verbose = FALSE)
dtrain <- xgb.DMatrix(data = data_sparse[1:nrow(transformed_train_vars_removed), ], label = target) # Create design matrix without intercept
gc(verbose = FALSE)
dtest <- xgb.DMatrix(data = data_sparse[(nrow(transformed_train_vars_removed)+1):nrow(data), ]) # Create design matrix without intercept


gc(verbose = FALSE)
set.seed(27272436)
temp_model <- xgb.cv(data = dtrain,
                     nthread = 8,
                     nfold = 4,
                     #nrounds = 2, # quick test
                     nrounds = 1000000,
                     max_depth = 6,
                     eta = 0.0404096, # Santander overfitting magic number X2
                     subsample = 0.70,
                     colsample_bytree = 0.70,
                     booster = "gbtree",
                     metrics = c("mae"),
                     maximize = FALSE,
                     early_stopping_rounds = 25,
                     objective = "reg:linear",
                     print_every_n = 10,
                     verbose = TRUE)

gc(verbose = FALSE)
set.seed(27272436)
temp_model <- xgb.train(data = dtrain,
                           nthread = 8,
                           #nrounds = 2, # quick test
                           nrounds = floor(temp_model$best_iteration * 1.25),
                           max_depth = 6,
                           eta = 0.0404096, # Santander overfitting magic number X2
                           subsample = 0.70,
                           colsample_bytree = 0.70,
                           booster = "gbtree",
                           eval_metric = "mae",
                           maximize = FALSE,
                           objective = "reg:linear",
                           print_every_n = 10,
                           verbose = TRUE,
                           watchlist = list(train = transformed_train_vars_removed))
