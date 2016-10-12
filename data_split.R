# head(train,5)
# 
# # learn things
# summary(train)
# 
# # check out the distribution of target variable (loss)
# hist(train$loss,plot = TRUE,)
# ?hist
# 
# # subsample a test set, will be used later
# train_sub <- train[sample(nrow(train), floor(0.25 * nrow(train))), ]
# summary(train_sub)
# 
# # see if it is satisfactory
# head(train_sub[order(-train_sub$loss),],5)
# 
# library(corrplot)
# corrplot(train,method = 'circle')
# 
# as.numeric(train_sub$cont4)
# # Applying 
# df <- apply(train_sub,2,as.numeric)
# 
# df <- sapply( train_sub, function(x) if("factor" %in% class(x) ) { 
#   as.numeric(x)
# } else {x } )
# library(plyr)
# df <- colwise(as.numeric)(train_sub)
# 
# # Now that we have it down, let's write out the train/holdout sets for this analysis

# read in data
train <- read.csv('data/train.csv')
test <- read.csv('data/test.csv')

train_rows = nrow(train)

# Since the factorization COULD be different between test and train, we should
# concatenate the train and test sets together in a data frame, convert to
# numeric, then separate.

test$loss <- NA # since there is no loss in the test, encode as NA
train_and_test <- rbind(train,test) # concatenate

# Need colwise from plyr
library(plyr)
train_and_test <- colwise(as.numeric)(train_and_test)

# now split train and test back up and remove loss column from test
train <- train_and_test[1:train_rows,]
test <- train_and_test[-nrow(train_and_test),]

smp_size <- floor(0.75 * nrow(train))

set.seed(2727)
train_ind <- sample(seq_len(nrow(train)), size = smp_size)

train_sub <- train[train_ind,]
holdout <- train[-train_ind,]

write.csv(train_sub,file = 'data/train_sub.csv') # subset for training algorithms
write.csv(holdout,file = 'data/holdout.csv') # holdout set for testing, test.csv for predictions
