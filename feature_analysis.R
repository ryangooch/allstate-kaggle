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