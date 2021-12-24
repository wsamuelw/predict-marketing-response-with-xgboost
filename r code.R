# the end goal is to predict if customers accepted the last offer - response = 1 using a basic xgboost model with tidymodels

# load packages ----
library(tidymodels)
library(rpart.plot) # plot tree
library(vip) # feature importance
library(naniar) # visualize missingness
library(stringr) # format string
library(lubridate) # format datetime
library(corrplot) # visualize correlation

# import the data ----
# https://www.kaggle.com/jackdaoud/marketing-data/version/1
data <- read.csv('/Users/samuelwong/Desktop/Work/Data/Kaggle/Marketing Analytics/marketing_data.csv', stringsAsFactors = T, na.strings=c("","NA"))

# exmaine the data ----
glimpse(data)

# Rows: 2,240
# Columns: 28
# $ ID                  <int> 1826, 1, 10476, 1386, 5371, 7348, 4073, 1991, 4047, 9477, 2079, 5642, 10530, 2964, 10311, 837, 10521, 10175, 1473, 2795, 2285, 115, 10470, 4065, …
# $ Year_Birth          <int> 1970, 1961, 1958, 1967, 1989, 1958, 1954, 1967, 1954, 1954, 1947, 1979, 1959, 1981, 1969, 1977, 1977, 1958, 1960, 1958, 1954, 1966, 1979, 1976, 1…
# $ Education           <fct> Graduation, Graduation, Graduation, Graduation, Graduation, PhD, 2n Cycle, Graduation, PhD, PhD, 2n Cycle, Master, PhD, Graduation, Graduation, G…
# $ Marital_Status      <fct> Divorced, Single, Married, Together, Single, Single, Married, Together, Married, Married, Married, Together, Widow, Married, Married, Married, Ma…
# $ Income              <fct> "$84,835.00 ", "$57,091.00 ", "$67,267.00 ", "$32,474.00 ", "$21,474.00 ", "$71,691.00 ", "$63,564.00 ", "$44,931.00 ", "$65,324.00 ", "$65,324.0…
# $ Kidhome             <int> 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0…
# $ Teenhome            <int> 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1…
# $ Dt_Customer         <fct> 6/16/14, 6/15/14, 5/13/14, 5/11/14, 4/8/14, 3/17/14, 1/29/14, 1/18/14, 1/11/14, 1/11/14, 12/27/13, 12/9/13, 12/7/13, 10/16/13, 10/5/13, 9/11/13, …
# $ Recency             <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1…
# $ MntWines            <int> 189, 464, 134, 10, 6, 336, 769, 78, 384, 384, 450, 140, 431, 3, 16, 63, 63, 18, 53, 5, 213, 275, 40, 308, 266, 80, 454, 454, 27, 184, 155, 423, 7…
# $ MntFruits           <int> 104, 5, 11, 0, 16, 130, 80, 0, 0, 0, 26, 4, 82, 10, 4, 6, 6, 0, 1, 0, 9, 11, 2, 0, 21, 1, 0, 0, 0, 174, 7, 42, 0, 0, 12, 21, 22, 1, 16, 0, 2, 0, …
# $ MntMeatProducts     <int> 379, 64, 59, 1, 24, 411, 252, 11, 102, 102, 535, 61, 441, 8, 12, 57, 57, 2, 5, 3, 76, 68, 23, 73, 300, 37, 171, 171, 12, 256, 80, 706, 1, 21, 9, …
# $ MntFishProducts     <int> 111, 7, 15, 0, 11, 240, 15, 0, 21, 21, 73, 0, 80, 3, 2, 13, 13, 0, 2, 0, 4, 25, 0, 0, 65, 0, 8, 8, 0, 50, 13, 73, 0, 0, 0, 106, 138, 0, 43, 97, 6…
# $ MntSweetProducts    <int> 189, 0, 2, 0, 0, 32, 34, 0, 32, 32, 98, 13, 20, 16, 4, 13, 13, 0, 1, 0, 3, 7, 4, 0, 8, 1, 19, 19, 1, 30, 7, 197, 0, 0, 14, 20, 89, 3, 16, 172, 1,…
# $ MntGoldProds        <int> 218, 37, 30, 0, 34, 43, 65, 7, 5, 5, 26, 4, 102, 32, 321, 22, 22, 2, 10, 5, 30, 7, 23, 23, 44, 3, 32, 32, 5, 32, 10, 197, 0, 17, 7, 20, 29, 3, 16…
# $ NumDealsPurchases   <int> 1, 1, 1, 1, 2, 1, 1, 1, 3, 3, 1, 2, 1, 1, 0, 4, 4, 1, 2, 1, 3, 3, 2, 2, 4, 3, 12, 12, 2, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1, 2, 2, 3, 1, 3, 2, 1, 1, 2,…
# $ NumWebPurchases     <int> 4, 7, 3, 1, 3, 4, 10, 2, 6, 6, 5, 3, 3, 1, 25, 2, 2, 1, 2, 1, 5, 5, 2, 5, 8, 2, 9, 9, 2, 5, 5, 4, 1, 7, 2, 4, 2, 0, 2, 5, 2, 11, 8, 4, 1, 4, 1, 5…
# $ NumCatalogPurchases <int> 4, 3, 2, 0, 1, 7, 10, 1, 2, 2, 6, 1, 6, 1, 0, 1, 1, 0, 0, 0, 2, 1, 1, 1, 8, 1, 2, 2, 0, 4, 1, 8, 0, 1, 0, 3, 3, 0, 4, 5, 0, 9, 2, 0, 2, 5, 0, 4, …
# $ NumStorePurchases   <int> 6, 7, 5, 2, 2, 5, 7, 3, 9, 9, 10, 6, 6, 2, 0, 5, 5, 3, 3, 2, 5, 8, 3, 8, 6, 4, 8, 8, 3, 6, 5, 9, 2, 7, 3, 4, 13, 3, 9, 12, 3, 12, 5, 4, 2, 4, 3, …
# $ NumWebVisitsMonth   <int> 1, 5, 2, 7, 7, 2, 6, 5, 4, 4, 1, 4, 1, 6, 1, 4, 4, 4, 8, 7, 7, 5, 4, 7, 6, 7, 8, 8, 5, 2, 6, 2, 6, 8, 7, 1, 1, 4, 4, 3, 8, 6, 5, 8, 6, 3, 8, 4, 5…
# $ AcceptedCmp3        <int> 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1…
# $ AcceptedCmp4        <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…
# $ AcceptedCmp5        <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…
# $ AcceptedCmp1        <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0…
# $ AcceptedCmp2        <int> 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…
# $ Response            <int> 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1…
# $ Complain            <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…
# $ Country             <fct> SP, CA, US, AUS, SP, SP, GER, SP, US, IND, US, SP, IND, CA, SP, SP, SP, SP, CA, CA, SA, IND, GER, SP, IND, SP, SP, SP, SP, CA, SP, CA, AUS, IND, …

# visualize missingness
vis_miss(data)

# take a look at the customers with no income for any obvious pattern or any data quality issue
data %>% 
  filter(is.na(Income)) %>% 
  View()

# visualize correlation
select_if(data, is.numeric) %>% # select all the numeric variables
  cor() %>% 
  corrplot()

# feature engineering ----

# convert date format
data$Dt_Customer <-mdy(data$Dt_Customer)

# tenure *probably better to use the data extract date for this
data$Tenure <- Sys.Date() - data$Dt_Customer

# calculate the age using the current year
data$age <- year(Sys.Date()) - data$Year_Birth

# number of childern
data$num_childern <- data$Kidhome + data$Teenhome

# convert into factor for the model
data$Response <- as.factor(data$Response)

# convert Income into numeric
data$Income <- data$Income %>% 
  str_remove("[$]") %>% 
  str_remove("[,]") %>% 
  str_remove(".00") %>% 
  replace(is.na(.), 0) %>% 
  as.numeric()

# check imbalance
data %>% 
  group_by(Response) %>% 
  summarise(n = n(),
            total = nrow(.),
            pc = n / total)

# Response     n    total    pc
#        0  1906    2240  0.851
#        1   334    2240  0.149

# model training ----
# split data 
set.seed(222) 
data_split <- initial_split(data, prop = 0.8, strata = Response) # enforce similar distributions
data_split

# <Analysis/Assess/Total>
# <1791/449/2240>

train <- training(data_split); nrow(train) # 1792
test <- testing(data_split); nrow(test) # 448

# define the model *xgboost in this case
xgboost_spec <- boost_tree() %>% 
  set_mode("classification") %>% 
  set_engine("xgboost")

# Train the model on the training set
xgboost_model <- fit(xgboost_spec, Response ~. -ID, train)

# feature importance
vip(xgboost_model)

# making predictions ----
# predicting on new data
prediction_class <- predict(xgboost_model, new_data = test, type = "class")
prediction_prob <- predict(xgboost_model, new_data = test, type = "prob")
prediction_all <- cbind(test, prediction_class, prediction_prob)

# evaluation metrics ----
# confusion matric
conf_mat(data = prediction_all, estimate = .pred_class, truth = Response)

#           Truth
# Prediction   0   1
#           0 365  33
#           1  17  34

# accuracy 
accuracy(prediction_all, estimate = .pred_class, truth = Response) # 0.889

# precision
precision(prediction_all, Response, .pred_class) # 0.917

# recall
recall(prediction_all, Response, .pred_class) # 0.955

# f1 score
f_meas(prediction_all, Response, .pred_class) # 0.936

# log loss
mn_log_loss(prediction_all, Response, .pred_0) # 0.245

# calculate single-threhold sensitivity
sens(prediction_all, estimate = .pred_class, truth = Response) # 0.955

# calculate area under curve
roc_auc(prediction_all, estimate = .pred_0, truth = Response) # 0.920

# Calculate the ROC curve for all thresholds
# Plot the ROC curve
roc_curve(prediction_all, estimate = .pred_0, truth = Response) %>% 
  autoplot()

