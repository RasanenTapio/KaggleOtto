###############
# Model selection and tuning
# Change parameters in grid_search to tune model
###############

#### CREATE CLUSTER ####

# Working directory
setwd("c:/ddata/datat")

### Parameters ###

transform_data <- 2 # 1 logaritmic, 2 sqrt

###### LIBRARIES ######
library(h2o)
localH2O <- h2o.init(nthread=4,Xmx="10g") # allocate more memory

###### READ DATA ######
trainfull <- read.csv("C:/ddata/datat/train.csv", header = TRUE)
testfull <- read.csv("C:/ddata/datat/test.csv", header = TRUE)

###### Utility functions ######
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}

###### DATA TRANSFORMATION ######
for(i in 1:9) {
levels(trainfull$target)[i] <- i
}
trainfull$target <- as.numeric(trainfull$target)

trainfull <- trainfull[,-1]

###### Logaritmic transformation ######
if (transform_data == 1) {
	nimet <- c(names(trainfull)[94], names(trainfull)[-94])
	trainfull <- cbind(trainfull$target, log10(trainfull[,-94]+1)) # logaritmi
	names(trainfull) <- nimet

	nimet <- names(testfull)
	testfull <- cbind(testfull$id, log10(testfull[,-1]+1)) # logaritmi
	names(testfull) <- nimet
	print("log10")
}

###### Square root transformation ######
if (transform_data == 2) {
	nimet <- c(names(trainfull)[94], names(trainfull)[-94])
	trainfull <- cbind(trainfull$target, sqrt(trainfull[,-94]+(3/8))) # logaritmi
	names(trainfull) <- nimet

	nimet <- names(testfull)
	testfull <- cbind(testfull$id, sqrt(testfull[,-1]+(3/8))) # logaritmi
	names(testfull) <- nimet
	print("sqrt+3/4")
}

#### If full data to train:
trainx.hex <- as.h2o(localH2O,trainfull)
#### TUNING AND MODEL SELECTION ####

# Select response and predictors
predictors <- 2:(ncol(trainx.hex))
response <- 1

#mallit <- list(c(897, 565, 343), c(1011,674,449),c(997,665,443)) # (b)
mallit <- c(276,184,81)

# 10 with validation set (best fit to validation data)
# 10 without validation set (best fit to training data)

grid_search <- h2o.deeplearning(x=predictors, y=response,
		data=trainx.hex,
		hidden=mallit,
		activation="Tanh",
		classification=T,
		  hidden_dropout_ratio= c(0,0,0),
		  input_dropout_ratio = 0,
		  epochs=15,
		  l1=0,
		  l2=0,
		  rho=0.99,
		  epsilon=1e-10,
		  train_samples_per_iteration = 2000,
		  max_w2=10,
		  balance_classes = T,
		  rate_decay = 0.05,
		seed=c(17,47,7,9,13,1,2,3))

#### MODEL SELECTION ####
best_model1 <- grid_search@model[[1]]
best_model2 <- grid_search@model[[2]]
best_model3 <- grid_search@model[[3]]
best_model1
best_params <- best_model1@model$params
best_params$train_samples_per_iteration
best_params$hidden
#  best_model1@model$params$l2; best_model2@model$params$l2; best_model3@model$params$l2
best_model1@model$params$hidden; best_model2@model$params$hidden; best_model3@model$params$hidden

bench_model_predicted1 <- as.data.frame(h2o.predict(best_model1,test.hex))[,-1]
bench_model_predicted2 <- as.data.frame(h2o.predict(best_model2,test.hex))[,-1]
bench_model_predicted3 <- as.data.frame(h2o.predict(best_model3,test.hex))[,-1]
avg_model <- (bench_model_predicted1 + bench_model_predicted2 + bench_model_predicted3)/3

round(LogLoss(oikeat_valid, bench_model_predicted1),3)
round(LogLoss(oikeat_valid, bench_model_predicted2),3)
round(LogLoss(oikeat_valid, bench_model_predicted3),3)
round(LogLoss(oikeat_valid, avg_model),3) # 0.51 (best avg.) -> 0.508 -> # 0.505

#### Utility (Average) ####
#### Search Grid ####
models_to_get <- 3
comb_model <- as.data.frame(h2o.predict(grid_search@model[[1]],test.hex))[,-1]
for (i in 2:models_to_get) {
	add_model <- as.data.frame(h2o.predict(grid_search@model[[i]],test.hex))[,-1]
	comb_model <- comb_model + add_model
}
comb_model <- comb_model*(1/models_to_get)
round(LogLoss(oikeat_valid, comb_model),3)

###########################
#### STACKING ENSEBMLE ####
###########################

#### GENERATE TRAINING DATA (METADATA) ####

pred.ensemble1 <- as.data.frame(h2o.predict(best_model1,train0.hex))[,-1]
pred.ensemble2 <- as.data.frame(h2o.predict(best_model2,train0.hex))[,-1]
pred.ensemble3 <- as.data.frame(h2o.predict(best_model3,train0.hex))[,-1]

train_stack <- cbind(apu_train0[,1], pred.ensemble1, pred.ensemble2, pred.ensemble3)

#### GENERATA TEST DATA (METADATA) ####

test.ensemble1 <- as.data.frame(h2o.predict(best_model1,test.hex))[,-1]
test.ensemble2 <- as.data.frame(h2o.predict(best_model2,test.hex))[,-1]
test.ensemble3 <- as.data.frame(h2o.predict(best_model3,test.hex))[,-1]

test_stack <- cbind(valid[,1], test.ensemble1, test.ensemble2, test.ensemble3)

names(test_stack) <- c("target", paste("C",1:27, sep = ""))
names(train_stack) <- c("target", paste("C",1:27, sep = ""))

train.ensemble <- as.h2o(localH2O,train_stack)
test.ensemble <- as.h2o(localH2O,test_stack)

#### TRAIN (METADATA) ####
predictors_e <- 2:28
response_e <- 1

# Ensemble nnet:
# Etsitään tähänkin tarkoitukseen paras malli gridillä?
grid_ensemble <- h2o.deeplearning(x=predictors_e, y=response_e,
		data=train.ensemble, 
		hidden=list(c(27,27,27),c(67,27,27), c(123, 67, 46)),
		activation="Tanh",
		classification=T,
		  hidden_dropout_ratio=c(0,0,0),
		  input_dropout_ratio = 0,
		  epochs=10,
		  l1=0,
		  l2=0,
		  rho=0.99,
		  epsilon=1e-8,
		  holdout_fraction=0.1,
		  train_samples_per_iteration = 2000,
		  max_w2=10,
		  balance_classes = T,
		  rate_decay = 0.05,
		seed=1)

#### TEST (METADATA) ####		
best_ensemble <- grid_ensemble@model[[1]]
best_ensemble
best_ensemble@model$params$hidden

stack_prob <- as.data.frame(h2o.predict(best_ensemble,test.ensemble))[,-1]
round(LogLoss(oikeat_valid, stack_prob),3) # 0.601 #0.82 #0.683

# SHUT DOWN
# Shut down cluster
# h2o.shutdown(localH2O)