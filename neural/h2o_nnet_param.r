### h2o 10 model train with nnet equivalent parameters

#### CREATE CLUSTER ####

# Working directory
setwd("c:/ddata/datat")

### Parameters

transform_data <- 2 # 1 logaritmic, 2 sqrt
###### LIBRARIES ######
library(nnet)
#library(randomForest)
#install.packages("h2o")
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

#### END OF EXPERIMENT 2 ####

### Split by target before sample ###
apu <- split(trainfull, trainfull$target)

# Initialize empty datasets
train1 <- trainfull[0,]
train2 <- train1
train3 <- train1
train4 <- train1
train5 <- train1
train6 <- train1
train7 <- train1
train8 <- train1
train9 <- train1

valid <- train1
nimet <- names(trainfull)

# Tasainen otos jokaista luokkaa
for (i in 1:9 ){
	ot <- as.data.frame(apu[i])
	names(ot) <- nimet

	samp <- c(sample(1:dim(ot)[1], dim(ot)[1]))
	sampsplit <- split(samp, 1:10)
	samp1 <- sampsplit[[1]]
	samp2 <- sampsplit[[2]]
	samp3 <- sampsplit[[3]]
	samp4 <- sampsplit[[4]]
	samp5 <- sampsplit[[5]]
	samp6 <- sampsplit[[6]]
	samp7 <- sampsplit[[7]]
	samp8 <- sampsplit[[8]]
	samp9 <- sampsplit[[9]]
	samp_valid <- sampsplit[[10]]
	
	train1 <- rbind(train1, ot[samp1,])
	train2 <- rbind(train2, ot[samp2,])
	train3 <- rbind(train3, ot[samp3,])
	train4 <- rbind(train4, ot[samp4,])
	train5 <- rbind(train5, ot[samp5,])
	train6 <- rbind(train6, ot[samp6,])
	train7 <- rbind(train7, ot[samp7,])
	train8 <- rbind(train8, ot[samp8,])
	train9 <- rbind(train9, ot[samp9,])
	valid <- rbind(valid, ot[samp_valid,])
}

names(train1) <- nimet
names(train2) <- nimet
names(train3) <- nimet
names(valid) <- nimet

dim(trainfull) - dim(train1) - dim(train2) - dim(train3) - dim(train4) - dim(train5) - dim(valid) - dim(train6) - dim(train7) - dim(train8) - dim(train9)

###### Correct results ######
oikeat_valid <- matrix(0, dim(valid)[1], 9)

for(i in 1:dim(valid)[1]){
	for(j in 1:9) {
		if (valid$target[i] == j) oikeat_valid[i,j] <- 1
	}
}

###### Use samples of 3/10 and 1/10 of test ######
apu_train1 <- rbind(train1, train2, train3)
apu_train2 <- rbind(train4, train5, train6)
apu_train3 <- rbind(train7, train8, train9)
apu_train0 <- rbind(apu_train1, apu_train2, apu_train3) # full data

# Load data to cluster (3/10 of data)
train0.hex <- as.h2o(localH2O,apu_train0) # 9/10
#train1.hex <- as.h2o(localH2O,apu_train1) # 3/10
#train2.hex <- as.h2o(localH2O,apu_train2) # 3/10
#train3.hex <- as.h2o(localH2O,apu_train3) # 3/10

test.hex <- as.h2o(localH2O,valid) # 1/10 for validating ensemble

#### MODEL ####

# Select response and predictors
predictors <- 2:(ncol(train0.hex))
response <- 1

# c(256, 121), hidden=c(666,444,222),

# Matrix for results
log_results <- matrix(0,20,2)

for (i in 1:20) {
	model_temp <- h2o.deeplearning(x=predictors,
                          y=response,
                          data=train0.hex,
                          classification=T,
                          activation="Tanh",
                          hidden=c(62,62,62),
                          hidden_dropout_ratio=c(0.1,0.1,0.1),
                          input_dropout_ratio=0.05,
                          epochs=20,
                          l1=1e-5,
                          l2=1e-5,
                          rho=0.99,
                          epsilon=1e-8,
                          train_samples_per_iteration=2000,
						  holdout_fraction=0.1,
                          max_w2=10,
						  balance_classes = T,
						  use_all_factor_levels = T,
						  override_with_best_model = T,
						  rate_decay =  0.5,
                          seed=i)

	assign(paste("model",i,sep=""),model_temp)
	
	tulos_temp <- as.data.frame(h2o.predict(model_temp,test.hex))[,-1]
	
	# Save model and logloss
	log_results[i,1] <- paste("model", i, sep="")
	log_results[i,2] <- LogLoss(oikeat_valid, tulos_temp)
	# Report results
	print(log_results[i,])
}

# Predict with validation set and calculate LogLoss
# setting: train_samples_per_iteration = -1 (all data) or -2 (auto tuning)
# holdout_fraction
# use_all_factor_levels

########### STACKING ENSEMBLE #############

best_models <- as.data.frame(log_results)
names(best_models) <- c("model", "logloss")
best_models <- best_models[order(best_models$logloss),]

# get x models
models_to_get <- 10

#### GET 20 best models
#comb_model <- as.data.frame(h2o.predict(model7,testfull.hex))[,-1]
comb_model <- comb_model*0
for (i in 1:models_to_get) {
	add_model <- as.data.frame(h2o.predict(get(paste("model", i, sep = "")),test0.hex))[,-1]
	comb_model <- comb_model + add_model
}
comb_model <- comb_model*(1/models_to_get)
LogLoss(oikeat_valid, comb_model) # 0.708555

#### GENERATE TRAINING DATA (METADATA)

pred.ensemble1 <- as.data.frame(h2o.predict(best_model1,train0.hex))[,-1]
pred.ensemble2 <- as.data.frame(h2o.predict(best_model2,train0.hex))[,-1]
pred.ensemble3 <- as.data.frame(h2o.predict(best_model3,train0.hex))[,-1]

train_stack <- cbind(apu_train0[,1], pred.ensemble1, pred.ensemble2, pred.ensemble3)

#### GENERATA TEST DATA (METADATA)

test.ensemble1 <- as.data.frame(h2o.predict(model1,test.hex))[,-1]
test.ensemble2 <- as.data.frame(h2o.predict(model2,test.hex))[,-1]
test.ensemble3 <- as.data.frame(h2o.predict(model3,test.hex))[,-1]

test_stack <- cbind(valid[,1], test.ensemble1, test.ensemble2, test.ensemble3)

names(test_stack) <- c("target", paste("C",1:27, sep = ""))
names(train_stack) <- c("target", paste("C",1:27, sep = ""))

# dim
dim(test_stack); dim(train_stack) # 1 responce, 27 variables (3x9)

#### TEST (Use Naive Bayes instead? or nnet package?)

train.ensemble <- as.h2o(localH2O,train_stack)
test.ensemble <- as.h2o(localH2O,test_stack)

predictors <- 2:(ncol(train.ensemble))
response <- 1

model_e <- h2o.deeplearning(x=predictors,
                          y=response,
                          data=train.ensemble,
                          classification=T,
                          activation="TanhWithDropout",
                          hidden=50,
                          input_dropout_ratio=0.05,
                          epochs=50,
                          l1=1e-5,
                          l2=1e-5,
                          rho=0.99,
                          epsilon=1e-8,
                          train_samples_per_iteration=4000,
                          max_w2=10,
                          seed=1)	

						  
tulos_ensemble <- as.data.frame(h2o.predict(model_e,test.ensemble))[,-1]

LogLoss(oikeat_valid, tulos_ensemble) # 0.911-- # 0.792

# Ensemble nnet:
library(nnet)
stack_nnet <- nnet(as.factor(target) ~ ., data = train_stack, size = 67, maxit = 100,
	decay = 0.5, MaxNWts = 10000)
stack_prob <- predict(stack_nnet, test.ensemble , type = "raw")
	
LogLoss(oikeat_valid, stack_prob) # 0.6741 ... better than 0.792
	
# Avg
tulos_avg <- test.ensemble1 * 0.34 + test.ensemble2 * 0.33 + test.ensemble3 * 0.33

LogLoss(oikeat_valid, tulos_avg) # 0.8795 # 0.752

######## SAVE RESULTS ##########
### Tallennetaan tulokset
uusin_tulos <- "ensemble_20m_test1"
kansio <-  "C:/ddata/results/nn_"
palautus <- comb_model

tulosnimi <- paste(kansio, uusin_tulos, ".csv", sep = "")

comb_ennuste <- as.matrix(palautus)
comb_ennuste <- cbind(as.character(testfull$id), comb_ennuste)

colnames(comb_ennuste) <- c("id","Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7",
	"Class_8","Class_9")

write.table(comb_ennuste, file = tulosnimi, row.names = FALSE,
	quote = FALSE, col.names = TRUE, sep=",")
	
# SHUT DOWN
# Shut down cluster
# h2o.shutdown(localH2O)