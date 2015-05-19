####################
## Use meta data of 3 models to create stacking ensemble
####################
#### CREATE CLUSTER ####

#install.packages("h2o")
library(h2o)
localH2O <- h2o.init(nthread=4,Xmx="10g") # allocate memory

#### CREATE SAMPLES #### 
trainfull <- read.csv("C:/ddata/datat/train.csv", header = TRUE)
testfull <- read.csv("C:/ddata/datat/test.csv", header = TRUE)

# Log Loss function
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}

for(i in 1:9) {
levels(trainfull$target)[i] <- i
}
trainfull$target <- as.numeric(trainfull$target)

trainfull <- trainfull[,-1]

### TRANSFORMATIONS

## Logaritmic transformation
nimet <- c(names(trainfull)[94], names(trainfull)[-94])
trainfull <- cbind(trainfull$target, log10(trainfull[,-94]+1))
names(trainfull) <- nimet

nimet <- names(testfull)
testfull <- cbind(testfull$id, log10(testfull[,-1]+1))
names(testfull) <- nimet

### Or Square root (variance-stabilizing)
nimet <- c(names(trainfull)[94], names(trainfull)[-94])
trainfull <- cbind(trainfull$target, sqrt(trainfull[,-94]+(3/8)))
names(trainfull) <- nimet

nimet <- names(testfull)
testfull <- cbind(testfull$id, sqrt(testfull[,-1]+(3/8)))
names(testfull) <- nimet

### Split by target
apu <- split(trainfull, trainfull$target)

	# Initial:
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
	
	# Create matrix for testing & validation
	
	oikeat_valid <- matrix(0, dim(valid)[1], 9)

	for(i in 1:dim(valid)[1]){
		for(j in 1:9) {
			if (valid$target[i] == j) oikeat_valid[i,j] <- 1
		}
	}

apu_train1 <- rbind(train1, train2, train3)
apu_train2 <- rbind(train4, train5, train6)
apu_train3 <- rbind(train7, train8, train9)
apu_train0 <- rbind(apu_train1, apu_train2, apu_train3)

# Load data to cluster (3/10 of data)
train0.hex <- as.h2o(localH2O,apu_train0) # 9/10
train1.hex <- as.h2o(localH2O,apu_train1) # 3/10
train2.hex <- as.h2o(localH2O,apu_train2) # 3/10
train3.hex <- as.h2o(localH2O,apu_train3) # 3/10

test.hex <- as.h2o(localH2O,valid[,-1])

#### MODEL ####

# Select responce and predictors
predictors <- 2:(ncol(train.hex))
response <- 1

# Fit 3 models and select 3 best:

model1 <- h2o.deeplearning(x=predictors,
                          y=response,
                          data=train1.hex,
                          classification=T,
                          activation="TanhWithDropout",
                          hidden=c(123, 81),
                          hidden_dropout_ratio=c(0.5, 0.5),
                          input_dropout_ratio=0.05,
                          epochs=50,
                          l1=1e-5,
                          l2=1e-5,
                          rho=0.99,
                          epsilon=1e-8,
                          train_samples_per_iteration=4000,
                          max_w2=10,
                          seed=1)

model2 <- h2o.deeplearning(x=predictors,
                          y=response,
                          data=train2.hex,
                          classification=T,
                          activation="TanhWithDropout",
                          hidden=c(256, 121),
                          hidden_dropout_ratio=c(0.5, 0.5),
                          input_dropout_ratio=0.05,
                          epochs=50,
                          l1=1e-5,
                          l2=1e-5,
                          rho=0.99,
                          epsilon=1e-8,
                          train_samples_per_iteration=2000,
                          max_w2=10,
                          seed=1)


model3 <- h2o.deeplearning(x=predictors,
                          y=response,
                          data=train3.hex,
                          classification=T,
                          activation="TanhWithDropout",
                          hidden=c(666,444,222),
                          hidden_dropout_ratio=c(0.5,0.5,0.5),
                          input_dropout_ratio=0.05,
                          epochs=50,
                          l1=1e-5,
                          l2=1e-5,
                          rho=0.99,
                          epsilon=1e-8,
                          train_samples_per_iteration=2000,
                          max_w2=10,
                          seed=1)

# Predict with validation set and calculate LogLoss

# model1-model3

tulos1 <- as.data.frame(h2o.predict(model1,test.hex))[,-1]
tulos2 <- as.data.frame(h2o.predict(model2,test.hex))[,-1]
tulos3 <- as.data.frame(h2o.predict(model3,test.hex))[,-1]

# prev # current # improved
LogLoss(oikeat_valid, tulos1) # 0.87 # 0.7378 # 0.84
LogLoss(oikeat_valid, tulos2) # 0.96 # 0.73
LogLoss(oikeat_valid, tulos3) # 0.89 # 1.23

########### STACKING ENSEMBLE #############

#### GENERATE TRAINING DATA (METADATA)

pred.ensemble1 <- as.data.frame(h2o.predict(model1,train0.hex))[,-1]
pred.ensemble2 <- as.data.frame(h2o.predict(model2,train0.hex))[,-1]
pred.ensemble3 <- as.data.frame(h2o.predict(model3,train0.hex))[,-1]

train_stack <- cbind(apu_train0[,1], pred.ensemble1, pred.ensemble2, pred.ensemble3)

#### GENERATA TEST DATA (METADATA)

test.ensemble1 <- as.data.frame(h2o.predict(model1,test.hex))[,-1]
test.ensemble2 <- as.data.frame(h2o.predict(model2,test.hex))[,-1]
test.ensemble3 <- as.data.frame(h2o.predict(model3,test.hex))[,-1]

test_stack <- cbind(valid[,1], test.ensemble1, test.ensemble2, test.ensemble3)

names(test_stack) <- c("target", paste("C",1:27, sep = ""))
names(train_stack) <- c("target", paste("C",1:27, sep = ""))

# Check dimensions
dim(test_stack); dim(train_stack) # 1 responce, 27 variables (3x9)

#### TEST ####

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
