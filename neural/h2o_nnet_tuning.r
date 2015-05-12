#### TUNING AND MODEL SELECTION ####

grid_search <- h2o.deeplearning(x=predictors, y=response,
		data=train0.hex, 
		hidden=c(276,184,81),
		activation="Tanh",
		classification=T,
		  hidden_dropout_ratio=c(0,0,0),
		  input_dropout_ratio = 0,
		  epochs=10,
		  l1=0,
		  l2=0,
		  rho=0.99,
		  epsilon=1e-10,
		  holdout_fraction=0.1,
		  train_samples_per_iteration = 2000,
		  max_w2=10,
		  balance_classes = T,
		  rate_decay = 0.05,
		seed=c(1,2,3,4))

# epocs = 20 ja c(276,184,81) 1h per model
# epocs = 20 ja c(276,184,81) 15min per model (?)
# fast_mode = false?

# Also test 2-layer models?
# nfolds = 5

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

LogLoss(oikeat_valid, bench_model_predicted1) # n. 0.57
LogLoss(oikeat_valid, bench_model_predicted2)
LogLoss(oikeat_valid, bench_model_predicted3)
LogLoss(oikeat_valid, avg_model) # 0.51 (best avg.) -> 0.55

#### Utility (Average) ####
models_to_get <- 2
comb_model <- as.data.frame(h2o.predict(best_model1,test.hex))[,-1]
for (i in 2:models_to_get) {
	add_model <- as.data.frame(h2o.predict(grid_search@model[[i]],test.hex))[,-1]
	comb_model <- comb_model + add_model
}
comb_model <- comb_model*(1/models_to_get)
round(LogLoss(oikeat_valid, comb_model),3)

##################
#### STACKING ####
##################

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
		hidden=list(c(67,67,67),c(128,67,37),c(276,271,123),c(276,184,81)),
		activation=c("Tanh", "RectifierWithDropout"),
		classification=T,
		  hidden_dropout_ratio=c(0,0,0),
		  input_dropout_ratio = 0,
		  epochs=1.5,
		  l1=0,
		  l2=0,
		  rho=0.99,
		  epsilon=1e-8,
		  holdout_fraction=0.1,
		  train_samples_per_iteration = 2000,
		  max_w2=10,
		  balance_classes = T,
		  rate_decay = 0.05,
		seed=2)

#### TEST (METADATA) ####		
best_ensemble <- grid_ensemble@model[[1]]
best_ensemble
best_ensemble@model$params$hidden

stack_prob <- as.data.frame(h2o.predict(best_ensemble,test.ensemble))[,-1]
round(LogLoss(oikeat_valid, stack_prob),3) # 0.601

# sampleakin voi muuttaa:
# validation=test.hex,