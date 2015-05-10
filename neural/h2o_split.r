#install.packages("h2o")
library(h2o)
localH2O <- h2o.init(nthread=6,Xmx="10g") # allocate memory

# Load data to cluster
train.hex <- as.h2o(localH2O,train1)
test.hex <- as.h2o(localH2O,valid[,-1])

# Select responce and predictors
predictors <- 2:(ncol(train.hex)-1)
response <- 1

model1 <- h2o.deeplearning(x=predictors,
                          y=response,
                          data=train.hex,
                          classification=T,
                          activation="TanhWithDropout",
                          hidden=c(1012, 512, 321),
                          hidden_dropout_ratio=c(0.5,0.5,0.5),
                          input_dropout_ratio=0.05,
                          epochs=50,
                          l1=1e-5,
                          l2=1e-5,
                          rho=0.99,
                          epsilon=1e-8,
                          train_samples_per_iteration=500,
                          max_w2=10,
                          seed=1)
						  
# Predict with validation set and calculate LogLoss

# model1

tulos <- as.data.frame(h2o.predict(model1,test.hex))

# katsotaan:

vertaile <- tulos[, -1]

LogLoss(oikeat_valid, vertaile) # 0.7313258 asetuksella hidden = c(312,121)

# 0.7367676 hidden=c(512,321)

# h2o.shutdown(localH2O)