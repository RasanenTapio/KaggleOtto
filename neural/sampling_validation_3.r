# Neural Netwokrs# datat
trainfull <- read.csv("C:/ddata/datat/train.csv", header = TRUE)
testfull <- read.csv("C:/ddata/datat/test.csv", header = TRUE)

# Tulosten arviointia varten
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}

library(nnet)
library(randomForest)
library(caret)
library(deepnet)

for(i in 1:9) {
levels(trainfull$target)[i] <- i
}
trainfull$target <- as.numeric(trainfull$target)

trainfull <- trainfull[,-1]

nimet <- c(names(trainfull)[94], names(trainfull)[-94])
trainfull <- cbind(trainfull$target, log10(trainfull[,-94]+1)) # logaritmi muunnos
names(trainfull) <- nimet

nimet <- names(testfull)
testfull <- cbind(testfull$id, log10(testfull[,-1]+1)) # logaritmi muunnos
names(testfull) <- nimet

### jaetaan osiin
apu <- split(trainfull, trainfull$target)

# Jaetaan testausvaiheessa 4 yhtäsuureen osaa, joista 2 ensimmäistä train ja kolmas valid

# Initial:

	# Nollataan
	
	train_base <- trainfull[0,]
	train1 <- train_base
	train2 <- train_base
	train3 <- train_base

	valid <- train_base
	nimet <- names(trainfull)

# Ensin otos 10 %
	# Tasainen otos jokaista luokkaa
	for (i in 1:9 ){
		ot <- as.data.frame(apu[i])
		names(ot) <- nimet

		samp <- c(sample(1:dim(ot)[1], (0.1*dim(ot)[1])))
		
		train_base <- rbind(train_base, ot[-samp,])
		valid <- rbind(valid, ot[samp,])
	}

# Tarkastus
dim(trainfull) - dim(train_base) - dim(valid)

# jaetaan jäävä osuus vielä 3 osaan
apu <- split(train_base, train_base$target)
	
	# Tasainen otos jokaista luokkaa
	for (i in 1:9 ){
		ot <- as.data.frame(apu[i])
		names(ot) <- nimet

		samp <- c(sample(1:dim(ot)[1], dim(ot)[1]))
		sampsplit <- split(samp, 1:3)
		samp1 <- sampsplit[[1]]
		samp2 <- sampsplit[[2]]
		samp3 <- sampsplit[[3]]
		
		train1 <- rbind(train1, ot[samp1,])
		train2 <- rbind(train2, ot[samp2,])
		train3 <- rbind(train3, ot[samp3,])
	}

	names(train1) <- nimet
	names(train2) <- nimet
	names(train3) <- nimet

	
	dim(train_base) - dim(train1) - dim(train2) - dim(train3)
	
	# Oikeiden matriisi
	
	oikeat_valid <- matrix(0, dim(valid)[1], 9)

	for(i in 1:dim(valid)[1]){
		for(j in 1:9) {
			if (valid$target[i] == j) oikeat_valid[i,j] <- 1
		}
	}

# LOOP
hidden_nodes <- 61
maxIteraatiot <- 100
myDecay <- 0.1

# Tulostaulu
tarkkuus <- matrix(0, 3, 4)

## Testisetin muoto

### Muodostetaan oikeat 1--9

for (k in 1:3) {

	testset <- get(paste("train", k, sep = ""))

	oikeat <- matrix(0, dim(testset)[1], 9)

	for(i in 1:dim(testset)[1]){
		for(j in 1:9) {
			if (testset$target[i] == j) oikeat[i,j] <- 1
		}
	}
	
	assign(paste("oikeat", k, sep = ""), oikeat)
}

message(paste("Alkoi", Sys.time()))
alku <- Sys.time()
for(k in 2:3) {
	# Loopin alku: otetaan otos
	
	trainset <- get(paste("train", k, sep = ""))
	
	# Mallinnus ja mallin testaus
	
	nnettimalli <- nnet(as.factor(target) ~ ., data = trainset, size = hidden_nodes,
		maxit = maxIteraatiot, MaxNWts = 8500, decay = myDecay)
	
	#forestmalli <- randomForest(as.factor(target) ~ .,
	#	data = trainset, ntree = puut)
	
	assign(paste("malli", k, sep = ""), nnettimalli)
	
	tarkkuus[k,1] <- paste("malli", k, sep = "")
	for (i in 1:3){
		if (i != k) {
		
			testset <- get(paste("train", i, sep = ""))
			oikeat <- get(paste("oikeat", i, sep = ""))
			ennuste <- predict(nnettimalli, testset, type = "raw")

			logloss <- LogLoss(oikeat, ennuste)
			tarkkuus[k,i + 1] <- round(logloss, 4)
		}
		else {
			tarkkuus[k,i + 1] <- NA
		}
	}

	#tarkkuusrf[k,1] <- paste("mallirf", k, sep = "")
	#tarkkuusrf[k,2] <- loglossrf
	message(paste(" ", tarkkuus[k,], sep=" "))
}
message(paste("Loppui", Sys.time()))
loppu <- Sys.time()
kesti <- loppu - alku
message(kesti);

########## MALLIT JA PAINOT ########## 


########## ENSEMBLE 1: STACK ########## 

train <- rbind(train1, train2, train3)

ennuste1 <- predict(malli1, train, type = "raw")
ennuste2 <- predict(malli2, train, type = "raw")
ennuste3 <- predict(malli3, train, type = "raw")

#model_rf1 <- train(as.factor(target)~., 
 #              data=train,tuneLength = 9, maxit = 125)

# testataan max:

train_stack <- data.frame(target = train$target,
	C1 = ennuste1, C2 = ennuste2, C3 = ennuste3)

# Muodostetaan malli stack
stack_nnet <- nnet(as.factor(target) ~ ., data = train_stack, size = 37, maxit = 100,
	decay = 0.5, MaxNWts = 2500)

train_dat <- as.matrix(train_stack[,-1])
train_res <- as.matrix(train_stack[,1])
	
nn <- nn.train(x = train_dat, y = train_res, hidden = c(5))
	
# Muodostetaan validoinnista ennusteet luokille:
ennuste1 <- predict(malli1, valid, type = "raw")
ennuste2 <- predict(malli2, valid, type = "raw")
ennuste3 <- predict(malli3, valid, type = "raw")

# ennuste comb:

ennuste_comb <- ennuste1*0.34 + ennuste2*0.33 + ennuste3*0.33

valid_stack <- data.frame(target = valid$target,
	C1 = ennuste1, C2 = ennuste2, C3 = ennuste3)

stack_prob <- predict(stack_nnet, valid_stack, type = "raw")

test_dat <- train_dat

pred_nn <- nn.predict(nn, test_dat) ### LISÄYS

head(cbind(valid$target,stack_prob))

ennuste_rf <- predict(model_rf1, valid_stack, type = "prob")

LogLoss(oikeat_valid, stack_prob)
LogLoss(oikeat_valid, ennuste_comb)
LogLoss(oikeat_valid, ennuste_rf)

##########  TEST SET AND SUBMISSION ########## 

ennuste1s <- predict(malli1, testfull, type = "raw")
ennuste2s <- predict(malli2, testfull, type = "raw")
ennuste3s <- predict(malli3, testfull, type = "raw")
ennuste4s <- predict(malli4, testfull, type = "raw")

test_stack <- data.frame(C1 = ennuste1s, C2 = ennuste2s, C3 = ennuste3s, C4 = ennuste4s)
	
ennuste_s <- predict(stack_nnet, test_stack, type = "raw") 

### Tallennetaan tulokset
uusin_tulos <- "stack_ensemble_4m_test2"
kansio <-  "C:/ddata/results/nn_"
palautus <- ennuste_s

tulosnimi <- paste(kansio, uusin_tulos, ".csv", sep = "")

comb_ennuste <- as.matrix(palautus)
comb_ennuste <- cbind(as.character(testfull$id), comb_ennuste) ## huom
# ensimmäinen sarake as string?

colnames(comb_ennuste) <- c("id","Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7",
	"Class_8","Class_9")

write.table(comb_ennuste, file = tulosnimi, row.names = FALSE,
	quote = FALSE, col.names = TRUE, sep=",")
	
round(head(palautus),2)

############### HYVIEN MALLIEN TALLENNUS ###################

# Hyvät mallit voidaan tallentaa mallit kansioon:

# save(m1, file = "C:/ddata/mallit/my_model1.rda")
save(malli2, file = "C:/ddata/mallit/malli053a.rda")
save(malli4, file = "C:/ddata/mallit/malli053b.rda")
save(malli1, file = "C:/ddata/mallit/malli053c.rda")