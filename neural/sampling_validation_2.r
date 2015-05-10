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

for(i in 1:9) {
levels(trainfull$target)[i] <- i
}
trainfull$target <- as.numeric(trainfull$target)

trainfull <- trainfull[,-1]

#nimet <- c(names(trainfull)[94], names(trainfull)[-94])
#trainfull <- cbind(trainfull$target, log10(trainfull[,-94]+1)) # logaritmi
#names(trainfull) <- nimet

#nimet <- names(testfull)
#testfull <- cbind(testfull$id, log10(testfull[,-1]+1)) # logaritmi
#names(testfull) <- nimet

### jaetaan osiin
apu <- split(trainfull, trainfull$target)

# Jaetaan testausvaiheessa 4 yhtäsuureen osaa, joista 2 ensimmäistä train ja kolmas valid

# Initial:

	# Nollataan
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
	
	# Oikeiden matriisi
	
	oikeat_valid <- matrix(0, dim(valid)[1], 9)

	for(i in 1:dim(valid)[1]){
		for(j in 1:9) {
			if (valid$target[i] == j) oikeat_valid[i,j] <- 1
		}
	}

# LOOP
iteraatiot <- 5
otos <- 0.9
otos_replace <- FALSE 
hidden_nodes <- 62
maxIteraatiot <- 75
myDecay <- 0.1
puut = 251

# Tulostaulu
tarkkuus <- matrix(0, iteraatiot, 10)

## Testisetin muoto

### Muodostetaan oikeat 1--9

for (k in 1:9) {

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
for(k in 8:9) {
	# Loopin alku: otetaan otos
	
	trainset <- get(paste("train", k, sep = ""))
	
	# Mallinnus ja mallin testaus
	
	nnettimalli <- nnet(as.factor(target) ~ ., data = trainset, size = hidden_nodes,
		maxit = maxIteraatiot, MaxNWts = 8500, decay = myDecay)
	
	#forestmalli <- randomForest(as.factor(target) ~ .,
	#	data = trainset, ntree = puut)
	
	assign(paste("malli", k, sep = ""), nnettimalli)
	
	#tarkkuus[k,1] <- paste("malli", k, sep = "")
	#for (i in 1:9){
	#	if (i != k) {
	#	
	#		testset <- get(paste("train", i, sep = ""))
	#		oikeat <- get(paste("oikeat", i, sep = ""))
	#		ennuste <- predict(nnettimalli, testset, type = "raw")
#
	#		logloss <- LogLoss(oikeat, ennuste)
	#		tarkkuus[k,i + 1] <- round(logloss, 4)
	#	}
	#	else {
	#		tarkkuus[k,i + 1] <- NA
	#	}
	#}
	
	#message(paste("Malli", k, "tarkkuus", logloss, sep=" "))
}
message(paste("Loppui", Sys.time()))
loppu <- Sys.time()
kesti <- loppu - alku
message(kesti);

########## MALLIT JA PAINOT ########## 


########## ENSEMBLE 1: STACK ########## 

train <- rbind(train1, train2, train3, train4, train5, train6, train7, train8, train9)

ennuste1 <- predict(malli1, train, type = "raw")
ennuste2 <- predict(malli2, train, type = "raw")
ennuste3 <- predict(malli3, train, type = "raw")
ennuste4 <- predict(malli4, train, type = "raw")
ennuste5 <- predict(malli5, train, type = "raw")
ennuste6 <- predict(malli6, train, type = "raw")
ennuste7 <- predict(malli7, train, type = "raw")
ennuste8 <- predict(malli8, train, type = "raw")
ennuste9 <- predict(malli9, train, type = "raw")

# testataan max:

train_stack <- data.frame(target = train$target,
	C1 = ennuste1, C2 = ennuste2, C3 = ennuste3, C4 = ennuste4, C5 = ennuste5,
	C6 = ennuste6, C7 = ennuste7, C8 = ennuste8, C9 = ennuste9)

# Muodostetaan malli stack
stack_nnet <- nnet(as.factor(target) ~ ., data = train_stack, size = 67, maxit = 100,
	decay = 0.5, MaxNWts = 10000)
stack_nnet2 <- nnet(as.factor(target) ~ ., data = train_stack, size = 100, maxit = 100,
	decay = 0.5, MaxNWts = 10000)
	
stack_rf <- randomForest(as.factor(target) ~ ., data = train_stack, ntree = 501)

# Muodostetaan validoinnista ennusteet luokille:
ennuste1 <- predict(malli1, valid, type = "raw")
ennuste2 <- predict(malli2, valid, type = "raw")
ennuste3 <- predict(malli3, valid, type = "raw")
ennuste4 <- predict(malli4, valid, type = "raw")
ennuste5 <- predict(malli5, valid, type = "raw")
ennuste6 <- predict(malli6, valid, type = "raw")
ennuste7 <- predict(malli7, valid, type = "raw")
ennuste8 <- predict(malli8, valid, type = "raw")
ennuste9 <- predict(malli9, valid, type = "raw")

ennuste_comb <- ennuste1*1/9 + ennuste2*1/9 + ennuste3*1/9 + ennuste4*1/9 + ennuste5*1/9 +
	ennuste6*1/9 + ennuste7*1/9 + ennuste8*1/9 + ennuste9*1/9

valid_stack <- data.frame(target = valid$target,
	C1 = ennuste1, C2 = ennuste2, C3 = ennuste3, C4 = ennuste4, C5 = ennuste5,
	C6 = ennuste6, C7 = ennuste7, C8 = ennuste8, C9 = ennuste9)

stack_prob <- predict(stack_nnet, valid_stack, type = "raw")
stack_prob2 <- predict(stack_nnet2, valid_stack, type = "raw")
#stack_prob_rf <- predict(stack_rf, valid_stack, type = "prob")

#head(cbind(valid$target,stack_prob))

LogLoss(oikeat_valid, stack_prob) 
LogLoss(oikeat_valid, stack_prob2)
#LogLoss(oikeat_valid, ennuste_comb)
#LogLoss(oikeat_valid, stack_prob_rf)

# Testataan taas pitääkö paikkansa:
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