library(caret);library(randomForest)
training<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
nsv<-nearZeroVar(training,saveMetrics = TRUE)
training<-training[,nsv$nzv==FALSE]
training<-training[,-(1:6)]
sums<-colSums(is.na(training))
nacols<-as.vector(sums>2)
training<-training[,nacols==FALSE]
set.seed(135)
training.rf<-randomForest(classe~.,data=training)
pred<-predict(training.rf,testing,"vote")
print(pred)

set.seed(12321)
folds<-createFolds(y=training$classe,k=10,list=TRUE,returnTrain=TRUE)
acc<-numeric(10)
for (i in 1:10){
        cv.train<-training[folds[[i]],]
        cv.test<-training[-folds[[i]],]
        cvrf<-randomForest(classe~.,data=cv.train)
        cvpred<-predict(cvrf,cv.test)
        acc[i]<-confusionMatrix(cvpred,cv.test$classe)$overall[1]
        print(paste("Accuracy for fold", i, ":", acc[i]))
        rm(cvrf)
}
print(mean(acc))
