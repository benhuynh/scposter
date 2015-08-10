library(caret)
library(ggthemes)
library(reshape2)
data <- read.csv("aggregatedata.csv")
data[1] <- NULL
set.seed(12)
training <- createDataPartition(y=data$class,p=.60,list=FALSE)
trainset <- data[training,]
fulltestset <- data[-training,]
secondsplit <- createDataPartition(y=fulltestset$class,p=.5,list=FALSE)
testset <- fulltestset[secondsplit,]
validationset <- fulltestset[-secondsplit,]
if(file.exists('rf_model.RData')) {
  load('rf_model.RData')
} else {
  rf_model<-train(class ~ .,data=trainset,method="rf",
                  trControl=trainControl(method='cv',number=5,verboseIter=TRUE),importance=TRUE,do.trace=100,proximity=FALSE)
  save(rf_model,file = 'rf_model.RData')
}
if(file.exists('svmrbf.RData')) {
  load('svmrbf.RData')
} else {
  svmrbf <- train(class ~ .,data=trainset,method='svmRadial',
                  trControl=trainControl(method="cv",number=5,verboseIter=TRUE),do.trace=100)
  save(svmrbf,file = 'svmrbf.RData')
}
if(file.exists('svmlin.RData')) {
  load('svmlin.RData')
} else{
  svmlin <- train(class ~ .,data=trainset,method='svmLinear',
                  trControl=trainControl(method="cv",number=5,verboseIter=TRUE))
  save(svmlin,file = 'svmlin.RData')
}
if(file.exists('logit.RData')) {
  load('logit.RData')
} else {
  logitboost<-train(class ~ .,data=trainset,method='LogitBoost',
                    trControl=trainControl(method='cv',number=5,verboseIter=TRUE))
  save(logitboost,file = 'logit.RData')
}
if(file.exists('nn1.RData')) {
  load('nn1.RData')
} else {
  nn1 <- train(class ~ .,data=trainset,method='nnet',trControl=trainControl(method='cv',number=5,verboseIter=TRUE))
  save(nn1,file = 'nn1.RData')
}
if(file.exists('vimprbf.RData')) {
  load('vimprbf.RData')
} else {
  vimprbf <- varImp(svmrbf)
  save(vimprbf,file='vimprbf.RData')
}
bandpoints <- data.frame(x=c(21,51,85),y=c(vimprbf$importance[21,1],vimprbf$importance[51,1],vimprbf$importance[85,1])/100)

models <- list(rf_model,svmrbf,svmlin,logitboost,nn1)
predfunc <- function(model,set) {
  acc <- confusionMatrix(predict(model,set),set$class)$overall[c('Accuracy','AccuracyLower','AccuracyUpper')]
  return(acc)
}
acclist <- t(as.data.frame(lapply(models,predfunc,testset)))

obtestset <- testset
obtestset$ob <- obtestset$X21/obtestset$X51
obtestset$mw <- obtestset$X85/obtestset$X51
obtestset$pred1[obtestset$ob > 3] <- 'Water'
obtestset$pred1[obtestset$ob <= 3] <- 'NotWater'
obtestset$pred2[obtestset$ob > 2] <- 'Water'
obtestset$pred2[obtestset$ob <= 2] <- 'NotWater'
obtestset$pred3[obtestset$mw < .625] <- 'Water'
obtestset$pred3[obtestset$mw >= .625] <- 'NotWater'
obpred1 <- confusionMatrix(obtestset$pred1,obtestset$class)$overall
obpred1 <- t(obpred1[c('Accuracy','AccuracyLower','AccuracyUpper')])
obpred2 <- confusionMatrix(obtestset$pred2,obtestset$class)$overall
obpred2 <- t(obpred2[c('Accuracy','AccuracyLower','AccuracyUpper')])
obpred3 <- confusionMatrix(obtestset$pred3,obtestset$class)$overall
obpred3 <- t(obpred3[c('Accuracy','AccuracyLower','AccuracyUpper')])
obtestset$pred1[obtestset$pred1 == 'Water'] <- 1
obtestset$pred1[obtestset$pred1 == 'NotWater'] <- 0
obtestset$pred2[obtestset$pred2 == 'Water'] <- 1
obtestset$pred2[obtestset$pred2 == 'NotWater'] <- 0
obtestset$pred3[obtestset$pred3 == 'Water'] <- 1
obtestset$pred3[obtestset$pred3 == 'NotWater'] <- 0
if(file.exists('stack.RData')) {
  load('stack.RData')
} else {
  stack <- train(class ~ pred1+pred2+pred3,data=obtestset,method='logreg',trControl=trainControl(method='cv',number=5,verboseIter=TRUE))
  save(stack,file='stack.RData')
}
stackpred <- predfunc(stack,obtestset)
acclist <- rbind(obpred3,obpred1,obpred2,stackpred,acclist) 
rownames(acclist) <- c('MUDDYWATER','CLEARWATER','MURKYWATER','OB Ensemble','Random Forest','SVMRBF','SVM','Logit','Neural Net')

xnames <- factor(rownames(acclist),levels=rownames(acclist))

bandroc <- qplot(seq(1:234),vimprbf$importance[,1]/100,geom='line',
                 main='ROC Performance of Spectral Bands in Water Detection'
                 ,ylab='AUROC Curve',xlab='Spectral Band Number') + geom_point(aes(x=bandpoints$x,y=bandpoints$y),col='red',size=8) +theme_bw()+ theme(axis.title.x=element_text(size=20),axis.title.y=element_text(size=20),
        axis.text.x=element_text(size=12),axis.text.y=element_text(size=12),
        title=element_text(size=20))

vimprf <- varImp(rf_model)
bandpointz <- data.frame(x=c(21,51,85),y=c(vimprf$importance[21,1],vimprf$importance[51,1],vimprf$importance[85,1])/100)
bandrf <- qplot(seq(1:234),vimprf$importance[,1]/100,geom='line',
                 main='Random Forest Variable Ranking of Spectral Bands'
                 ,ylab='Variable Importance',xlab='Band Number') + geom_point(aes(x=bandpointz$x,y=bandpointz$y),col='blue',size=5) +theme_bw()+ theme(axis.title.x=element_text(size=20),axis.title.y=element_text(size=20),
                                                                                                                                               axis.text.x=element_text(size=12),axis.text.y=element_text(size=12),
                                                                                                                                               title=element_text(size=20))


accplot <- ggplot(as.data.frame(acclist)) +
  geom_errorbar(aes(x=xnames,ymin=AccuracyLower,ymax=AccuracyUpper),width=.2) +
  geom_point(aes(x=xnames,y=Accuracy),size=5) + xlab('Models') + ggtitle('Performance by Model') +
  coord_flip() +theme_bw() + theme(axis.title.x=element_text(size=20),axis.title.y=element_text(size=20),
                                   axis.text.x=element_text(size=12),axis.text.y=element_text(size=12),
                                   title=element_text(size=25)) + geom_vline(xintercept=4.5,lty=2)

FourClassTrainingSet <- read.csv("~/conference_poster_1/FourClassTrainingSet.txt")
specdata <- rbind(colMeans(subset(FourClassTrainingSet,label==1))[1:234],colMeans(subset(FourClassTrainingSet,label==2))[1:234],
                  colMeans(subset(FourClassTrainingSet,label==3))[1:234],colMeans(subset(FourClassTrainingSet,label==4))[1:234])
colnames(specdata) <- seq(1:234)
specplot <- ggplot() + geom_line(aes(x=seq(1:234),y=specdata[1,],color='Clouds')) + geom_line(aes(x=seq(1:234),y=specdata[2,],col='Dry Land')) +
  geom_line(aes(x=seq(1:234),y=specdata[3,],color='Water')) + geom_line(aes(x=seq(1:234),y=specdata[4,],color='Vegetation')) +
  scale_color_manual(name='Type',values=c('red','#F0E442','green','blue')) + theme_bw() + labs(title='Training Set Spectra',y='Reflectance',x='Spectral Band Number') +
  theme(axis.title.x=element_text(size=20),axis.title.y=element_text(size=20),
        axis.text.x=element_text(size=12),axis.text.y=element_text(size=12),
        title=element_text(size=20),legend.text=element_text(size=12))




  
