library(dplyr)
library(corrplot)
library(keras)
library(randomForest)
library(randomForestExplainer)
library(e1071)
library(ggplot2)
library(tensorflow)
library(class)
library(caret)
library(gridExtra)
library(MASS)
plot <- corrplot(x)
fulldata <- read.csv('/Users/mileskee7/Desktop/College/Senior Year/Semester 1/W/course-paper-or-presentation-mileskee/data/fulldata.csv')
fulldata <- fulldata[,-1]
fulldata$fromp5 <- ifelse(fulldata$conf %in% c('P12','B10','SEC','B12','BE','ACC'),1,0)
fulldata$top5 <- ifelse(fulldata$Conference %in% c('P12','B10','SEC','B12','BE','ACC'),1,0)
fulldata$oimp <- ifelse(fulldata$ORTG > fulldata$ORtg,1,0)
fulldata$dimp <- ifelse(fulldata$DBPM > fulldata$dbpm,1,0)
boxplot(fulldata$ORtg~fulldata$oimp)
nrow(fulldata %>% filter(oimp==1)) / nrow(fulldata)
nrow(fulldata %>% filter(oimp==0)) / nrow(fulldata)
nrow(fulldata %>% filter(dimp==1)) / nrow(fulldata)
nrow(fulldata %>% filter(dimp==0)) / nrow(fulldata)
# Linear Regression -----------------------------------------------------------
###ORTG linear model
ortg <- fulldata %>% dplyr::select(Min_per,ORtg,usg,eFG,ORB_per,AST_per,TO_per,ftr,ORTG,fromp5,top5,oimp)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- ortg[traindex,]
label_train <- train$ORTG
train2 <- train %>% dplyr::select(-ORTG) %>% scale()
test <- ortg[-traindex,]
label_test <- test$ORTG
lin_ortg <- lm(ORTG~ORtg+eFG+ORB_per+AST_per+TO_per+ftr+as.factor(fromp5)+as.factor(top5)+Min_per+usg,data=train)
summary(lin_ortg)
coefs <- data.frame(summary(lin_ortg)$coefficients)
coefs$variable <- rownames(coefs)
coefs <- coefs[-1,]
coefs[7,5] <- 'fromp5'
coefs[8,5] <- 'top5'
plot1 <- ggplot(coefs,aes(x=variable,y=Estimate,fill=coefs$Pr...t..<.05)) + geom_bar(stat='identity') +
  geom_text(aes(label=round(coefs$Std..Error,4)),stat='identity',colour='black',size=2.5,angle=90) + labs(fill='Significant') +
  ggtitle('Offensive Linear Regression Coefficients') + scale_fill_manual(values=c("TRUE"='green',"FALSE"='red')) +
  theme(axis.text=element_text(size=8),axis.text.x = element_text(angle = 90, hjust = 1),text=element_text(size=8))
plot1
pdf("/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/olincoeffs.pdf",height=7.5,width=10)
print(plot1)
dev.off()
predictions <- predict(lin_ortg,test)
test <- cbind(test,predictions)
plot(abs(test_ortg$predictions-test_ortg$ORTG),ylab='Absolute Error',main='Test Set Error of Offensive Rating Linear Model')
mean(sqrt((test$predictions-test$ORTG)^2))
#DRTG linear model
drtg <- fulldata %>% dplyr::select(Min_per,DRB_per,blk_per,stl_per,dbpm,DBPM,fromp5,top5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- drtg[traindex,]
label_train_d <- train$DBPM
train2 <- train %>% dplyr::select(-DBPM) %>% scale()
test <- drtg[-traindex,]
label_test_d <- test$DBPM
lin_drtg <- lm(DBPM~dbpm+DRB_per+blk_per+stl_per+fromp5+top5+Min_per,data=train)
summary(lin_drtg)
coefs <- data.frame(summary(lin_drtg)$coefficients)
coefs$variable <- rownames(coefs)
coefs
coefs <- coefs[-1,]
pdf("/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/dlincoeffs.pdf",height=7.5,width=10)
plot2 <- ggplot(coefs,aes(x=variable,y=Estimate,fill=coefs$Pr...t..<.05)) + geom_bar(stat='identity') +
  geom_text(aes(label=round(coefs$Std..Error,4)),stat='identity',colour='black',size=2.5,angle=90) + labs(fill='Significant') +
  ggtitle('Defensive Linear Regression Coefficients') + scale_fill_manual(values=c("TRUE"='green',"FALSE"='red')) + 
  theme(axis.text=element_text(size=8),axis.text.x = element_text(angle = 90, hjust = 1),text=element_text(size=8))
print(plot2)
dev.off()
predictions <- predict(lin_drtg,test)
test <- cbind(test,predictions)
plot(test$predictions-test$DBPM,ylab='Absolute Error',main='Test Set Error of dbpm Linear Model')
mean(sqrt((test$predictions-test$dbpm)^2)) 
improved <- test %>% filter(DBPM-dbpm > 0)
nrow(improved %>% filter(predictions-dbpm > 0)) / nrow(improved)
deproved <- test %>% filter((DBPM-dbpm < 0))
nrow(deproved %>% filter(predictions-dbpm < 0)) / nrow(deproved)
(nrow(improved %>% filter(predictions-dbpm > 0)) + nrow(deproved %>% filter(predictions-dbpm < 0))) / (nrow(improved) + nrow(deproved))
# ML Numerical Prediction -------------------------------------------------
#ORTG ML model
ortg <- fulldata %>% dplyr::select(Min_per,ORtg,usg,eFG,ORB_per,AST_per,TO_per,ftr,ORTG,fromp5,top5,oimp)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- ortg[traindex,]
test <- ortg[-traindex,]
label_train <- train$ORTG
train2 <- train %>% dplyr::select(-ORTG) %>% scale()
label_test <- test$ORTG
test2 <- test %>% dplyr::select(-ORTG) %>% scale()
model <- keras_model_sequential() %>%
  layer_dense(units=11, activation='relu',input_shape=ncol(train2)) %>%
  layer_dense(units=1)
model %>% compile(
  loss='mse',
  optimizer=optimizer_rmsprop(),
  metrics=list('mean_absolute_error')
)
fit <- model %>% fit(
  train2,
  label_train,
  batch_size=32,
  epochs=100,
  validation_split=.1,
  verbose=1
)
save_model_weights_tf(model, './checkpoints/my_checkpoint1')
load_model_weights_tf(model, './checkpoints/my_checkpoint1')
plot(fit)
model %>% evaluate(test2,label_test)
prediction <- predict(model,test2)
plot(abs(prediction-label_test),ylab='Absolute Error',main='Test Set Error of ORtg Machine Learning Model')
mean(sqrt((prediction-label_test)^2))
#DRTG ML model
drtg <- fulldata %>% dplyr::select(Min_per,DRB_per,blk_per,stl_per,dbpm,DBPM,fromp5,top5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- drtg[traindex,]
label_train_d <- train$DBPM
train2 <- train %>% dplyr::select(-DBPM) %>% scale()
test <- drtg[-traindex,]
label_test_d <- test$DBPM
test2 <- test %>% dplyr::select(-DBPM) %>% scale()
model2 <- keras_model_sequential() %>%
  layer_dense(units=7,activation='relu', input_shape=ncol(train2)) %>%
  layer_dense(units=1)
model2 %>% compile(
  loss='mse',
  optimizer=optimizer_rmsprop(),
  metrics=list('mean_absolute_error')
)  
fit2 <- model2 %>% fit(
  train2,
  label_train_d,
  batch_size=32,
  epochs=100,
  validation_split=.1,
  verbose=1
)
save_model_weights_tf(model2, './checkpoints_def/my_checkpoint2')
load_model_weights_tf(model2, './checkpoints_def/my_checkpoint2')
plot(fit2)
model2 %>% evaluate(test2,label_test_d)
prediction <- predict(model2,test2)
plot(abs(prediction-label_test_d),ylab='Absolute Error',main='Test Set Error of ORtg Machine Learning Model')
mean(sqrt((prediction-label_test_d)^2))
# Random Forest -----------------------------------------------------------
#offensive
ortg <- fulldata %>% dplyr::select(Min_per,ORtg,usg,eFG,ORB_per,AST_per,TO_per,ftr,ORTG,fromp5,top5,oimp)
ortg$oimp <- as.factor(ortg$oimp)
ortg$top5 <- as.factor(ortg$top5)
ortg$fromp5 <- as.factor(ortg$fromp5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- ortg[traindex,]
test <- ortg[-traindex,]
label_test <- test$ORTG
rf_ortg <- randomForest(ORTG~ORtg+eFG+ORB_per+AST_per+TO_per+ftr+fromp5+top5+Min_per+usg,data=train,mtry=4,ntree=300,localImp=T)
x <- measure_importance(rf_ortg)
x
plot1 <- plot_multi_way_importance(x,size_measure = 'no_of_nodes',x_measure='mse_increase',y_measure='times_a_root')
plot1
dictions <- predict(rf_ortg,test)
mean(sqrt((dictions-label_test)^2))
#drtg
drtg <- fulldata %>% dplyr::select(Min_per,DRB_per,blk_per,stl_per,dbpm,DBPM,fromp5,top5,dimp)
drtg$dimp <- as.factor(drtg$dimp)
drtg$top5 <- as.factor(drtg$top5)
drtg$fromp5 <- as.factor(drtg$fromp5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- drtg[traindex,]
test <- drtg[-traindex,]
label_test <- test$DBPM
rf_drtg <- randomForest(DBPM~dbpm+DRB_per+blk_per+stl_per+fromp5+top5+Min_per,data=train,mtry=4,ntree=300,localImp=T)
x <- measure_importance(rf_drtg)
x
pdf("/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/rfreg.pdf",height=8.5,width=11)
plot2 <- plot_multi_way_importance(x,size_measure = 'no_of_nodes',x_measure='mse_increase',y_measure='times_a_root')
plot2
grid.arrange(plot1,plot2,ncol=2)
dev.off()
dictions <- predict(rf_drtg,test)
mean(sqrt((dictions-label_test)^2))

# SVM ---------------------------------------------------------------------
#offensive
ortg <- fulldata %>% dplyr::select(Min_per,ORtg,usg,eFG,ORB_per,AST_per,TO_per,ftr,ORTG,fromp5,top5,oimp)
ortg$oimp <- as.factor(ortg$oimp)
ortg$top5 <- as.factor(ortg$top5)
ortg$fromp5 <- as.factor(ortg$fromp5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- ortg[traindex,]
label_train <- train$ORTG
test <- ortg[-traindex,]
label_test <- test$ORTG
svm_o <- svm(ORTG~ORtg+eFG+ORB_per+AST_per+TO_per+ftr+fromp5+top5+Min_per+usg,data=train,kernel="linear",cost=5)
dictions <- predict(svm_o,test)
plot(test$ORtg,test$ORTG)
points(test$ORtg,dictions,col='red')
mean(sqrt((dictions-label_test)^2))
#defensive
drtg <- fulldata %>% dplyr::select(Min_per,DRB_per,blk_per,stl_per,dbpm,DBPM,fromp5,top5,dimp)
drtg$dimp <- as.factor(drtg$dimp)
drtg$top5 <- as.factor(drtg$top5)
drtg$fromp5 <- as.factor(drtg$fromp5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- drtg[traindex,]
test <- drtg[-traindex,]
label_test <- test$DBPM
svm_d <- svm(DBPM~dbpm+DRB_per+blk_per+stl_per+fromp5+top5+Min_per,data=train,kernel='linear',cost=5)
svm2 <- train(dimp~dbpm+DRB_per+blk_per+stl_per+fromp5+top5+Min_per,data=train, method = "svmLinear", trControl=trainControl(method = 'repeatedcv', 10, 20),  preProcess = c("center","scale"), tuneGrid = expand.grid(C = seq(1, 100, length = 20)))
control <- trainControl(method="cv",number=10) 
metric <- "Accuracy"
model <- train(DBPM~dbpm+DRB_per+blk_per+stl_per+fromp5+top5+Min_per, data = train, method = "svmLinear", 
               preProc = c("center","scale"),
               tuneGrid = expand.grid(C =1:20),
               metric=metric, trControl=control)
model
plot(model)
p <- predict(svm_d,test)
mean(sqrt((p-label_test)^2))
#Classification


#CLASSIFICATION

# Logistic Regression ------------------------------------------------------
##ORTG classification model
ortg <- fulldata %>% dplyr::select(Min_per,ORtg,usg,eFG,ORB_per,AST_per,TO_per,ftr,ORTG,fromp5,top5,oimp)
ortg$fromp5 <- as.factor(ortg$fromp5)
ortg$top5 <- as.factor(ortg$top5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- ortg[traindex,]
test <- ortg[-traindex,]
log_ortg <- glm(oimp~ORtg+eFG+ORB_per+AST_per+TO_per+ftr+fromp5+top5+Min_per+usg,data=train,family='binomial')
summary(log_ortg)
coefs <- data.frame(summary(log_ortg)$coefficients)
coefs$variable <- rownames(coefs)
coefs <- coefs[-1,]
coefs$Pr...z..
coefs[7,5] <- 'fromp5'
coefs[8,5] <- 'top5'
pdf("/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/ologcoeffs.pdf",height=8.5,width=11)
plot1 <- ggplot(coefs,aes(x=variable,y=Estimate,fill=factor(coefs$Pr...z..<.05))) + geom_bar(stat='identity') +
  geom_text(aes(label=round(coefs$Std..Error,4)),stat='identity',colour='black',size=2.5,angle=90) + labs(fill='Significant') +
  ggtitle('Offensive Logisitic Regression Coefficients') + scale_fill_manual(values=c("TRUE"='green',"FALSE"='red')) +
  theme(axis.text=element_text(size=8),axis.text.x = element_text(angle = 90, hjust = 1),text=element_text(size=8))
print(plot1)
dev.off()
predictions <- predict(log_ortg,test,type='response')
test_ortg <- cbind(test,predictions)
test_ortg$round_predict <- ifelse(test_ortg$predictions > .5,1,0)
mean((test_ortg$oimp-test_ortg$predictions)^2)
improved <- test_ortg %>% filter(oimp==1)
nrow(improved %>% filter(round_predict==1)) / nrow(improved)
deproved <- test_ortg %>% filter(oimp==0)
nrow(deproved %>% filter(round_predict==0)) / nrow(deproved)
(nrow(improved %>% filter(round_predict==1)) + nrow(deproved %>% filter(round_predict==0))) / (nrow(improved) + nrow(deproved))
##DRTG classification
drtg <- fulldata %>% dplyr::select(Min_per,DRB_per,blk_per,stl_per,dbpm,DBPM,fromp5,top5)
drtg$dimp <- ifelse(drtg$DBPM > drtg$dbpm,1,0)
drtg$top5 <- as.factor(drtg$top5)
drtg$fromp5 <- as.factor(drtg$fromp5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- drtg[traindex,]
test <- drtg[-traindex,]
log_drtg <- glm(dimp~dbpm+DRB_per+blk_per+stl_per+fromp5+top5+Min_per,data=train,family='binomial')
summary(log_drtg)
coefs <- data.frame(summary(log_drtg)$coefficients)
coefs$variable <- rownames(coefs)
coefs <- coefs[-1,]
coefs[5,5] <- 'fromp5'
coefs[6,5] <- 'top5'
pdf("/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/dlogcoeffs.pdf",height=8.5,width=11)
plot2 <- ggplot(coefs,aes(x=variable,y=Estimate,fill=factor(coefs$Pr...z..<.05))) + geom_bar(stat='identity') +
  geom_text(aes(label=round(coefs$Std..Error,4)),stat='identity',colour='black',size=2.5,angle=90) + labs(fill='Significant') +
  ggtitle('Defensive Logisitic Regression Coefficients') + scale_fill_manual(values=c("TRUE"='green',"FALSE"='red')) +
  theme(axis.text=element_text(size=8),axis.text.x = element_text(angle = 90, hjust = 1),text=element_text(size=8))
print(plot2)
dev.off()
predictions <- predict(log_drtg,test,type='response')
test_drtg <- cbind(test,predictions)
test_drtg$round_predict <- ifelse(test_drtg$predictions > .5,1,0)
mean((test_drtg$dimp-test_drtg$predictions)^2)
improved <- test_drtg %>% filter(dimp==1)
nrow(improved %>% filter(round_predict==1)) / nrow(improved)
deproved <- test_drtg %>% filter(dimp==0)
nrow(deproved %>% filter(round_predict==0)) / nrow(deproved)
(nrow(improved %>% filter(round_predict==1)) + nrow(deproved %>% filter(round_predict==0))) / (nrow(improved) + nrow(deproved))

# ML Classification -------------------------------------------------------
##Offensive classification
ortg <- fulldata %>% dplyr::select(Min_per,ORtg,usg,eFG,ORB_per,AST_per,TO_per,ftr,fromp5,top5,oimp)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- ortg[traindex,]
test <- ortg[-traindex,]
label_train <- train$oimp
label_train <- to_categorical(label_train,2)
train2 <- train %>% dplyr::select(-oimp) %>% scale()
label_test <- test$oimp
label_test <- to_categorical(label_test,2)
test2 <- test %>% dplyr::select(-oimp) %>% scale()
model_class <- keras_model_sequential() %>%
  layer_dense(units=11, activation='relu',input_shape=ncol(train2)) %>%
  layer_dense(units=2, activation='sigmoid')
model_class %>% compile(
  loss='binary_crossentropy',
  optimizer=optimizer_rmsprop(),
  metrics=list('accuracy')
)
fit_class <- model_class %>% fit(
  train2,
  label_train,
  batch_size=32,
  epochs=100,
  validation_split=.1,
  verbose=1
)
save_model_weights_tf(model_class, './checkpoints_off_class/my_checkpoint')
load_model_weights_tf(model_class, './checkpoints_off_class/my_checkpoint')
model_class %>% evaluate(test2,label_test)
prediction <- predict(model_class,test2)
prediction_prob <- prediction[,2]
mean(sqrt((prediction_prob-label_test[,2])^2))
test_ortg_nn <- cbind(test,prediction_prob)
test_ortg_nn$round_predict <- ifelse(test_ortg_nn$prediction_prob > .5, 1,0)
improved <- test_ortg_nn %>% filter(oimp==1)
nrow(improved %>% filter(round_predict==1)) / nrow(improved)
deproved <- test_ortg_nn %>% filter(oimp==0)
nrow(deproved %>% filter(round_predict==0)) / nrow(deproved)
(nrow(improved %>% filter(round_predict==1)) + nrow(deproved %>% filter(round_predict==0))) / (nrow(improved) + nrow(deproved))
##Defensive classificatoin
drtg <- fulldata %>% dplyr::select(Min_per,DRB_per,blk_per,stl_per,dbpm,DBPM,fromp5,top5)
drtg$dimp <- ifelse(drtg$DBPM > drtg$dbpm,1,0)
drtg <- drtg %>% dplyr::select(-DBPM)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- drtg[traindex,]
test <- drtg[-traindex,]
label_train <- train$dimp
label_train <- to_categorical(label_train,2)
train2 <- train %>% dplyr::select(-dimp) %>% scale()
label_test <- test$dimp
label_test <- to_categorical(label_test,2)
test2 <- test %>% dplyr::select(-dimp) %>% scale()
model_classd <- keras_model_sequential() %>%
  layer_dense(units=7, activation='relu',input_shape=ncol(train2)) %>%
  layer_dense(units=2, activation='sigmoid')
model_classd %>% compile(
  loss='binary_crossentropy',
  optimizer=optimizer_rmsprop(),
  metrics=list('accuracy')
)
fit_classd <- model_classd %>% fit(
  train2,
  label_train,
  batch_size=32,
  epochs=100,
  validation_split=.1,
  verbose=1
)
save_model_weights_tf(model_classd, './checkpoints_def_class/my_checkpoint')
load_model_weights_tf(model_classd, './checkpoints_def_class/my_checkpoint')
model_classd %>% evaluate(test2,label_test)
prediction <- predict(model_classd,test2)
prediction_prob <- prediction[,2]
mean(sqrt((prediction_prob-label_test[,2])^2))
test_drtg_nn <- cbind(test,prediction_prob)
test_drtg_nn$round_predict <- ifelse(test_drtg_nn$prediction_prob > .5, 1,0)
improved <- test_drtg_nn %>% filter(dimp==1)
nrow(improved %>% filter(round_predict==1)) / nrow(improved)
deproved <- test_drtg_nn %>% filter(dimp==0)
nrow(deproved %>% filter(round_predict==0)) / nrow(deproved)
(nrow(improved %>% filter(round_predict==1)) + nrow(deproved %>% filter(round_predict==0))) / (nrow(improved) + nrow(deproved))


# KNN ---------------------------------------------------------------------
#offensive
ortg <- fulldata %>% dplyr::select(Min_per,ORtg,usg,eFG,ORB_per,AST_per,TO_per,ftr,ORTG,fromp5,top5,oimp)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- ortg[traindex,]
label_train <- train$oimp
train2 <- train %>% dplyr::select(-c(ORTG,oimp)) %>% scale()
test <- ortg[-traindex,]
label_test <- test$oimp
test2 <- test %>% dplyr::select(-c(ORTG,oimp)) %>% scale()
ctrl <- trainControl(method='repeatedcv',repeats=3)
knncv <- train(as.factor(oimp)~Min_per+ORtg+usg+eFG+ORB_per+AST_per+TO_per+ftr+fromp5+top5, data=scale(train), trControl=ctrl,method='knn',tuneLength=20)
knncv
mod <- knn(
  train = train2, 
  test = test2,
  cl = label_train, 
  k=23
)
test <- cbind(test,mod)
pdf('/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/knn.off.model.pdf')
plot1 <- ggplot(data=test) + geom_point(aes(x=ORtg,y=usg,colour=mod)) + ggtitle('ORtg vs. usg', subtitle='KNN Model') + labs(colour='oimp')
plot2 <- ggplot(data=test) + geom_point(aes(x=ORB_per,y=usg,colour=mod)) + ggtitle('orb_per vs. usg',subtitle='KNN Model') + labs(colour='oimp')
plot1
plot2
grid.arrange(plot1,plot2,ncol=2)
dev.off()
plot(test$eFG,test$usg,col=mod) #yes
plot(test$Min_per,test$usg,col=mod) #no
plot(test$AST_per,test$usg,col=mod) #no
plot(test$TO_per,test$usg,col=mod)#kinda
plot(test$ftr,test$usg,col=mod) #no
plot(test$usg,test$fromp5,col=mod) #kinda
plot(test$usg,test$top5,col=mod) #no
cm <- table(label_test,mod)
cm
acc <- sum(diag(cm))/length(label_test)
correct_improve <- cm[2,2]/(cm[2,2]+cm[2,1])
correct_deprove <- cm[1,1]/(cm[1,1]+cm[1,2])
print(correct_improve)
print(correct_deprove)
print(acc)
#defensive
drtg <- fulldata %>% dplyr::select(Min_per,DRB_per,blk_per,stl_per,dbpm,DBPM,fromp5,top5,dimp)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- drtg[traindex,]
label_train <- train$dimp
train2 <- train %>% dplyr::select(-c(DBPM,dimp)) %>% scale()
test <- drtg[-traindex,]
label_test <- test$dimp
test2 <- test %>% dplyr::select(-c(DBPM,dimp)) %>% scale()
ctrl <- trainControl(method='repeatedcv',repeats=3)
knncv <- train(as.factor(dimp)~dbpm+Min_per+DRB_per+blk_per+stl_per+fromp5+top5, data=scale(train), trControl=ctrl,method='knn',tuneLength=20)
knncv
mod <- knn(
  train = train2, 
  test = test2,
  cl = label_train, 
  k=25
)
test <- cbind(test,mod)
plot1 <- ggplot(data=test) + geom_point(aes(x=dbpm,y=Min_per,colour=mod)) + ggtitle('dbpm vs min_per',subtitle='KNN Model') + labs(colour='dimp')
plot2 <- ggplot(data=test) + geom_point(aes(x=DRB_per,y=Min_per,colour=mod)) + ggtitle('drb_per vs min_per',subtitle='KNN Model') + labs(colour='dimp')
plot1
plot2
pdf('/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/knn.def.model.pdf')
grid.arrange(plot1,plot2,ncol=2)
dev.off()
plot(test$dbpm,test$Min_per,col=mod) #yes
plot(test$DRB_per,test$Min_per,col=mod) #no
plot(test$blk_per,test$Min_per,col=mod) #no
plot(test$stl_per,test$Min_per,col=mod) #no
plot(test$Min_per,test$fromp5,col=mod) #no
plot(test$Min_per,test$top5,col=mod) #yes
cm <- table(label_test,mod)
cm
acc <- sum(diag(cm))/length(label_test)
correct_improve <- cm[2,2]/(cm[2,2]+cm[2,1])
correct_deprove <- cm[1,1]/(cm[1,1]+cm[1,2])
print(correct_improve)
print(correct_deprove)
print(acc)
library(ggplot2)
knn_ortg <- cbind(test,mod)
ggplot(test) + 
  geom_point(aes(x=ORtg, y =usg, col=mod)) +
  geom_point(aes(x = ORtg, y = usg, col = as.factor(oimp)), 
             size = 4, shape = 1, data = train) + xlim(c(75,125))
# LDA ---------------------------------------------------------------------
#offensive
ortg <- fulldata %>% dplyr::select(Min_per,ORtg,usg,eFG,ORB_per,AST_per,TO_per,ftr,ORTG,fromp5,top5,oimp)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- ortg[traindex,]
label_train <- train$ORTG
train2 <- train %>% dplyr::select(-ORTG) %>% scale()
test <- ortg[-traindex,]
label_test <- test$oimp
lda_ortg <- lda(oimp~ORtg+eFG+ORB_per+AST_per+TO_per+ftr+fromp5+top5+Min_per+usg,data=train)
coeffs <- data.frame(lda_ortg$scaling)
coeffs$variable <- rownames(coeffs)
plot1 <- ggplot(data=coeffs,aes(x=variable,y=LD1)) + geom_bar(stat='identity') + ggtitle('Offensive Variable Importance',subtitle='LDA Model')+
  theme(axis.text=element_text(size=8),axis.text.x = element_text(angle = 90, hjust = 1))
plot1
p <- predict(lda_ortg,test)
predictions <- p$class
data <- data.frame(imp=test$oimp,lda=p$x)
plot(data)
cm <- table(label_test,predictions)
cm
acc <- sum(diag(cm))/length(label_test)
print(acc)
correct_improve <- cm[2,2]/(cm[2,2]+cm[2,1])
correct_deprove <- cm[1,1]/(cm[1,1]+cm[1,2])
print(correct_improve)
print(correct_deprove)
print(acc)
#defensive
drtg <- fulldata %>% dplyr::select(Min_per,DRB_per,blk_per,stl_per,dbpm,DBPM,fromp5,top5,dimp)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- drtg[traindex,]
label_train <- train$DBPM
train2 <- train %>% dplyr::select(-DBPM) %>% scale()
test <- drtg[-traindex,]
label_test <- test$dimp
lda_drtg <- lda(dimp~dbpm+DRB_per+blk_per+stl_per+fromp5+top5+Min_per,data=train)
lda_drtg
coeffs <- data.frame(lda_drtg$scaling)
coeffs$variable <- rownames(coeffs)
pdf('/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/ldacoeffs.pdf',height=8.5,width=11)
plot2 <- ggplot(data=coeffs,aes(x=variable,y=LD1)) + geom_bar(stat='identity') + ggtitle('Defensive Variable Importance',subtitle='LDA Model')+
  theme(axis.text=element_text(size=8),axis.text.x = element_text(angle = 90, hjust = 1))
plot2
grid.arrange(plot1,plot2,ncol=2)
dev.off()
p <- predict(lda_drtg,test)
predictions <- p$class
cm <- table(label_test,predictions)
cm
acc <- sum(diag(cm))/length(label_test)
print(acc)
correct_improve <- cm[2,2]/(cm[2,2]+cm[2,1])
correct_deprove <- cm[1,1]/(cm[1,1]+cm[1,2])
print(correct_improve)
print(correct_deprove)
print(acc)
# Random Forest -----------------------------------------------------------
#offensive
ortg <- fulldata %>% dplyr::select(Min_per,ORtg,usg,eFG,ORB_per,AST_per,TO_per,ftr,ORTG,fromp5,top5,oimp)
ortg$oimp <- as.factor(ortg$oimp)
ortg$top5 <- as.factor(ortg$top5)
ortg$fromp5 <- as.factor(ortg$fromp5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- ortg[traindex,]
label_train <- train$ORTG
train2 <- train %>% dplyr::select(-ORTG)
test <- ortg[-traindex,]
label_test <- test$oimp
rf_ortg <- randomForest(oimp~ORtg+eFG+ORB_per+AST_per+TO_per+ftr+fromp5+top5+Min_per+usg,data=train,mtry=2,ntree=300,localImp=T)
x <- measure_importance(rf_ortg)
x
pdf("/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/orfclass.pdf")
plot1 <- plot_multi_way_importance(x,size_measure = 'no_of_nodes',x_measure='accuracy_decrease',y_measure='times_a_root')
plot1
dev.off()
plot(rf_ortg)
p <- predict(rf_ortg,test)
cm <- table(label_test,p)
cm
acc <- sum(diag(cm))/length(label_test)
correct_improve <- cm[2,2]/(cm[2,2]+cm[2,1])
correct_deprove <- cm[1,1]/(cm[1,1]+cm[1,2])
print(correct_improve)
print(correct_deprove)
print(acc)
#defensive
drtg <- fulldata %>% dplyr::select(Min_per,DRB_per,blk_per,stl_per,dbpm,DBPM,fromp5,top5,dimp)
drtg$dimp <- as.factor(drtg$dimp)
drtg$top5 <- as.factor(drtg$top5)
drtg$fromp5 <- as.factor(drtg$fromp5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- drtg[traindex,]
label_train <- train$DBPM
train2 <- train %>% dplyr::select(-DBPM)
test <- drtg[-traindex,]
label_test <- test$dimp
rf_drtg <- randomForest(dimp~dbpm+DRB_per+blk_per+stl_per+fromp5+top5+Min_per,data=train,mtry=2,ntree=300,localImp=T)
x <- measure_importance(rf_drtg)
x
pdf("/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/rfclass.pdf",height=8.5,width=11)
plot2 <- plot_multi_way_importance(x,size_measure = 'no_of_nodes',x_measure='accuracy_decrease',y_measure='times_a_root')
plot2
grid.arrange(plot1,plot2,ncol=2)
dev.off()
p <- predict(rf_drtg,test)
cm <- table(label_test,p)
cm
acc <- sum(diag(cm))/length(label_test)
correct_improve <- cm[2,2]/(cm[2,2]+cm[2,1])
correct_deprove <- cm[1,1]/(cm[1,1]+cm[1,2])
print(correct_improve)
print(correct_deprove)
print(acc)
# SVM ---------------------------------------------------------------------
ortg <- fulldata %>% dplyr::select(Min_per,ORtg,usg,eFG,ORB_per,AST_per,TO_per,ftr,ORTG,fromp5,top5,oimp)
ortg$oimp <- as.factor(ortg$oimp)
ortg$top5 <- as.factor(ortg$top5)
ortg$fromp5 <- as.factor(ortg$fromp5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- ortg[traindex,]
label_train <- train$oimp
train2 <- train %>% dplyr::select(-ORTG)
test <- ortg[-traindex,]
label_test <- test$oimp
svm_o <- svm(oimp~ORtg+eFG+ORB_per+AST_per+TO_per+ftr+fromp5+top5+Min_per+usg,data=train,kernel="linear",cost=5)
svm_o
plot(svm_o,data=train,ORtg~Min_per)
svm2 <- train(oimp~ORtg+eFG+ORB_per+AST_per+TO_per+ftr+fromp5+top5+Min_per+usg,data=train, method = "svmLinear", trControl=trainControl(method = 'repeatedcv', 10, 20),  preProcess = c("center","scale"), tuneGrid = expand.grid(C = seq(1, 100, length = 20)))
control <- trainControl(method="cv",number=10) 
metric <- "Accuracy"
model <- train(oimp~ORtg+eFG+ORB_per+AST_per+TO_per+ftr+fromp5+top5+Min_per+usg, data = train, method = "svmLinear", 
               preProc = c("center","scale"),
               tuneGrid = expand.grid(C =1:20),
               metric=metric, trControl=control)
model
plot(model)
p <- predict(svm_o,test)
library(ggplot2)
pdf('/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/svm.off.model.pdf')
plot1 <- ggplot(data=test) + geom_point(aes(x=eFG,y=usg,colour=p)) + ggtitle('eFG vs. usg', subtitle='SVM Model') + labs(colour='oimp')
plot2 <- ggplot(data=test) + geom_point(aes(x=Min_per,y=usg,colour=p)) + ggtitle('Min_per vs. usg',subtitle='SVM Model') + labs(colour='oimp')
plot1
plot2
grid.arrange(plot1,plot2,ncol=2)
dev.off()
plot(test$ORtg,test$usg,col=p) #yes
plot(test$ORB_per,test$usg,col=p) #no
plot(test$eFG,test$usg,col=mod) #yes
plot(test$Min_per,test$usg,col=mod) #no
plot(test$AST_per,test$usg,col=mod) #no
plot(test$TO_per,test$usg,col=mod)#yes
plot(test$ftr,test$usg,col=mod) #no
plot(test$usg,test$fromp5,col=mod) #no
plot(test$usg,test$top5,col=mod) #no
cm <- table(label_test,p)
cm
acc <- sum(diag(cm))/length(label_test)
correct_improve <- cm[2,2]/(cm[2,2]+cm[2,1])
correct_deprove <- cm[1,1]/(cm[1,1]+cm[1,2])
print(acc)
print(correct_improve)
print(correct_deprove)
#defensive
library(caret)
drtg <- fulldata %>% dplyr::select(Min_per,DRB_per,blk_per,stl_per,dbpm,DBPM,fromp5,top5,dimp)
drtg$dimp <- as.factor(drtg$dimp)
drtg$top5 <- as.factor(drtg$top5)
drtg$fromp5 <- as.factor(drtg$fromp5)
set.seed(1234)
traindex <- sample(1:nrow(fulldata),.7*nrow(fulldata))
train <- drtg[traindex,]
label_train <- train$DBPM
train2 <- train %>% dplyr::select(-DBPM)
test <- drtg[-traindex,]
label_test <- test$dimp
svm_d <- svm(dimp~dbpm+DRB_per+blk_per+stl_per+fromp5+top5+Min_per,data=train,kernel='linear',cost=5)
svm2 <- train(dimp~dbpm+DRB_per+blk_per+stl_per+fromp5+top5+Min_per,data=train, method = "svmLinear", trControl=trainControl(method = 'repeatedcv', 10, 20),  preProcess = c("center","scale"), tuneGrid = expand.grid(C = seq(1, 100, length = 20)))
control <- trainControl(method="cv",number=10) 
metric <- "Accuracy"
model <- train(dimp~dbpm+DRB_per+blk_per+stl_per+fromp5+top5+Min_per, data = train, method = "svmLinear", 
               preProc = c("center","scale"),
               tuneGrid = expand.grid(C =1:20),
               metric=metric, trControl=control)
model
plot(model)
p <- predict(svm_d,test)
pdf('/Users/mileskee7/Desktop/College/Senior Year/Semester 2/Thesis/svm.def.model.pdf')
plot1 <- ggplot(data=test) + geom_point(aes(x=dbpm,y=Min_per,colour=p)) + ggtitle('dbpm vs min_per',subtitle='SVM Model') + labs(colour='dimp')
plot2 <- ggplot(data=test) + geom_point(aes(x=DRB_per,y=Min_per,colour=p)) + ggtitle('drb_per vs min_per',subtitle='SVM Model') + labs(colour='dimp')
plot1
plot2
grid.arrange(plot1,plot2,ncol=2)
dev.off()
plot(test$dbpm,test$Min_per,col=p) #yes
plot(test$DRB_per,test$Min_per,col=p) #no
plot(test$blk_per,test$Min_per,col=p) #no
plot(test$stl_per,test$Min_per,col=p) #no
plot(test$Min_per,test$fromp5,col=p) #no
plot(test$Min_per,test$top5,col=p) #yes
cm <- table(label_test,p)
cm
acc <- sum(diag(cm))/length(label_test)
correct_improve <- cm[2,2]/(cm[2,2]+cm[2,1])
correct_deprove <- cm[1,1]/(cm[1,1]+cm[1,2])
print(acc)
print(correct_improve)
print(correct_deprove)



