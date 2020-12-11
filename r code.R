library(dplyr)
library(xlsx)
library(stringr)
library(tidyr)
library(tidyverse)
library(ggplot2)
library(dummies)
library(caret)
library(corrplot)
library(psych)
library(MASS)
library(rpart)
library(rpart.plot)
library(boot)
library(tree)
library(neuralnet)
src <- read.csv('/Users/vigneshthanigaisivabalan/NEU/DM/Project/bank-additional/bank-additional-full.csv',TRUE, sep = ';')


#########                 Exploratary Data Analysis    ---- Source Data


### show discription of each of the column and explain why this is important in understanding why study of what each column talks about is important in analysis of the dataset


str(src) # to show the schema of the src 

head(src) # to display a sample of the data 

length(src) # to display the total size of the dataset

summary(src) # to display the statistics of each of column


# interpreting the boxplot of the numerical data to check for outliers

src.numeric <- Filter(is.numeric,src)

par(mfrow=c(3,2))
p1<- boxplot(src.numeric$age)
p2<- boxplot(src.numeric$duration)
p3<- boxplot(src.numeric$campaign)
p4<- boxplot(src.numeric$pdays)
p5<- boxplot(src.numeric$previous)
p6<- boxplot(src.numeric$emp.var.rate)
p7<- boxplot(src.numeric$cons.price.idx)
p8<- boxplot(src.numeric$cons.conf.idx)
p9<- boxplot(src.numeric$euribor3m)
p10<- boxplot(src.numeric$nr.employed)
dev.off()
##### we analyze each of the numeric

# For Age

ggplot(data=src, xName='Age')+
  geom_histogram(aes(x=age, fill =..count..),binwidth = 1)+scale_fill_gradient("Count", low = "green", high = "red")

ggplot(data=src, xName='Age')+geom_boxplot(aes(y=age))

# For Previous

ggplot(data=src, xName='Previous Contacts')+
  geom_bar(aes(x=previous, fill =..count..),binwidth = 1)+scale_fill_gradient("Count", low = "blue", high = "orange")


# For Duration

ggplot(data = src)+geom_density(aes(x=duration/60),fill="#69b3a2", color="#e9ecef", alpha=0.8)+xlim(0,40) # to show on average how much duration each call last


## For Emp. Rate


ggplot(data = src,aes(x=emp.var.rate))+geom_bar()+
  scale_x_continuous(breaks=c(1.1,1.4,-0.1,-0.2,-1.8,-2.9,-3.4,-3.0,-1.7,-1.1), labels=c("1.1","1.4","-0.1","-0.2","-1.8","-2.9","-3.4","-3.0","-1.7","-1.1"))

#scale_fill_manual(as.factor(src$emp.var.rate))

#scale_fill_gradient(exlabl)

ggplot(data = src)+geom_density(aes(x=emp.var.rate),fill="#69b3a2", color="#e9ecef", alpha=0.8) # to show on average how much duration each call last

ggplot(data=src, xName='Age')+
  geom_histogram(aes(x=emp.var.rate)) +scale_x_continuous(breaks=c(1.1,1.4,-0.1,-0.2,-1.8,-2.9,-3.4,-3.0,-1.7,-1.1), labels=c("1.1","1.4","-0.1","-0.2","-1.8","-2.9","-3.4","-3.0","-1.7","-1.1"))


# For Cons.conf.idx

ggplot(data = src)+geom_bar(aes(x=cons.conf.idx))
  


#### Graphical interperatation for Cateogorical variables


# For education:

ggplot(data = src)+geom_bar(aes(x=education, fill = education))


## For Martial Status

ggplot(data = src)+geom_bar(aes(x=marital, fill= marital))


## For Type of Jobs

ggplot(data = src)+ geom_bar(aes(x= job, fill = job))

## For Previous Call outcome

ggplot(data = src)+geom_bar(aes(x=poutcome, fill = poutcome))


## For  housing 

ggplot(data = src)+ geom_bar(aes(x= housing, fill = housing))


## For loan

ggplot(data =src)+ geom_bar(aes(x= loan, fill = loan))


## For defaulters

ggplot(data = src)+geom_bar(aes(x=default, fill = default))

## For Type of Contact made

ggplot(data = src)+geom_bar(aes(x= contact, fill = contact))


### For month

ggplot(data= src)+ geom_bar(aes(x= month, fill = month))

# no of null records in the dataset

colSums(is.na(src)) 


########## Checking for MultiColinearty

plot(src.numeric)

# to find the number categories in each of the categorical variable we are dealing with

src.chr <- Filter(is.character,src)


sapply(src.chr, function(x) length(unique(x)))
categ = NULL
categ1 = NULL
for (i in names(src.chr)) {
  categ = rbind(categ,i)
}

for (i in names(src.chr)) {
  categ1 = rbind(categ1,unique(src.chr[,i]))
}

categ = cbind(categ,categ1)




####      ONE HOT ENCODING  

## we can use both dummies or use contrast to create the factors for each of the categorical variable

## using contracts

dumme <- dummyVars(" ~ .", data = src.chr[-11]) ### having issue
srcc1 <- data.frame(predict(dumme, newdata = src.chr[-11])) ### having issue

srcc2 <- fastDummies::dummy_cols(src.chr[-11], remove_first_dummy = TRUE)
srcc2<- srcc2[,-c(1:10)]


#### Summary Feature Engineering

sprintf("The total number of columns in the intial dataset %d", length(src))
sprintf("The total number of Categorical Features %d", length(src.chr))
sprintf("The total number of Numerical Features %d", length(src.numeric))
sprintf("The total number of columns after feature engineering %d", length(src1))
print("The New Dummy variables added data frame")
names(src1)


######## Standarising the data & Merging Data

normalize <-function(x) {return((x -min(x))) / (max(x) -min(x))}

src.numeric.std <- as.data.frame(lapply(src.numeric, normalize))
src.chr$y <- as.factor(ifelse(src$y == "yes", 1,0))

srcc1$y <- src.chr$y
srcc2$y <- src.chr$y

srcc1 <- srcc2

srcc1<-src.chr


###### Dimension Reduction

# Perform Scree Plot and Parallel Analysis

fa.parallel(src.numeric, fa = "pc", n.iter = 100, show.legend = FALSE) # we shall consider only the numeric column

pca <- principal(src.numeric[,1:10],nfactors = 4, rotate = "none", scores = TRUE)
pca <- cbind(as.data.frame(pca$scores), srcc1)

principal(src.numeric,nfactors=4,rotate="none")

### Training & Test Data Generation

src.df <- cbind(src.numeric,srcc1)  #now we shall bind the nomalized numeric data with the dummy variable dataframe


## For PCA less data 
set.seed(99)
train.index <- sample(row.names(src.df), 0.6*dim(src.df)[1])
valid.index <- setdiff(row.names(src.df), train.index)
train.df <- src.df[train.index,]
valid.df <- src.df[valid.index,]


## For PCA applied data

train.pca.df <- pca[train.index,]
valid.pca.df <- pca[valid.index,]


###### Evaluating Performance

performance_list <-data.frame("Model" = character(),"AUC" = numeric(),"Accuracy" = numeric())

model_names <-list()
lift_charts <-list()
roc_curves <-list()

evaluate_performance <-function(pred, labels, model_name) {
  model_names[[length(model_names) + 1]] <<-model_name
  # Accuracy
  pred.class <-ifelse(slot(pred, "predictions")[[1]] > 0.5, "Yes", "No")
  levels(pred.class) <-make.names(levels(factor(pred.class)))
  acc <-confusionMatrix(table(pred.class, labels))$overall[[1]] * 100
  # ROC Plot
  roc <-performance(pred, "tpr", "fpr")
  plot(roc, col = "red", lwd = 2, main = paste0(model_name, " ROC Curve"))abline(a = 0, b = 1)
  roc_curves[[length(roc_curves) + 1]] <<-roc
  auc <-performance(pred, measure = "auc")
  temp <-data.frame("Model" = model_name,"AUC" = auc@y.values[[1]],"Accuracy" = acc)
  performance_list <<-rbind(performance_list, temp)
  print("Updated Performance List")
  lift <-performance(pred, "tpr", "rpp")
  plot(lift, main = paste0(model_name, " Lift Curve"), col = "green")abline(a = 0, b = 1)
  lift_charts[[length(lift_charts) + 1]] <<-liftrm(list = c("auc", "acc", "roc", "pred.class", "temp", "lift"))
}


#######  KNN Model

tr.control <- trainControl(method = "repeatedcv",number = 10,repeats = 3,classProbs = TRUE,summaryFunction = twoClassSummary)

knn.model <- train(y~., data = train.df,method = "knn",trControl = tr.control, metric ="ROC")
knn.model
plot(knn.model)

knn.valid_pred <- predict(knn.model,valid.df, type = "prob")
knn.prediction <-prediction(knn.valid_pred[, 2], valid.df$y)



perf_val <- performance(knn.model, "tpr", "fpr")
plot(perf_val, col = "green", lwd = 1.5)


###### Logistic Regression

### PCA data
logi.model.pca <- glm(y~., family = binomial, data = train.pca.df, maxit=100) # Full Model
summary(logi.model.pca)    ### this model has high AIC value and many i/p with high P value 


logi.model.pca.red <- glm(y~., family = binomial, data = train.pca.df, maxit=100) %>% stepAIC(trace = FALSE) # Reduced Model
summary(logi.model.pca)

anova(logi.model.pca, logi.model.pca.red, test = "Chisq")

logi.model.pca.fit <- predict.glm(logi.model.pca,valid.pca.df,type="response", se.fit = TRUE)



### Original Data

logi.model.ori <- glm(y~., family = binomial, data = train.df, maxit=100) ## Full Model
summary(logi.model.ori)    ### this model has high AIC value and many i/p with high P value 


logi.model.ori.red <- glm(y~., family = binomial, data = train.df, maxit=100) %>% stepAIC(trace = FALSE) #Reduced Model
summary(logi.model.ori.red)


anova(logi.model.ori, logi.model.ori.red, test = "Chisq") ## model tuning result

logi.model.pca.fit <- predict.glm(logi.model.ori.red,valid.df,type="response", se.fit = TRUE)



##### Desicion Tree

## PCA Data
set.seed(20)
dctree.model.pca <-rpart(y ~ ., data = train.pca.df, method = "class", cp = 0.00001, minsplit = 2, xval = 5)
rpart.plot(dctree.model.pca, main = "Classification Tree using PCA data")


### pruning data 

dctree.model.pca.prune <- prune(dctree.model.pca,cp=dctree.model.pca$cptable[which.min(dctree.model.pca$cptable[,"xerror"]), "CP"])
rpart.plot(dctree.model.pca.prune,type =2, main = "Pruned Classification Tree using PCA data")


## Original Data
set.seed(20)
dctree.model.ori <-rpart(y ~ ., data = train.pca.df, method = "class")
rpart.plot(dctree.model.ori, main = "Classification Tree using PCA data")


### pruning data 

dctree.model.ori.prune <- prune(dctree.model.pca,cp=dctree.model.pca$cptable[which.min(dctree.model.pca$cptable[,"xerror"]), "CP"])
rpart.plot(dctree.model.ori.prune,type =2, main = "Pruned Classification Tree using PCA data")


#####  Neural Networks

#### PCA Dataset

set.seed(20)

formu = y ~ age + duration + campaign + pdays + previous + emp.var.rate + cons.price.idx + cons.conf.idx + euribor3m + nr.employed + 
  `job_blue-collar` + job_entrepreneur + job_housemaid + job_management + job_retired + `job_self-employed` + job_services + job_student + 
  job_technician + job_unemployed + job_unknown + marital_married + marital_single + marital_unknown + education_basic.6y + education_basic.9y + 
  education_high.school + education_illiterate + education_professional.course + education_university.degree + education_unknown + default_unknown + 
  default_yes + housing_unknown + housing_yes + loan_unknown + loan_yes + contact_telephone + month_aug + month_dec + month_jul + 
  month_jun + month_mar + month_may + month_nov + month_oct + month_sep + day_of_week_mon + day_of_week_thu + day_of_week_tue + day_of_week_wed + poutcome_nonexistent + poutcome_success

nn.model.pca <-neuralnet(y~.,train.pca.df,hidden = 3, linear.output = T)
