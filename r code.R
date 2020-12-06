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

sapply(src.chr, function(x) length(unique(x)))

src.chr <- Filter(is.character,src)
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

dumme <- dummyVars(" ~ .", data = src.chr[-11])
srcc1 <- data.frame(predict(dumme, newdata = src.chr[-11]))

#### Summary Feature Engineering

sprintf("The total number of columns in the intial dataset %d", length(src))
sprintf("The total number of Categorical Features %d", length(src.chr))
sprintf("The total number of Numerical Features %d", length(src.numeric))
sprintf("The total number of columns after feature engineering %d", length(src1))
print("The New Dummy variables added data frame")
names(src1)


######## Standarising the data

normalize <-function(x) {return((x -min(x))) / (max(x) -min(x))}

src.numeric.std <- sapply(src.numeric, function(x) normalize)
src.df <- cbind(src.numeric,srcc1)
src.df$y <- src$y
###### Dimension Reduction

# Perform Scree Plot and Parallel Analysis

fa.parallel(src.numeric, fa = "pc", n.iter = 100, show.legend = FALSE)

pc <-principal(df.norm[, 1:47], nfactors = 5, rotate = "none", scores = TRUE)
pc <-cbind(as.data.frame(pc$scores), df.norm$PHISHING_WEBSITE) %>%rename(PHISHING_WEBSITE = "df.norm$PHISHING_WEBSITE")


pca <- principal(src.numeric[,1:10],nfactors = 5, rotate = "none", scores = TRUE)
pca$scores

principal(src.numeric,nfactors=5,rotate="none")

### Training & Test Data Generation


## For PCA less data 
set.seed(99)
train.index <- sample(row.names(src.df), 0.6*dim(src.df)[1])
valid.index <- setdiff(row.names(src.df), train.index)
train.df <- src.df[train.index,]
valid.df <- src.df[valid.index,]


## For PCA applied data





