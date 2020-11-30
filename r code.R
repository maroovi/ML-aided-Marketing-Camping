library(dplyr)
library(xlsx)
library(stringr)
library(tidyr)
library(tidyverse)


src <- read.csv('/Users/vigneshthanigaisivabalan/NEU/DM/Project/bank-additional/bank-additional-full.csv',TRUE, sep = ';')


#########                 Exploratary Data Analysis    ---- Source Data


### show discription of each of the column and explain why this is important in understanding why study of what each column talks about is important in analysis of the dataset


str(src) # to show the schema of the src 

head(src) # to display a sample of the data 

length(src) # to display the total size of the dataset

summary(src) # to display the statistics of each of column


# to find the number categories in each of the categorical variable we are dealing with

src.chr <- Filter(is.character,src)
for (i in names(src.chr)) {
  ifelse(length(factor(src.chr[i])) < 5, factor(src.chr[i]),0)
}
ifelse(length(factor(src.chr)) < 5, factor(src.chr),0)

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

ggplot(data = src,aes(x=cons.conf.idx))+geom_bar()+
  
  
  