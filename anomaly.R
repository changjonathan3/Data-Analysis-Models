# Polynomial Regression

# Importing the dataset
dataset = read.csv('anomaly.csv')
dataset = dataset[1:2]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Annual.Anomaly ~ .,
             data = dataset)

# Fitting Polynomial Regression to the dataset
dataset$Year2 = dataset$Year^2
dataset$Year3 = dataset$Year^3
dataset$Year4 = dataset$Year^4
poly_reg = lm(formula = Annual.Anomaly ~ .,
              data = dataset)

# Visualising the Linear Regression results
#install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Year, y = dataset$Annual.Anomaly),
             colour = 'red') +
  geom_line(aes(x = dataset$Year, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Temp anomaly (Linear Regression)') +
  xlab('Year') +
  ylab('Annual Anomaly')

# Visualising the Polynomial Regression results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Year, y = dataset$Annual.Anomaly),
             colour = 'red') +
  geom_line(aes(x = dataset$Year, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Temp Anomaly (Polynomial Regression)') +
  xlab('Year') +
  ylab('Annual Anomaly')

# Predicting a new result with Linear Regression
predict(lin_reg, data.frame(Year = 6.5))

# Predicting a new result with Polynomial Regression
predict(poly_reg, data.frame(Year = 6.5,
                             Year2 = 6.5^2,
                             Year3 = 6.5^3,
                             Year4 = 6.5^4))