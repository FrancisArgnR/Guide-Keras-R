####################################################
### Load the libraries and read and plot the data


library(keras)
library(quantmod)
library(forecast)

data <- read.csv("./data.csv") # The data are normalized

plot(data$x,type="l")


####################################################
### Create the matrix with the lags of the data (Because in real live, you have to predict with data of the past, you don't know the data of today)

# Lags
ii <- 4

# Matrix for lagged data (the original serie is now smaller due to the lags, lossing the values in the head)
j <- matrix(NA, nrow=length(data)-(ii), ncol=((ii)+1))
# Move the original serie the number of lags
j[,1] <- data[(ii+1):length(data)]  

# Fill the lag matrix
for(i in 2:(ii+1)){
  j[,i] <-Lag(data[1:length(data)],(i-1))[(ii+1):length(data)]
}
colnames(j) <- c("y",paste0("Lag.", 1:(ncol(j)-1)))

####################################################
### Get the train and test data (66% -> train, 33% -> test) (Train only with past data)

k <- trunc(length(data)*66/100)+1   # Position to divide the data

x.train <- as.matrix(j[(1:k),2:ncol(j)])
y.train <- as.matrix(j[(1:k),1]); colnames(y.train) <- "y"
x.test <- as.matrix(j[(k+1):nrow(j),2:ncol(j)])
y.test <- as.matrix(j[(k+1):nrow(j),1]); colnames(y.test) <- "y"

#################################################### 
### Define and fit the model (multilayer perceptron (more than one hidden layer))

# Seed to ensure the reproducibility (not complete)
set.seed(100)

# Create a sequential model and add layers using the pipe (%>%) operator
model <- keras_model_sequential() 

# The best combination of neurons, activation function, epoch, batch size and also layer is obtain by trian al error
model %>% 
  layer_dense(units = 4, activation = 'tanh', input_shape = c((ncol(j)-1))) %>% 
  layer_dense(units = 3, activation = 'tanh') %>% 
  #layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1, activation = 'linear')

summary(model)

# Set the measure of error and the optimizer
model %>% compile(
  loss = 'mse',
  optimizer = 'rmsprop'
)

# Fit the model
history <- model %>% fit(x.train, y.train, epochs = 50, batch_size = 1)

plot(history)

#################################################### 
### Calcula the error commited and plot the results

# Estimate model performance
trainScore <- model %>% evaluate(x.train, y.train, verbose=0)
print(paste('Train Score: ',trainScore,' MSE. ',sqrt(trainScore),' RMSE.', sep=""))

testScore <- model %>% evaluate(x.test, y.test, verbose=0)
print(paste('Test Score: ',testScore,' MSE. ',sqrt(testScore),' RMSE.', sep=""))

accuracy(test[,1],data[(k+1+ii):length(data)])[2]

# Prediction

#train <- model %>% predict_classes(x.train)
#test <- model %>% predict_classes(x.test)
train <- model %>% predict(x.train)
test <- model %>% predict(x.test)

# Plot the results

plot(data,type="l")
lines((1+ii):(k+ii),train,col="blue")
lines((k+1+ii):(length(data)),test,col="red")

