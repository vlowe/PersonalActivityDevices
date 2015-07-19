# Classify data from personal activity devices when used during
# different exercise types.
# Vicki Lowe

# Output
# accuracy 0.9942098
# accuracy.by.class 0.9993939 0.9946714 0.9912281 0.9849095 0.9972119
# true.positive 0.998184 0.9937888 0.9826087 0.9959308 0.9981395
# false.positive 0.001815981 0.00621118 0.0173913 0.004069176 0.001860465

# Load libraries
library(doParallel)
library(doRNG)
library(foreach)
library(randomForest)

CreatePlot <- function(x, y, position){
  # Create scatterplot between two vectors
  #
  # Args:
  #          x: Variable to be plotted on the x-axis
  #          y: Variable to be plotted on the y-axis
  #   position: Location of the legend
  png(paste(x, "_", y, ".png", sep = ""))
  plot(data.frame(data[x], data[y]), col = data$Class,
       main = paste(x, " VS ", y, sep = ""), 
       xlab = x,
       ylab = y)
  legend(position, inset = 0.05, legend = c("A", "B", "C", "D", "E"),
         col = 1:5, pch = 1)
  dev.off()
}

# Set up parallel processing on cores available
kNumCores = detectCores()
cl <- makeCluster(kNumCores)
registerDoParallel(cl)

# Make results reproducible by setting same seed
set.seed(123)  # Arbitrarily chosen

# Read data
data <- read.csv(file = 'Dataset.csv', header = T, sep = ',', 
                  na.strings = c("#DIV/0!", "NA"))

# Fix up typos in header names
names(data) <- gsub("picth", "pitch", names(data))

# Identify whether each column contains any NaN values
complete.columns <- apply(data, 2, function(x){
  !anyNA(x)
})

# Drop obvious nonsense data (e.g. Obs, timestamp)
valid.data <- data[complete.columns]
consistent.data <- (valid.data[c(2, 8:length(valid.data))])

# Set training data to 70% and test data to 30%
ind <- sample(2, nrow(consistent.data), replace = TRUE, prob = c(0.7, 0.3))
train.data <- consistent.data[ind == 1, ]
test.data <- consistent.data[ind == 2, ]

# Generate the random forest with 25 runs on each core
data.rf <- foreach(ntree = rep(25, kNumCores), .combine = combine, 
                   .multicombine = TRUE, .packages = 'randomForest') %dorng%
  randomForest(Class~., data = train.data, ntree = ntree,
               importance = TRUE)

confusion.matrix.train <- table(predict(data.rf), train.data$Class)

# Test model
class.pred <- predict(data.rf, newdata = test.data)
confusion.matrix <- table(class.pred, test.data$Class)

# Calculate results
accuracy                <- sum(diag(confusion.matrix))/sum(confusion.matrix)
accuracy.by.class       <- diag(confusion.matrix)/colSums(confusion.matrix)
true.positive.by.class  <- diag(confusion.matrix)/rowSums(confusion.matrix)
false.positive.by.class <- ((rowSums(confusion.matrix) - diag(confusion.matrix))
                            / rowSums(confusion.matrix))

# Print results
importance(data.rf)
confusion.matrix
cat(accuracy, fill = T, labels = "accuracy")
cat(accuracy.by.class, fill = T, labels = "accuracy.by.class")
cat(true.positive.by.class, fill = T, labels = "true.positive.by.class")
cat(false.positive.by.class, fill = T, labels = "false.positive.by.class")

# Plot margin of predictions
png('marginPlot.png')
plot(margin(data.rf, test.data$Class))
dev.off()

# Plot of importance of variables
png('varImpPlot.png')
varImpPlot(data.rf, main = "Importance of Variables")
dev.off()

# Plots of different variables which show importance
CreatePlot("gyros_dumbbell_y", "pitch_forearm", "topright")
CreatePlot("yaw_belt", "roll_belt", "bottomright")
CreatePlot("yaw_belt", "pitch_belt", "topright")
CreatePlot("gyros_dumbbell_y", "pitch_forearm", "bottomleft")

# Shut down cluster
stopCluster(cl)
