pop_data <- read.csv("/home/shreyashkharat/Datasets/creditcard.csv", header = TRUE)
pop_data_fixed <- read.csv("/home/shreyashkharat/Datasets/creditcard.csv", header = TRUE)
summary(pop_data)

# From the summary it is clear that the variable Amount has outliers.

# Outlier treatment for variable Amount
max_Amount <- quantile(pop_data$Amount, 0.90)
pop_data$Amount[pop_data$Amount > max_Amount] <- max_Amount
summary(pop_data$Amount)

# Train Test split
require("caTools")
set.seed(0)
split = sample.split(pop_data, SplitRatio = 0.8)
train_set = subset(pop_data, split == TRUE)
test_set = subset(pop_data, split == FALSE)

# Logistic Regression Model
model_logi <- glm(Class~., data = train_set, family = binomial)
summary(model_logi)
# Calculating probabilities
logi_prob <- predict(model_logi, test_set, type = "response")
# Declaring prediction array
logi_predict <- rep("NO", 64313)
logi_predict[logi_prob > 0.5] <- "YES"
# Confusion Matrix
table(logi_predict, test_set$Class)
(64184 + 78)/64313
# The Logistic Regression Model gives an R^2 value of 0.9992.
78/(78+39)
# The Logistic Regression Model gives an R^2 value of 0.6667 for fraudulent transactions.
# So, this model is moderately accurate.

# Linear Discriminant Analysis Model
require("MASS")
model_lda <- lda(Class~., train_set)
# Calculation of probabilities
lda_probs <- predict(model_lda, test_set, type = "response")
# Declaring prediction array
lda_predict <- lda_probs$class
# Confusion Matrix
table(lda_predict, test_set$Class)
(64181 + 84)/64313
# The Linear Discriminant Analysis model gives an R^2 value of 0.9992.
84/(84+33)
# The Linear Discriminant Analysis model gives an R^2 value of 0.7179 for fraudulent transactions.
# So, this model is also moderately accurate.

# Quadratic Discriminant Analysis Model
model_qda <- qda(Class~., train_set)
# Calculation of probabilities
qda_probs <- predict(model_qda, test_set, type = "response")
# Declaring prediction array
qda_predict <- qda_probs$class
# Confusion Matrix
table(qda_predict, test_set$Class)
(62624 + 96)/64313
# The Quadratic Discriminant Analysis model gives an R^2 value of 0.9752.
96/(96+21)
# The Quadratic Discriminant Analysis model gives an R^2 value of 0.8205 for fraudulent transactions.
# So, this model is highly accurate.

# K Nearest Neighbor Model
require("class")
# Gathering required arguments for knn function.
train_x = train_set[, -31]
train_y = train_set$Class
test_x = test_set[, -31]
test_y = test_set$Class
# Scaling train_x and test_x
train_x_scale <- scale(train_x)
test_x_scale <- scale(test_x)
set.seed(0)
#model_knn <- knn(train_x_scale, test_x_scale, train_y, k = 10)
# The K Nearest Neighbor model is a lazy algorithm, hence computationally expensive for large data sets.
# I have written the necessary code but due to unavailability of required computational strength, 
# the observations for K Nearest Neighbor are not mentioned.
# The Logistic Regression and Linear Discriminant Analysis model have equal total R^2 of 0.9992.
# The Quadratic Discriminant Analysis gave the highest conditional R^2 as 0.8205.
