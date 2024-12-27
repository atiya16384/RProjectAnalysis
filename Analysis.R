# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(knitr)
library(broom)      # For tidy model summaries
library(caret)      # For cross-validation
library(ggcorrplot) # For correlation matrix visualization
library(boot)       # For bootstrapping
library(mgcv)       # For GAMs
library(effectsize) # For effect size calculations
library(rstatix)    # For pairwise_t_test

# Load datasets
iris_data <- read.csv("iris_results.csv", stringsAsFactors = FALSE)
heart_disease_data <- read.csv("heart_disease_results.csv", stringsAsFactors = FALSE)
wine_quality_data <- read.csv("wine_quality_results.csv", stringsAsFactors = FALSE)

# Print column names and first few rows for debugging
cat("Iris Dataset Columns and Sample Rows:\n")
print(colnames(iris_data))
print(head(iris_data))

cat("\nHeart Disease Dataset Columns and Sample Rows:\n")
print(colnames(heart_disease_data))
print(head(heart_disease_data))

cat("\nWine Quality Dataset Columns and Sample Rows:\n")
print(colnames(wine_quality_data))
print(head(wine_quality_data))

# Normalize column names for consistency
normalize_colnames <- function(df) {
  colnames(df) <- gsub("\\.", "_", colnames(df))
  return(df)
}
iris_data <- normalize_colnames(iris_data)
heart_disease_data <- normalize_colnames(heart_disease_data)
wine_quality_data <- normalize_colnames(wine_quality_data)

# Combine datasets for analysis
datasets <- bind_rows(
  iris_data %>% mutate(Dataset = "Iris"),
  heart_disease_data %>% mutate(Dataset = "Heart Disease"),
  wine_quality_data %>% mutate(Dataset = "Wine Quality")
)

# Print combined dataset columns and first few rows for debugging
cat("\nCombined Dataset Columns and Sample Rows:\n")
print(colnames(datasets))
print(head(datasets))

# Ensure all numeric columns are correctly typed
datasets <- datasets %>%
  mutate(across(-Dataset, as.numeric))

# Summary Statistics
summary_stats <- datasets %>%
  group_by(Dataset) %>%
  summarise(
    Mean_Custom_Accuracy = mean(Custom_Accuracy, na.rm = TRUE),
    Mean_Sklearn_Accuracy = mean(Sklearn_Accuracy, na.rm = TRUE),
    Mean_Custom_Training_Time = mean(Custom_Training_Time, na.rm = TRUE),
    Mean_Sklearn_Training_Time = mean(Sklearn_Training_Time, na.rm = TRUE),
    Accuracy_Difference = Mean_Custom_Accuracy - Mean_Sklearn_Accuracy,
    Training_Time_Difference = Mean_Custom_Training_Time - Mean_Sklearn_Training_Time
  )

kable(summary_stats, caption = "Summary Statistics of Accuracy and Training Time")

# Linear Regression Analysis
linear_model <- lm(Custom_Accuracy ~ Train_Size + Max_Depth + Min_Samples_Split + Min_Samples_Leaf, data = datasets)

# Tidy linear regression summary
linear_model_summary <- broom::tidy(linear_model)
kable(linear_model_summary, caption = "Linear Regression Summary")

# 1. Bootstrapping for Mean Accuracy
bootstrap_mean <- function(data, indices) {
  d <- data[indices, ]
  return(mean(d$Custom_Accuracy, na.rm = TRUE))
}

bootstrap_results <- boot(data = datasets, statistic = bootstrap_mean, R = 1000)
print(bootstrap_results)
plot(bootstrap_results)

# 2. Generalized Additive Models (GAMs) for Accuracy
k_value <- 4
gam_model <- gam(
  Custom_Accuracy ~ 
    s(Train_Size, k = min(k_value, length(unique(datasets$Train_Size)))) + 
    s(Max_Depth, k = min(k_value, length(unique(datasets$Max_Depth)))) + 
    s(Min_Samples_Split, k = min(k_value, length(unique(datasets$Min_Samples_Split)))) + 
    s(Min_Samples_Leaf, k = min(k_value, length(unique(datasets$Min_Samples_Leaf)))), 
  data = datasets
)

# GAM Summary
gam_summary <- summary(gam_model)
print(gam_summary)
plot(gam_model, pages = 1)

# Compare Linear Regression and GAM Performance
linear_model_r2 <- summary(linear_model)$r.squared
gam_model_r2 <- gam_summary$r.sq
cat("Linear Model R-squared:", linear_model_r2, "\n")
cat("GAM Model R-squared:", gam_model_r2, "\n")

# Pairwise Cohen's d for Accuracy
pairwise_results <- datasets %>%
  pairwise_t_test(
    Custom_Accuracy ~ Dataset, 
    paired = FALSE, 
    p.adjust.method = "bonferroni"
  )

# Add effect size column using t-statistic
pairwise_results <- pairwise_results %>%
  mutate(
    effect_size = map2_dbl(statistic, df, ~ sqrt((.x^2) / (.x^2 + .y)))
  )

# Print the results
print(pairwise_results)


# Partial Eta Squared for ANOVA
anova_model <- aov(Custom_Accuracy ~ Dataset, data = datasets)
eta_squared_accuracy <- eta_squared(anova_model)
print(eta_squared_accuracy)

# Paired T-tests
t_test_results <- datasets %>%
  group_by(Dataset) %>%
  summarise(
    Accuracy_T_Value = t.test(Custom_Accuracy, Sklearn_Accuracy, paired = TRUE)$statistic,
    Accuracy_P_Value = t.test(Custom_Accuracy, Sklearn_Accuracy, paired = TRUE)$p.value,
    Training_T_Value = t.test(Custom_Training_Time, Sklearn_Training_Time, paired = TRUE)$statistic,
    Training_P_Value = t.test(Custom_Training_Time, Sklearn_Training_Time, paired = TRUE)$p.value
  )

# Display T-Test Results
kable(t_test_results, caption = "Paired T-Test Results")

# 4. Advanced Data Visualization
violin_plot <- ggplot(datasets, aes(x = Dataset, y = Custom_Accuracy, fill = Dataset)) +
  geom_violin() +
  theme_minimal() +
  labs(title = "Violin Plot of Accuracy by Dataset", x = "Dataset", y = "Custom Accuracy")
print(violin_plot)

density_plot <- ggplot(datasets, aes(x = Custom_Accuracy, fill = Dataset)) +
  geom_density(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Density Plot of Accuracy Across Datasets", x = "Accuracy", y = "Density")
print(density_plot)

# Correlation Matrix
correlation_matrix <- datasets %>%
  ungroup() %>% 
  dplyr::select(Train_Size, Max_Depth, Min_Samples_Split, Min_Samples_Leaf, Custom_Accuracy, Custom_Training_Time) %>% 
  cor(use = "pairwise.complete.obs")

# Visualize Correlation Matrix
ggcorrplot(correlation_matrix, method = "circle", type = "lower", lab = TRUE, title = "Correlation Matrix")

# Cross-Validation for Linear Regression
cv_control <- trainControl(method = "cv", number = 5)
cv_linear_model <- train(
  Custom_Accuracy ~ Train_Size + Max_Depth + Min_Samples_Split + Min_Samples_Leaf, 
  data = datasets, 
  method = "lm", 
  trControl = cv_control
)

cv_linear_model_results <- as.data.frame(cv_linear_model$results)
kable(cv_linear_model_results, caption = "Cross-Validation Results for Linear Regression")

# Additional Analysis: ANOVA for Accuracy and Training Time
anova_accuracy <- aov(data = datasets, Custom_Accuracy ~ Dataset)
anova_training_time <- aov(data = datasets, Custom_Training_Time ~ Dataset)

# Summary of ANOVA
anova_accuracy_summary <- summary(anova_accuracy)
anova_training_time_summary <- summary(anova_training_time)

# Display ANOVA Results
anova_accuracy_summary
anova_training_time_summary

# Tukey’s HSD Test (if ANOVA is significant)
if (anova_accuracy_summary[[1]]$`Pr(>F)`[1] < 0.05) {
  tukey_accuracy <- TukeyHSD(anova_accuracy)
  print(tukey_accuracy)
}

if (anova_training_time_summary[[1]]$`Pr(>F)`[1] < 0.05) {
  tukey_training_time <- TukeyHSD(anova_training_time)
  print(tukey_training_time)
}

# Additional Visualization: Accuracy vs Dataset
accuracy_plot <- ggplot(datasets, aes(x = Dataset, y = Custom_Accuracy, fill = Dataset)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Custom Accuracy by Dataset", x = "Dataset", y = "Custom Accuracy")
print(accuracy_plot)

# Residual Analysis for Linear Model
residual_analysis <- ggplot(data = augment(linear_model), aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(title = "Residuals vs Fitted Values", x = "Fitted Values", y = "Residuals")
print(residual_analysis)

# Advanced Visualization: Interaction Plot
interaction_plot <- ggplot(datasets, aes(x = Train_Size, y = Custom_Accuracy, color = Dataset)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_minimal() +
  labs(title = "Interaction Plot: Train Size vs Custom Accuracy", x = "Train Size", y = "Custom Accuracy")
print(interaction_plot)



