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
library(GGally)     # For pair plots

# ---------------------- Data Loading ---------------------- #
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

# ------------------ Data Preprocessing ------------------- #
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

# ---------------------- Combined Analysis ---------------------- #
# Summary Statistics for Combined Data
combined_summary_stats <- datasets %>%
  group_by(Dataset) %>%
  summarise(
    Mean_Custom_Accuracy = mean(Custom_Accuracy, na.rm = TRUE),
    Mean_Sklearn_Accuracy = mean(Sklearn_Accuracy, na.rm = TRUE),
    Mean_Custom_Training_Time = mean(Custom_Training_Time, na.rm = TRUE),
    Mean_Sklearn_Training_Time = mean(Sklearn_Training_Time, na.rm = TRUE),
    Accuracy_Difference = Mean_Custom_Accuracy - Mean_Sklearn_Accuracy,
    Training_Time_Difference = Mean_Custom_Training_Time - Mean_Sklearn_Training_Time,
    .groups = "drop"
  )
kable(combined_summary_stats, caption = "Combined Summary Statistics of Accuracy and Training Time")

# Linear Regression for Combined Data
combined_linear_model <- lm(Custom_Accuracy ~ Train_Size + Max_Depth + Min_Samples_Split + Min_Samples_Leaf, data = datasets)
combined_linear_model_summary <- broom::tidy(combined_linear_model)
kable(combined_linear_model_summary, caption = "Combined Linear Regression Summary")

# Bootstrapping for Combined Mean Accuracy
bootstrap_mean <- function(data, indices) {
  d <- data[indices, ]
  return(mean(d$Custom_Accuracy, na.rm = TRUE))
}
combined_bootstrap_results <- boot(data = datasets, statistic = bootstrap_mean, R = 1000)
print(combined_bootstrap_results)
plot(combined_bootstrap_results)

# Generalized Additive Models (GAMs) for Combined Data
k_value <- 4
combined_gam_model <- gam(
  Custom_Accuracy ~ 
    s(Train_Size, k = min(k_value, length(unique(datasets$Train_Size)))) + 
    s(Max_Depth, k = min(k_value, length(unique(datasets$Max_Depth)))) + 
    s(Min_Samples_Split, k = min(k_value, length(unique(datasets$Min_Samples_Split)))) + 
    s(Min_Samples_Leaf, k = min(k_value, length(unique(datasets$Min_Samples_Leaf)))), 
  data = datasets
)
combined_gam_summary <- summary(combined_gam_model)
cat("Combined GAM Model Summary:\n")
print(combined_gam_summary)
plot(combined_gam_model, pages = 1)

# Advanced Statistical Analysis: Paired t-tests and ANOVA
paired_t_test_results <- datasets %>%
  group_by(Dataset) %>%
  summarise(
    T_Value = t.test(Custom_Accuracy, Sklearn_Accuracy, paired = TRUE)$statistic,
    P_Value = t.test(Custom_Accuracy, Sklearn_Accuracy, paired = TRUE)$p.value,
    Effect_Size = effectsize::repeated_measures_d(Custom_Accuracy, Sklearn_Accuracy)$Cohen_d,
    .groups = "drop"
  )
kable(paired_t_test_results, caption = "Paired T-Test Results for Accuracy")

anova_results <- aov(Custom_Accuracy ~ Dataset, data = datasets)
anova_summary <- summary(anova_results)
print(anova_summary)

if (anova_summary[[1]]$`Pr(>F)`[1] < 0.05) {
  tukey_results <- TukeyHSD(anova_results)
  print(tukey_results)
}

# Residual Diagnostics for Models
par(mfrow = c(1, 2))
plot(combined_linear_model, which = 1:2)
plot(combined_gam_model, pages = 1)

# AIC for Model Comparison
aic_comparison <- AIC(combined_linear_model, combined_gam_model)
kable(as.data.frame(aic_comparison), caption = "AIC Comparison of Linear and GAM Models")

# ---------------------- Visualization ---------------------- #
# Density Plots for Accuracy
ggplot(datasets, aes(x = Custom_Accuracy, fill = Dataset)) +
  geom_density(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Density Plot of Custom Accuracy by Dataset", x = "Custom Accuracy", y = "Density")

# Scatterplots for Computational Times
ggplot(datasets, aes(x = Custom_Training_Time, y = Sklearn_Training_Time, color = Dataset)) +
  geom_point(alpha = 0.7) +
  theme_minimal() +
  labs(title = "Scatterplot of Training Times (Custom vs Sklearn)", 
       x = "Custom Training Time", y = "Sklearn Training Time")

# Pairplot for Numeric Variables
numeric_columns <- datasets %>% select(Custom_Accuracy, Sklearn_Accuracy, Train_Size, Max_Depth)
ggpairs(numeric_columns, aes(color = datasets$Dataset))



# Predicted vs Observed Plot
ggplot(data.frame(Observed = datasets$Custom_Accuracy, Predicted = predict(combined_linear_model)), 
       aes(x = Observed, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(title = "Predicted vs Observed Accuracy", x = "Observed Accuracy", y = "Predicted Accuracy")

# Violin Plots for Accuracy
ggplot(datasets, aes(x = Dataset, y = Custom_Accuracy, fill = Dataset)) +
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
  theme_minimal() +
  labs(title = "Violin Plot of Custom Accuracy by Dataset", x = "Dataset", y = "Custom Accuracy")

# Correlation Heatmap
selected_columns <- datasets %>% select(Custom_Accuracy, Sklearn_Accuracy, Train_Size, Max_Depth, Min_Samples_Split, Min_Samples_Leaf)
correlation_matrix <- cor(selected_columns, use = "complete.obs")
ggcorrplot(
  correlation_matrix, 
  method = "circle", 
  type = "lower", 
  lab = TRUE, 
  title = "Heatmap of Feature Correlations"
)


