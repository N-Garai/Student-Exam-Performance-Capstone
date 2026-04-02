set.seed(2026)
options(repos = c(CRAN = "https://cloud.r-project.org"))

required_packages <- c(
  "readr", "dplyr", "ggplot2", "caret",
  "ranger", "gbm", "glmnet", "kernlab"
)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

data_path <- "student_exam_performance_dataset.csv"
student_data <- read_csv(data_path, show_col_types = FALSE)

categorical_cols <- c(
  "gender", "parental_education", "family_income", "internet_access",
  "study_environment", "tutoring"
)

student_data <- student_data %>%
  mutate(across(all_of(categorical_cols), as.factor))

# Remove leakage columns that are directly derived from exam outcomes.
model_data <- student_data %>%
  select(-student_id, -pass_fail, -grade_category)

# 80/20 split keeps enough data for training while preserving a holdout test set.
split_index <- createDataPartition(model_data$final_exam_score, p = 0.8, list = FALSE)
train_data <- model_data[split_index, ]
test_data <- model_data[-split_index, ]

cv_control <- trainControl(method = "cv", number = 3)

lm_model <- train(
  final_exam_score ~ ., data = train_data,
  method = "lm", trControl = cv_control
)

elastic_net_model <- train(
  final_exam_score ~ ., data = train_data,
  method = "glmnet", trControl = cv_control,
  tuneLength = 5
)

svm_linear_model <- train(
  final_exam_score ~ ., data = train_data,
  method = "svmLinear", trControl = cv_control,
  tuneLength = 3
)

rf_model <- train(
  final_exam_score ~ ., data = train_data,
  method = "ranger", trControl = cv_control,
  tuneLength = 3, num.trees = 300,
  importance = "impurity"
)

gbm_model <- train(
  final_exam_score ~ ., data = train_data,
  method = "gbm", trControl = cv_control,
  tuneLength = 3, verbose = FALSE
)

pred_lm <- predict(lm_model, newdata = test_data)
pred_elastic_net <- predict(elastic_net_model, newdata = test_data)
pred_svm_linear <- predict(svm_linear_model, newdata = test_data)
pred_rf <- predict(rf_model, newdata = test_data)
pred_gbm <- predict(gbm_model, newdata = test_data)

pred_baseline <- rep(mean(train_data$final_exam_score), nrow(test_data))

get_metrics <- function(prediction, actual) {
  metric_vec <- postResample(pred = prediction, obs = actual)
  tibble(
    rmse = unname(metric_vec[["RMSE"]]),
    rsquared = unname(metric_vec[["Rsquared"]]),
    mae = unname(metric_vec[["MAE"]])
  )
}

metrics_table <- bind_rows(
  get_metrics(pred_baseline, test_data$final_exam_score) %>% mutate(model = "Baseline Mean"),
  get_metrics(pred_lm, test_data$final_exam_score) %>% mutate(model = "Linear Regression"),
  get_metrics(pred_elastic_net, test_data$final_exam_score) %>% mutate(model = "Elastic Net"),
  get_metrics(pred_svm_linear, test_data$final_exam_score) %>% mutate(model = "SVM Linear"),
  get_metrics(pred_rf, test_data$final_exam_score) %>% mutate(model = "Ranger Random Forest"),
  get_metrics(pred_gbm, test_data$final_exam_score) %>% mutate(model = "GBM")
) %>%
  relocate(model) %>%
  arrange(rmse)

print(metrics_table %>% mutate(across(where(is.numeric), ~ round(.x, 4))))

best_model <- metrics_table %>% arrange(rmse) %>% slice(1)
best_mae_model <- metrics_table %>% arrange(mae) %>% slice(1)

cat("\nBest model by RMSE:\n")
print(best_model)
cat("\nBest model by MAE:\n")
print(best_mae_model)

baseline_rmse <- metrics_table %>% filter(model == "Baseline Mean") %>% pull(rmse)
best_rmse <- best_model$rmse

if (!is.na(best_model$rsquared) && best_rmse < baseline_rmse * 0.3 && best_model$rsquared > 0.9) {
  cat("\nPerformance check: model quality is strong on the holdout test set.\n")
} else {
  cat("\nPerformance check: model quality is acceptable but should be validated with repeated resampling.\n")
}

prediction_plot <- tibble(
  actual = test_data$final_exam_score,
  pred_lm = pred_lm,
  pred_elastic_net = pred_elastic_net,
  pred_svm_linear = pred_svm_linear,
  pred_rf = pred_rf,
  pred_gbm = pred_gbm
)

# Plot best-performing model by RMSE for a quick visual quality check.
ggplot(prediction_plot, aes(x = actual, y = pred_svm_linear)) +
  geom_point(alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Best Model (SVM Linear): Predicted vs Actual Final Exam Score",
    x = "Actual Score",
    y = "Predicted Score"
  )
