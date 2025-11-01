# --- --- --- 
# project: MGMT methylation status prediction
# author: evan chen
# info: k-NN baseline model
# results:
#   best weight function: rectangular
#   best k: 13
#   accuracy: 70.5%
# --- --- --- 

library(tidymodels)
library (tidyverse)

# 20 radiomics features selected by XGB model [figure 5]

xgb_features <- c(
  "TEXTURE_GLSZM_ET_T2_GLV",
  "TEXTURE_GLSZM_NET_T1_SZE",
  "TEXTURE_GLRLM_NET_T2_RLV",
  "TEXTURE_GLRLM_ED_T2_GLV",
  "HISTO_ET_T2_Bin6",
  "HISTO_ET_T2_Bin2",
  "TEXTURE_GLRLM_NET_T1_SRE",
  "TEXTURE_NGTDM_ET_FLAIR_Coarseness",
  "SPATIAL_Temporal",
  "TEXTURE_GLRLM_NET_T2_SRLGE",
  "TEXTURE_GLCM_ED_T2_AutoCorrelation",
  "TEXTURE_GLSZM_ED_T2_HGZE",
  "TEXTURE_GLSZM_NET_T1_GLV",
  "HISTO_NET_FLAIR_Bin7",
  "TEXTURE_NGTDM_NET_FLAIR_Contrast",
  "INTENSITY_Mean_ET_T1",
  "TEXTURE_GLOBAL_ET_T2_Kurtosis",
  "TEXTURE_GLRLM_ET_T2_GLN",
  "HISTO_ET_T1Gd_Bin6",
  "VOLUME_ET_OVER_ED"
)

# training data

training_data_raw <- read_csv("data/training_dataset.csv") |>
  select(MGMT_status, all_of(xgb_features)) |>
  mutate(MGMT_status = as_factor(str_to_lower(MGMT_status)))

# testing data

testing_data_raw <- read_csv("data/testing_dataset.csv") |>
  rename(MGMT_status = `MGMT promoter status`) |>
  select(MGMT_status, all_of(xgb_features)) |>
  mutate(MGMT_status = as_factor(str_to_lower(MGMT_status)))

# removing all observations with n/a values
training_data <- training_data_raw |>
  filter(across(all_of(xgb_features), is.finite))

testing_data <- testing_data_raw |>
  filter(across(all_of(xgb_features), is.finite))

print("--- --- --- data cleaning info --- --- --- ")
print(paste("original training rows:", nrow(training_data_raw)))
print(paste("cleaned training rows:", nrow(training_data)))
print(paste("rows removed from training:", nrow(training_data_raw) - nrow(training_data)))
print("--- --- --- --- --- --- --- --- --- --- ---")
print(paste("original testing rows:", nrow(testing_data_raw)))
print(paste("cleaned testing rows:", nrow(testing_data)))
print(paste("rows removed from testing:", nrow(testing_data_raw) - nrow(testing_data)))
print("--- --- --- --- --- --- --- --- --- --- ---")

# knn model specification [weight function and neighbors to be tuned during model fitting]

knn_spec <- nearest_neighbor(weight_func  = tune(), neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("classification")

knn_recipe <- recipe(MGMT_status ~ ., data = training_data) |>
  step_center(all_predictors()) |>
  step_scale(all_predictors())

knn_workflow <- workflow() |>
  add_recipe(knn_recipe) |>
  add_model(knn_spec)

# hyperparameters to tune: find best k and weight function

set.seed(1)

cv_folds <- vfold_cv(training_data, v = 5, strata = MGMT_status)

k_grid <- expand_grid(
  neighbors = seq(from = 1, to = 31, by = 2),
  weight_func = c("rectangular", "triangular", "gaussian")
)

knn_results <- workflow() |>
  add_recipe(knn_recipe) |>
  add_model(knn_spec) |>
  tune_grid(resamples = cv_folds, grid = k_grid)

knn_results_metrics <- knn_results |>
  collect_metrics()
  
accuracies <- knn_results_metrics |>
  filter(.metric == "accuracy")

# visualization of the tuning results

tune_plot <- accuracies |>
  ggplot(aes(x = neighbors, y = mean, color = weight_func)) +
  geom_point(size = 2) +
  geom_line(linewidth = 1) +
  facet_wrap(~weight_func) +
  labs(
    title = "k-NN model tuning: accuracy vs. hyperparameters", 
    x = "number of neighbors (k)",
    y = "mean accuracy (5-fold cv)",
    color = "weight function"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

tune_plot

# select the best overall model

best_parameters <- select_best(knn_results, metric = "accuracy")

print("--- --- --- best hyperparameters found --- --- ---")
print(best_parameters)

# finalise and test model

# finalises workflow and confirms best parameters
final_knn_workflow <- finalize_workflow(knn_workflow, best_parameters)

# trains final model on the full training_data
# evaluates it once on the testing data

final_knn_spec <- nearest_neighbor(neighbors = best_parameters$neighbors, weight_func = best_parameters$weight_func) |>
  set_engine("kknn") |>
  set_mode("classification")

final_knn_fit <-workflow() |>
  add_recipe(knn_recipe) |>
  add_model(final_knn_spec) |>
  fit(data = training_data)

print("--- --- --- final model trained --- --- ---")
print(final_knn_fit)

final_predictions <- predict(final_knn_fit, testing_data) |>
  bind_cols(testing_data)

print("--- --- --- final test set accuracy --- --- ---")
final_predictions |>
  metrics(truth = MGMT_status, estimate = .pred_class) |>
  filter(.metric == "accuracy") |>
  print()

# save plots to plots

ggsave("plots/knn_tuning_plot.png", 
       plot = tune_plot, 
       width = 8, 
       height = 5)


  
  
  




