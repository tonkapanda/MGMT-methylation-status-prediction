# --- --- ---
# project: MGMT methylation status prediction
# author: evan chen
# info: k-NN model
# edits:
#   - weight function, number of neighbors, and distance power tuned
#   - optimization metric: roc_auc
# --- --- ---

library(tidymodels)
library(tidyverse)

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
print(paste("original training observations:", nrow(training_data_raw)))
print(paste("cleaned training observations:", nrow(training_data)))
print(paste("observations removed from training:", nrow(training_data_raw) - nrow(training_data)))
print("--- --- --- --- --- --- --- --- --- --- ---")
print(paste("original testing observations:", nrow(testing_data_raw)))
print(paste("cleaned testing observations:", nrow(testing_data)))
print(paste("observations removed from testing:", nrow(testing_data_raw) - nrow(testing_data)))
print("--- --- --- --- --- --- --- --- --- --- ---")

# knn model specification [weight function, neighbors, and dist_power to be tuned]

knn_spec <- nearest_neighbor(weight_func = tune(), neighbors = tune(), dist_power = tune()) |>
    set_engine("kknn") |>
    set_mode("classification")

knn_recipe <- recipe(MGMT_status ~ ., data = training_data) |>
    step_center(all_predictors()) |>
    step_scale(all_predictors())

knn_workflow <- workflow() |>
    add_recipe(knn_recipe) |>
    add_model(knn_spec)

# hyperparameters to tune: find best k, weight function, and dist_power

set.seed(1)

cv_folds <- vfold_cv(training_data, v = 5, strata = MGMT_status)

k_grid <- expand_grid(
    neighbors = seq(from = 1, to = 31, by = 2),
    weight_func = c("rectangular", "triangular", "gaussian"),
    dist_power = c(1, 1.5, 2)
)

knn_results <- workflow() |>
    add_recipe(knn_recipe) |>
    add_model(knn_spec) |>
    tune_grid(resamples = cv_folds, grid = k_grid, metrics = metric_set(accuracy, roc_auc))

knn_results_metrics <- knn_results |>
    collect_metrics()

tunings <- knn_results_metrics |>
    filter(.metric == "roc_auc")

# visualization of the tuning results with dist_power

tune_plot <- tunings |>
    mutate(dist_power = as_factor(dist_power)) |>
    ggplot(aes(x = neighbors, y = mean, color = weight_func, shape = dist_power)) +
    geom_point(size = 2) +
    geom_line(aes(group = interaction(weight_func, dist_power)), linewidth = 1) +
    facet_wrap(~weight_func) +
    labs(
        title = "k-NN model tuning: roc_auc vs. hyperparameters",
        x = "number of neighbors (k)",
        y = "mean roc_auc (5-fold cv)",
        color = "weight function",
        shape = "dist power"
    ) +
    theme_minimal() +
    theme(legend.position = "right")

tune_plot

# select the best overall model based on roc_auc

best_parameters <- select_best(knn_results, metric = "roc_auc")

print("--- --- --- best hyperparameters found (roc_auc) --- --- ---")
print(best_parameters)

# finalise and test model

# finalises workflow and confirms best parameters
final_knn_workflow <- finalize_workflow(knn_workflow, best_parameters)

# trains final model on the full training_data
# evaluate on the testing data

final_knn_spec <- nearest_neighbor(neighbors = best_parameters$neighbors, weight_func = best_parameters$weight_func, dist_power = best_parameters$dist_power) |>
    set_engine("kknn") |>
    set_mode("classification")

final_knn_fit <- workflow() |>
    add_recipe(knn_recipe) |>
    add_model(final_knn_spec) |>
    fit(data = training_data)

print("--- --- --- final model trained --- --- ---")
print(final_knn_fit)

final_predictions <- predict(final_knn_fit, testing_data, type = "prob") |>
    bind_cols(predict(final_knn_fit, testing_data, type = "class")) |>
    bind_cols(testing_data)

print("--- --- --- final test set metrics --- --- ---")
final_predictions |>
    metrics(truth = MGMT_status, estimate = .pred_class, .pred_methylated) |>
    print()

# save plots to plots

ggsave("plots/knn_tuning_plot.png",
    plot = tune_plot,
    width = 8,
    height = 5
)
