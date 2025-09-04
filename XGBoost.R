################################################################################################
# R Machine Learning - XGBoost Interpretation Based on SHAP
# Load R packages and dataset
rm(list = ls())
library(shapviz)
library(ggplot2)
library(xgboost)
library(caret)
library(purrr)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(ggthemes)

# Set working directory
folder_path <- "F:/Research5/Environmental_Variable_Data/Pixel_Value_Extraction/"

# Get full paths of all CSV files in the folder
csv_files <- list.files(path = folder_path, pattern = "\\.csv$", full.names = TRUE)

# Initialize results list
results_list <- list()

# Loop through each CSV file
for (file in csv_files) {
  # Read current CSV file
  p1 <- read.csv(file)[, -c(1, 2, 3, 4, 5, 31, 36, 42, 43, 44)]
  df <- p1
  
  # Get file name without path and extension
  name <- basename(file)
  name <- sub("\\.csv$", "", name)
  
  # Split into training and testing sets
  set.seed(123)
  sample <- sample(c(TRUE, FALSE), nrow(df), replace = TRUE, prob = c(0.7, 0.3))
  trainData <- df[sample, ]
  testData <- df[!sample, ]
  
  # Prepare data for XGBoost
  dtrain <- xgb.DMatrix(data.matrix(trainData[, -1]), 
                        label = trainData[, 1])
  
  # Define parameter grid for tuning
  param_grid <- expand.grid(
    eta = c(0.01, 0.05, 0.1),
    max_depth = c(6, 8, 10),
    min_child_weight = c(1, 3, 5)
  )
  
  # Set up 5-fold cross-validation
  ctrl <- trainControl(
    method = "cv",
    number = 5,
    verboseIter = FALSE
  )
  
  # Tune parameters using caret
  xgb_tune <- train(
    x = data.matrix(trainData[, -1]),
    y = trainData[, 1],
    method = "xgbTree",
    trControl = ctrl,
    tuneGrid = param_grid,
    metric = "RMSE",
    nthread = 2,
    nrounds = 5000,
    objective = "reg:squarederror",
    eval_metric = "rmse",
    verbose = 0
  )
  
  # Get optimal parameters
  best_params <- xgb_tune$bestTune
  
  # Train final model with optimal parameters
  fit <- xgb.train(
    params = list(
      eta = best_params$eta,
      max_depth = best_params$max_depth,
      min_child_weight = best_params$min_child_weight,
      nthread = 2,
      eval_metric = "rmse",
      objective = "reg:squarederror"
    ),
    data = dtrain,
    nrounds = 5000,
    verbose = 0
  )
  
  # Make predictions on test set
  predictions <- predict(fit, newdata = as.matrix(testData[, -1]))
  
  # Calculate model performance metrics
  performance <- postResample(pred = predictions, obs = testData[, 1])
  r2 <- performance[["Rsquared"]]
  rmse <- performance[["RMSE"]]
  mae <- performance[["MAE"]]
  
  # Store performance metrics
  results_list[[name]] <- list(
    R2 = r2,
    RMSE = rmse,
    MAE = mae,
    Best_Params = best_params
  )
  
  # Print performance metrics
  cat(sprintf("\n%s - Model Performance:\n", name))
  cat(sprintf("R² = %.3f\n", r2))
  cat(sprintf("RMSE = %.3f\n", rmse))
  cat(sprintf("MAE = %.3f\n", mae))
  cat(sprintf("Best Parameters: eta=%.3f, max_depth=%d, min_child_weight=%d\n",
              best_params$eta, best_params$max_depth, best_params$min_child_weight))
  
  # Combine predicted and observed values
  lgb_res <- bind_cols(observed = testData[, 1], predicted = predictions)
  
  # Create scatter plot
  p_scatter <- ggplot(lgb_res, mapping = aes(observed, predicted)) +
    geom_point(alpha = 0.3, size = 2) +
    geom_abline(intercept = 0, slope = 1, col = "red") +
    labs(x = "Observed", y = "Predicted") +
    theme_bw() +
    ggtitle(paste("Model Performance -", name))
  
  print(p_scatter)
  
  # Save results
  write.csv(lgb_res, paste0("F:/Research5/Results/Attribution_Results/t/", name, "_Observed_Predicted.csv"))
  
  # Feature importance
  importance_matrix <- as.data.frame(xgb.importance(model = fit))
  write.csv(importance_matrix, paste0("F:/Research5/Results/Attribution_Results/t/", name, "_importance_matrix.csv"))
  
  # Calculate SHAP values
  shap_values <- shapviz(fit, X_pred = as.matrix(testData[, -1]), X = as.matrix(trainData[, -1]))
  
  # Create color palette
  colors <- brewer.pal(11, "RdBu")
  
  # Generate SHAP importance plot
  b <- sv_importance(shap_values, kind = "beeswarm") +
    theme_base() +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      text = element_text(family = "serif", size = 15),
      legend.position = "right",
      legend.key.height = unit(1.8, "cm"),
      legend.key.width = unit(0.3, "cm"),
      axis.ticks.length = unit(0.05, "cm"),
      axis.ticks = element_line(linewidth = 0.1)
    ) +
    scale_color_gradientn(colors = colors, na.value = 'transparent') +
    ggtitle(paste("SHAP Importance -", name))
  
  print(b)
  
  # Save plot
  t <- paste0("F:/Research5/Results/Attribution_Results/t/", name, "_beeswarm.pdf")
  ggsave(t, plot = b, width = 6.5, height = 4.2, units = "in")
}

# Save overall results
results_df <- do.call(rbind, lapply(results_list, function(x) {
  data.frame(
    R2 = x$R2,
    RMSE = x$RMSE,
    MAE = x$MAE,
    eta = x$Best_Params$eta,
    max_depth = x$Best_Params$max_depth,
    min_child_weight = x$Best_Params$min_child_weight
  )
}))
results_df$Dataset <- names(results_list)

write.csv(results_df, "F:/Research5/Results/Attribution_Results/t/Model_Performance_Summary.csv", row.names = FALSE)

# Print final summary
cat("\n=== FINAL SUMMARY ===\n")
print(results_df)
