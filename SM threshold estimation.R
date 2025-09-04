
############## Loop through each latitude and longitude point

# Load necessary libraries 
library(segmented)

# Read EF and SM data
ef_file <- "F:/论文撰写/论文5/数据/转tif/EF连接/2009EF.csv"
sm_file <- "F:/论文撰写/论文5/数据/转tif/SM连接/2009SM.csv"
ef <- read.csv(ef_file)
sm <- read.csv(sm_file)

# Initialize result storage vector
results <- data.frame(x = numeric(), y = numeric(), t = numeric(), s = numeric(), EFmax = numeric(), Err = numeric())

######## Define functions
# Define function to find the knee point
find_knee_point <- function(data, min_sample_size) {
  max_slope_change <- 0
  optimal_t <- NA
  nrows <- nrow(data)
  
  for (t_index in (min_sample_size + 1):(nrows - min_sample_size)) {
    df1 <- data[1:t_index, ]
    df2 <- data[(t_index + 1):nrow(data), ]
    
    # Ensure each subset after splitting contains at least min_sample_size rows
    if (nrow(df1) >= min_sample_size && nrow(df2) >= min_sample_size) {
      model1 <- lm(EF ~ SM, data = df1)
      model2 <- lm(EF ~ SM, data = df2)
      
      slope1 <- coef(model1)[2]
      slope2 <- coef(model2)[2]
      
      if (!is.na(slope1) && !is.na(slope2) && slope1 > 0 && slope2 > 0) {
        slope_change <- abs(slope2 - slope1)
        if (slope_change > max_slope_change) {
          max_slope_change <- slope_change
          optimal_t <- mean(c(df1$SM[nrow(df1)], df2$SM[1]))
        }
      }
    }
  }
  
  if (max_slope_change > 0) {
    return(t = optimal_t)
  } else {
    return(NA)
  }
}

# Loop through each latitude and longitude point
for (i in 1:nrow(ef)) {
  # Extract EF and SM row values for the current point
  ef_values <- t(ef[i, (720:1800)])
  sm_values <- t(sm[i, (720:1800)])
  data <- data.frame(SM = sm_values, EF = ef_values)
  colnames(data) <- c("SM", "EF")
  data <- na.omit(data)
  filtered_data <- subset(data, EF > 0 & EF < 1 & SM > 0 & SM < 1)
  
  if (nrow(filtered_data) < 100) {  # Skip if there are too few data points
    next
  }
  
  # Set minimum sample size limit
  min_sample_size <- 100
  
  # Find the knee point
  t_knee <- find_knee_point(filtered_data, min_sample_size)
  if (is.na(t_knee)) {  # Skip if no knee point is found
    next
  }
  
  # Define segmented regression model
  segmented_model <- function(beta, SM) {
    t <- beta[1]
    s <- beta[2]
    EFmax <- beta[3]
    EF <- ifelse(SM < t, s * SM, EFmax)
    return(EF) 
  }
  
  # Use the found knee point as the initial threshold
  initial_params <- c(t = t_knee, s = 1, EFmax = mean(data$EF))
  
  # Fit a linear model using lm(), then use segmented() to add segmented regression
  linear_model <- lm(EF ~ SM, data = data)
  
  # Attempt to fit the segmented regression model and handle possible errors
  segmented_fit <- try(segmented(linear_model, seg.Z = ~ SM, psi = list(SM = initial_params[1])), silent = TRUE)
  
  if (inherits(segmented_fit, "try-error") || 
      any(is.na(segmented_fit$psi[, "Est."])) || 
      any(is.infinite(segmented_fit$psi[, "Est."])) || 
      any(is.na(segmented_fit$se)) || 
      any(is.infinite(segmented_fit$se)) || 
      coef(segmented_fit)["SM"] < 0) {
    next  # Skip if segmented regression fails or s value is negative
  }
  
  # Extract fitting results
  fit_results <- summary(segmented_fit)
  
  # Accuracy evaluation section - calculate model performance metrics
  predicted <- predict(segmented_fit, newdata = data)
  observed <- data$EF
  
  # Calculate evaluation metrics
  r_squared <- 1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2)
  rmse <- sqrt(mean((observed - predicted)^2))
  mae <- mean(abs(observed - predicted))
  mse <- mean((observed - predicted)^2)
  nse <- 1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2)  # Nash-Sutcliffe efficiency coefficient
  
  # Store results, including accuracy evaluation metrics
  results <- data.frame(
    x = ef$x[i], 
    y = ef$y[i],
    t = fit_results$psi[, "Est."][1], 
    s = coef(segmented_fit)["SM"], 
    EFmax = mean(ef_values[sm_values >= fit_results$psi[, "Est."][1]]),
    Err = fit_results$psi[, "St.Err"],
    R_squared = r_squared,
    RMSE = rmse,
    MAE = mae,
    MSE = mse,
    NSE = nse,
    n_obs = nrow(data)  # Number of observed data points
  )
  
  # Set column names
  colnames(results) <- c("x", "y", "t", "s", "EFmax", "Err", "R_squared", "RMSE", "MAE", "MSE", "NSE", "n_obs")
  
  # Generate output filename
  output_file <- sprintf("F:/论文撰写/论文5/数据/转tif/t,s,efmax计算/2009/%d.csv", i)
  
  # Save results to CSV file
  write.csv(results, output_file, row.names = FALSE)
  
  # Optional: Output progress and accuracy information to the console
  if (i %% 100 == 0) {
    cat(sprintf("Processed %d points, Current R²: %.3f, RMSE: %.3f\n", 
                i, r_squared, rmse))
  }
}

# Read all result files and calculate overall accuracy metrics
result_files <- list.files("F:/论文撰写/论文5/数据/转tif/t,s,efmax计算/2009/", 
                          pattern = "\\.csv$", full.names = TRUE)

all_results <- do.call(rbind, lapply(result_files, read.csv))

# Calculate overall accuracy statistics
overall_accuracy <- data.frame(
  Mean_R2 = mean(all_results$R_squared, na.rm = TRUE),
  Median_R2 = median(all_results$R_squared, na.rm = TRUE),
  Mean_RMSE = mean(all_results$RMSE, na.rm = TRUE),
  Median_RMSE = median(all_results$RMSE, na.rm = TRUE),
  Successful_points = nrow(all_results),
  Total_points = nrow(ef)
)

print(overall_accuracy)
