
# Required Libraries
library(haven)
library(dplyr)
library(CVXR)
library(writexl)

# Train-Test Split Function 
train_test_split <- function(data, train_ratio = 0.7) {
  # Separate data from 2010-2016
  data_2010_2016 <- data[data$yearAdmission != 7, ]
  
  # Calculate training size for 2010-2016 data
  n_2010_2016 <- nrow(data_2010_2016)
  train_size <- floor(n_2010_2016 * train_ratio)
  
  # Random shuffle of 2010-2016 data
  shuffled_indices <- sample(n_2010_2016)
  
  # Split indices for 2010-2016 data
  train_indices <- shuffled_indices[1:train_size]
  
  # Prepare training data from 2010-2016
  train_data <- data_2010_2016[train_indices, ]
  
  # Prepare test data (only from 2017)
  test_data <- data[data$yearAdmission == 7, ]
  
  list(
    train_data = train_data,
    test_data = test_data,
    train_indices = train_indices
  )
}

# Main Evaluation Function
evaluate_regression <- function(data, train_ratio = 0.7) {
  # R-squared function
  rsquared <- function(y, predy){
    SSR <- sum((y-predy)^2)
    SST <- sum((y-mean(y))^2)
    R2 <- 1-SSR/SST
    return(R2)
  }
  
  # Prepare data
  split_data <- train_test_split(data, train_ratio)
  
  # Prepare training data
  train_data <- split_data$train_data
  train_grp <- train_data$grp
  n_train_grp <- sum(train_grp)
  
  # Prepare test data
  test_data <- split_data$test_data
  test_grp <- test_data$grp
  
  # Select covariates
  covariates <- c("ALZDMT_PRIOR","HAC","LOS_DAY_CNT","PRE_DAYS","SURG_TYPE2","SURG_TYPE3","TEACH_HOSPITAL","TERT_BED_SIZE1","TERT_BED_SIZE2", 
                  "yearAdmission",  # Now a continuous variable
                  grep("^ELX_GRP_", names(data), value = TRUE))
  
  # Prepare X and y for training
  X_train <- as.matrix(train_data[,covariates])
  y_train <- train_data$y
  mean_y_train_grp <- mean(y_train[train_data$grp == 1])
  
  # Prepare X and y for testing
  X_test <- as.matrix(test_data[,covariates])
  y_test <- test_data$y
  
  # OLS Model
  model_ols <- lm(y_train ~ X_train + 1)
  beta_ols <- as.matrix(coef(model_ols))
  
  # CVXR Constrained Regression
  k <- length(beta_ols)
  beta <- Variable(k)
  loss <- sum((y_train - cbind(1,X_train) %*% beta)^2)
  
  # Constrained Regression
  prob <- Problem(Minimize(loss), 
                  list((t(train_grp) %*% (cbind(1,X_train) %*% beta)) / n_train_grp == mean_y_train_grp))
  result <- solve(prob, solver = "ECOS")
  beta_acr <- result$getValue(beta)
  
  # Predictions on test set
  pred_ols_test <- cbind(1,X_test) %*% beta_ols
  pred_acr_test <- cbind(1,X_test) %*% beta_acr
  
  # Metrics Calculation
  metrics <- list(
    ols = list(
      r2 = rsquared(y_test, pred_ols_test),
      predictive_ratio_grp = mean(pred_ols_test[test_grp==1])/mean(y_test[test_grp==1]),
      predictive_ratio_ref = mean(pred_ols_test[test_grp==0])/mean(y_test[test_grp==0]),
      mean_residual_grp = mean(pred_ols_test[test_grp==1] - y_test[test_grp==1]),
      mean_residual_ref = mean(pred_ols_test[test_grp==0] - y_test[test_grp==0]),
      mean_residual_diff = mean(pred_ols_test[test_grp==1] - y_test[test_grp==1]) - 
        mean(pred_ols_test[test_grp==0] - y_test[test_grp==0]),
      fair_covariance = as.vector(cov(test_grp, y_test-pred_ols_test))
    ),
    acr = list(
      r2 = rsquared(y_test, pred_acr_test),
      predictive_ratio_grp = mean(pred_acr_test[test_grp==1])/mean(y_test[test_grp==1]),
      predictive_ratio_ref = mean(pred_acr_test[test_grp==0])/mean(y_test[test_grp==0]),
      mean_residual_grp = mean(pred_acr_test[test_grp==1] - y_test[test_grp==1]),
      mean_residual_ref = mean(pred_acr_test[test_grp==0] - y_test[test_grp==0]),
      mean_residual_diff = mean(pred_acr_test[test_grp==1] - y_test[test_grp==1]) - 
        mean(pred_acr_test[test_grp==0] - y_test[test_grp==0]),
      fair_covariance = as.vector(cov(test_grp, y_test-pred_acr_test))
    )
  )
  
  return(metrics)
}

# Aggregate results function (remains the same as in previous script)
aggregate_results <- function(results_list) {
  # Use sapply to extract metrics efficiently
  extract_metric <- function(results_list, method, metric) {
    sapply(results_list, function(run) run[[method]][[metric]])
  }
  
  # Compute means and standard deviations for each method and metric
  summary_metrics <- list(
    ols = list(
      r2 = list(
        mean = mean(extract_metric(results_list, "ols", "r2")),
        sd = sd(extract_metric(results_list, "ols", "r2"))
      ),
      predictive_ratio_grp = list(
        mean = mean(extract_metric(results_list, "ols", "predictive_ratio_grp")),
        sd = sd(extract_metric(results_list, "ols", "predictive_ratio_grp"))
      ),
      predictive_ratio_ref = list(
        mean = mean(extract_metric(results_list, "ols", "predictive_ratio_ref")),
        sd = sd(extract_metric(results_list, "ols", "predictive_ratio_ref"))
      ),
      mean_residual_grp = list(
        mean = mean(extract_metric(results_list, "ols", "mean_residual_grp")),
        sd = sd(extract_metric(results_list, "ols", "mean_residual_grp"))
      ),
      mean_residual_ref = list(
        mean = mean(extract_metric(results_list, "ols", "mean_residual_ref")),
        sd = sd(extract_metric(results_list, "ols", "mean_residual_ref"))
      ),
      mean_residual_diff = list(
        mean = mean(extract_metric(results_list, "ols", "mean_residual_diff")),
        sd = sd(extract_metric(results_list, "ols", "mean_residual_diff"))
      ),
      fair_covariance = list(
        mean = mean(extract_metric(results_list, "ols", "fair_covariance")),
        sd = sd(extract_metric(results_list, "ols", "fair_covariance"))
      )
    ),
    acr = list(
      r2 = list(
        mean = mean(extract_metric(results_list, "acr", "r2")),
        sd = sd(extract_metric(results_list, "acr", "r2"))
      ),
      predictive_ratio_grp = list(
        mean = mean(extract_metric(results_list, "acr", "predictive_ratio_grp")),
        sd = sd(extract_metric(results_list, "acr", "predictive_ratio_grp"))
      ),
      predictive_ratio_ref = list(
        mean = mean(extract_metric(results_list, "acr", "predictive_ratio_ref")),
        sd = sd(extract_metric(results_list, "acr", "predictive_ratio_ref"))
      ),
      mean_residual_grp = list(
        mean = mean(extract_metric(results_list, "acr", "mean_residual_grp")),
        sd = sd(extract_metric(results_list, "acr", "mean_residual_grp"))
      ),
      mean_residual_ref = list(
        mean = mean(extract_metric(results_list, "acr", "mean_residual_ref")),
        sd = sd(extract_metric(results_list, "acr", "mean_residual_ref"))
      ),
      mean_residual_diff = list(
        mean = mean(extract_metric(results_list, "acr", "mean_residual_diff")),
        sd = sd(extract_metric(results_list, "acr", "mean_residual_diff"))
      ),
      fair_covariance = list(
        mean = mean(extract_metric(results_list, "acr", "fair_covariance")),
        sd = sd(extract_metric(results_list, "acr", "fair_covariance"))
      )
    )
  )
  
  return(summary_metrics)
}

# Separate function to save results to Excel (remains the same)
save_results_to_excel <- function(results, output_dir = "output") {
  # Prepare OLS results
  ols_data <- data.frame(
    Metric = names(results$ols),
    Mean = sapply(results$ols, function(x) x$mean),
    SD = sapply(results$ols, function(x) x$sd)
  )
  
  # Prepare ACR results
  acr_data <- data.frame(
    Metric = names(results$acr),
    Mean = sapply(results$acr, function(x) x$mean),
    SD = sapply(results$acr, function(x) x$sd)
  )
  
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Save Excel file with two sheets
  output_file <- file.path(output_dir, "Analysis3.xlsx")
  write_xlsx(
    list(
      OLS_Results = ols_data,
      ACR_Results = acr_data
    ),
    path = output_file
  )
  
  # Print confirmation
  cat("Results saved to:", output_file, "\n")
}

# Main Execution Script
main <- function() {
  # Start timer
  start_time <- Sys.time()
  
  # Read data
  # data <- read_sas("adrd_state.sas7bdat")
  data <- read.csv("adrd_state_032425.csv")
  
  # Define subgroup
  data$grp <- ifelse(data$AGECAT == 1 & data$SEX == 1, 1, 0)
  # data$grp <- ifelse(data$AGECAT == 1, 1, 0)
  
  # Remove incomplete cases
  na_id <- which(is.na(data$TERT_BED_SIZE) | is.na(data$HAC))
  data <- data[-na_id,]
  
  # Remove AGE_CAT = 0
  data <- data[data$AGECAT!=0,]
  
  # Feature engineering
  data$PRE_DAYS <- rowSums(data[, paste0("PRE_DAYS_AT_HOME", 1:6)], na.rm = TRUE) 
  data$SURG_TYPE2 <- ifelse(data$SURG_TYPE==2, 1, 0)
  data$SURG_TYPE3 <- ifelse(data$SURG_TYPE==3, 1, 0)
  data$TERT_BED_SIZE1 <- ifelse(data$TERT_BED_SIZE==1, 1, 0)
  data$TERT_BED_SIZE2 <- ifelse(data$TERT_BED_SIZE==2, 1, 0)
  
  # Prepare target variable
  # data$y <- rowSums(data[, paste0("POST_DAYS_AT_HOME", 1:6)], na.rm = TRUE)
  data$y <- data$SUM_DAH6MO_MODIFIED
  
  # Repeated Evaluation
  set.seed(133)
  n_times <- 50
  
  # Compute results
  results_list <- lapply(1:n_times, function(x) evaluate_regression(data))
  final_results <- aggregate_results(results_list)
  
  # End timer and calculate duration
  end_time <- Sys.time()
  duration <- difftime(end_time, start_time, units = "mins")
  
  # Print duration
  cat("Total runtime for", n_times, "replications:", round(duration, 2), "minutes\n")
  
  return(final_results)
}

# Run the analysis and save results
results <- main()
save_results_to_excel(results)

  