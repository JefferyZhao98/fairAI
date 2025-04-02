
# Required Libraries
library(haven)
library(CVXR)
library(writexl)
library(ggplot2)
library(dplyr)
library(ggrepel)

# Read data
# data <- read_sas("adrd_state.sas7bdat")
data <- read.csv("adrd_state_032425.csv")

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

# Define subgroup
grp <- ifelse(data$AGECAT == 1 & data$SEX == 1, 1, 0)
# grp <- ifelse(data$AGECAT == 1, 1, 0)

# CONTINUOUS YEAR ADMISSION
# Prepare covariates
covariates_continuous <- c("ALZDMT_PRIOR","HAC","LOS_DAY_CNT","PRE_DAYS","SURG_TYPE2","SURG_TYPE3","TEACH_HOSPITAL","TERT_BED_SIZE1","TERT_BED_SIZE2", 
                           "yearAdmission", 
                           grep("^ELX_GRP_", names(data), value = TRUE))

# Prepare X and y
X_continuous <- as.matrix(data[, covariates_continuous])
y <- data$y
n_grp <- sum(grp)
mean_y_grp <- mean(y[grp == 1])

# OLS Regression (Continuous)
model_ols_continuous <- lm(y ~ X_continuous + 1)
beta_ols_continuous <- coef(model_ols_continuous)

# CVXR Constrained Regression (Continuous)
k <- length(beta_ols_continuous)
beta <- Variable(k)
loss <- sum((y - cbind(1,X_continuous) %*% beta)^2)

prob_continuous <- Problem(Minimize(loss), 
                           list((t(grp) %*% (cbind(1,X_continuous) %*% beta)) / n_grp == mean_y_grp))
result_continuous <- solve(prob_continuous, solver = "ECOS")
beta_acr_continuous <- result_continuous$getValue(beta)

# YEAR ADMISSION DUMMY VARIABLES
# Create dummy variables for yearAdmission
year_dummies <- model.matrix(~ as.factor(yearAdmission) - 1, data = data)[, -1]
colnames(year_dummies) <- paste0("yearAdmission_201", sort(unique(data$yearAdmission))[-1])

# Prepare covariates with dummies
covariates_dummies <- c("ALZDMT_PRIOR","HAC","LOS_DAY_CNT","PRE_DAYS","SURG_TYPE2","SURG_TYPE3","TEACH_HOSPITAL","TERT_BED_SIZE1","TERT_BED_SIZE2", 
                        colnames(year_dummies), 
                        grep("^ELX_GRP_", names(data), value = TRUE))

# Prepare X with dummies
X_dummies <- as.matrix(cbind(data[, setdiff(covariates_dummies, colnames(year_dummies))], year_dummies))

# OLS Regression (Dummies)
model_ols_dummies <- lm(y ~ X_dummies + 1)
beta_ols_dummies <- coef(model_ols_dummies)

# CVXR Constrained Regression (Dummies)
k_dummies <- length(beta_ols_dummies)
beta_dummies <- Variable(k_dummies)
loss_dummies <- sum((y - cbind(1,X_dummies) %*% beta_dummies)^2)

prob_dummies <- Problem(Minimize(loss_dummies), 
                        list((t(grp) %*% (cbind(1,X_dummies) %*% beta_dummies)) / n_grp == mean_y_grp))
result_dummies <- solve(prob_dummies, solver = "ECOS")
beta_acr_dummies <- result_dummies$getValue(beta_dummies)

# Prepare results dataframes
results_continuous_df <- data.frame(
  Coefficients = c("Intercept",covariates_continuous),
  OLS = beta_ols_continuous,
  ACR = beta_acr_continuous
)

results_dummies_df <- data.frame(
  Coefficients = c("Intercept",covariates_dummies),
  OLS = beta_ols_dummies,
  ACR = beta_acr_dummies
)

# Save to Excel with two sheets
write_xlsx(
  list(
    Continuous = results_continuous_df,
    Dummies = results_dummies_df
  ), 
  "./output/Analysis4.xlsx"
)

# Plotting  

# Compute change for plotting (choose continuous or dummy yearAdmission results)
# results_df <- results_dummies_df
results_df <- results_continuous_df
results_df <- results_df[-which(results_df$Coefficients=="Intercept"),]
results_df[which(results_df$Coefficients=="HAC"),"Coefficients"] <- "Hospital-Acquired Condition"
results_df[which(results_df$Coefficients=="ALZDMT_PRIOR"),"Coefficients"] <- "Alzheimer’s Disease"
results_df[which(results_df$Coefficients=="TEACH_HOSPITAL"),"Coefficients"] <- "Teaching Hospital"
results_df[which(results_df$Coefficients=="ELX_GRP_1"),"Coefficients"] <- "Congestive Heart Failure"
results_df[which(results_df$Coefficients=="ELX_GRP_2"),"Coefficients"] <- "Cardiac Arrhythmias"
results_df[which(results_df$Coefficients=="ELX_GRP_3"),"Coefficients"] <- "Valvular Disease"
results_df[which(results_df$Coefficients=="ELX_GRP_4"),"Coefficients"] <- "Pulmonary Circulation Disorders"
results_df[which(results_df$Coefficients=="ELX_GRP_5"),"Coefficients"] <- "Peripheral Vascular Disorders"
results_df[which(results_df$Coefficients=="ELX_GRP_6"),"Coefficients"] <- "Hypertension, Uncomplicated"
results_df[which(results_df$Coefficients=="ELX_GRP_7"),"Coefficients"] <- "Hypertension, Complicated"
results_df[which(results_df$Coefficients=="ELX_GRP_8"),"Coefficients"] <- "Paralysis"
results_df[which(results_df$Coefficients=="ELX_GRP_9"),"Coefficients"] <- "Other Neurological Disorders"
results_df[which(results_df$Coefficients=="ELX_GRP_10"),"Coefficients"] <- "Chronic Pulmonary Disease"
results_df[which(results_df$Coefficients=="ELX_GRP_11"),"Coefficients"] <- "Diabetes, Uncomplicated"
results_df[which(results_df$Coefficients=="ELX_GRP_12"),"Coefficients"] <- "Diabetes, Complicated"
results_df[which(results_df$Coefficients=="ELX_GRP_13"),"Coefficients"] <- "Hypothyroidism"
results_df[which(results_df$Coefficients=="ELX_GRP_14"),"Coefficients"] <- "Renal Failure"
results_df[which(results_df$Coefficients=="ELX_GRP_22"),"Coefficients"] <- "Coagulopathy"
results_df[which(results_df$Coefficients=="ELX_GRP_24"),"Coefficients"] <- "Weight Loss"
results_df[which(results_df$Coefficients=="ELX_GRP_25"),"Coefficients"] <- "Fluid and Electrolyte Disorders"
results_df[which(results_df$Coefficients=="ELX_GRP_31"),"Coefficients"] <- "Depression"

results <- bind_rows(  
  results_df %>%  
    mutate(Method = "OLS", Change = ACR - OLS, beta = OLS),  
  results_df %>%  
    mutate(Method = "ACR", Change = ACR - OLS, beta = ACR)  
)  

# Identify largest 5 increases and decreases by percentage change or magnitude
# change arrange argument to decide which one you want to use
top_increases <- results %>%   
  filter(Change > 0) %>% mutate(Percent = (ACR - OLS) / abs(OLS) * 100) %>% 
  arrange(desc(Percent)) %>%  # Change/Percent
  slice(1:10)  

top_decreases <- results %>%   
  filter(Change < 0) %>% mutate(Percent = (ACR - OLS) / abs(OLS) * 100) %>% 
  arrange(Percent) %>%   # Change/Percent
  slice(1:10)  

# Combine for plotting  
plot_data <- bind_rows(  
  mutate(top_increases, Type = "Increase"),  
  mutate(top_decreases, Type = "Decrease")  
) %>% select(-c(OLS,ACR))
plot_data$Method <- factor(plot_data$Method, levels = c("OLS","ACR"))

# Create the plot  
ggplot(plot_data) +  
  geom_line(aes(x = Method, y = beta, group = Coefficients, colour = Type, linetype = Type), linewidth = 1) +
  geom_label_repel(data = filter(plot_data, Method == "ACR"),
            aes(label = Coefficients, x = Method, y = beta), size = 3, show.legend = FALSE, nudge_y = 0.2, nudge_x = 0.2, direction = "y",segment.size  = 0.2, force_pull = 0) +  # Adding text labels  
  scale_color_manual(values = c("Increase" = "darkgreen", "Decrease" = "lightgreen")) +  # Darker and brighter colors  
  scale_linetype_manual(values=c("twodash", "solid")) +  # Line types  
  labs(y = "Coefficients (days)") +  # Y-axis title  
  coord_cartesian(xlim=c(1.57,2), clip = "off") +
  theme(axis.title.x = element_blank(),  
        legend.position = "none",   
        panel.grid.major = element_blank(),  # Remove major grid lines  
        panel.grid.minor = element_blank(),  # Remove minor grid lines  
        panel.background = element_blank(),  # Blank background  
        axis.text.x = element_text(size = 15),  # Larger x-axis font  
        axis.text.y = element_text(size = 15, margin = margin(l=0)),  # Larger y-axis font  
        axis.title.y = element_text(size = 18, margin = margin(l=0))) # size w830 x h900


