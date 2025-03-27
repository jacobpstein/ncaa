library(cmdstanr)
library(tidyverse)


# Prepare data for Stan (using the "diff" columns)
# Prepare data for Stan
stan_data <- list(
  N = nrow(team_combos_stats),               # Number of matchups
  K = length(base_stats),                    # Number of features (diff columns)
  X = as.matrix(team_combos_stats %>% select(ends_with("_diff"))), # Stat diffs
  rolling_rank_A = team_combos_stats$rolling_rank_A,  # Rankings for Team A
  rolling_rank_B = team_combos_stats$rolling_rank_B,  # Rankings for Team B
  y = team_combos_stats$result_A             # Outcome (1 if Team A wins, 0 if Team B wins)
)


# Compile the model
stan_model <- cmdstan_model("code/logistic_model.stan")

# Fit the model
fit <- stan_model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

# Summarize the results
fit_summary <- fit$summary()
print(fit_summary)

# Extract the predictions
predictions <- fit$draws("y_pred")

# Convert to a data frame for further analysis
predictions_df <- as.data.frame(predictions)

# Summarize the predicted probabilities for Team A to win
summary(predictions_df)

# Plot the predicted win probabilities for Team A
ggplot(predictions_df, aes(x = y_pred)) +
  geom_histogram(bins = 30, fill = "blue", color = "white") +
  theme_minimal() +
  labs(title = "Posterior Distribution of Win Probability for Team A")

# Compare predicted probabilities with actual outcomes
predicted_probs <- rowMeans(predictions_df)  # Average predictions across chains

# Evaluate accuracy
predicted_labels <- ifelse(predicted_probs > 0.5, 1, 0)
confusion_matrix <- table(Predicted = predicted_labels, Actual = team_combos_stats_imputed$result_A)

# Print confusion matrix
print(confusion_matrix)

# Compute accuracy
accuracy <- sum(predicted_labels == team_combos_stats_imputed$result_A) / length(predicted_labels)
print(paste("Accuracy:", accuracy))



