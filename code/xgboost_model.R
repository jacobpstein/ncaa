###########################################################################
# This file fits an xgboost model to the data from the data_import_and_merge.R 
# file. The final output is a data frame with the predicted win probabilities for
# every potential match up
# Session Info:
# R version 4.4.2 (2024-10-31)
# Platform: aarch64-apple-darwin20
# Running under: macOS Sequoia 15.3.2
# Author: Jacob Patterson-Stein
###########################################################################

# Set seed
set.seed(3192025)

# load libraries
library(tidyverse)
library(xgboost)

# Run our prior script
source("code/data_import_and_merge.R")

# Use all past seasons (not 2025)
train_data <- mens_combined_team_data|>
  filter(season < 2025) |> 
  select(-score_gap)

# Get Team A and B from same game
teamA_train <- train_data|>
  group_by(game_id)|>
  arrange(team_id)|>
  dplyr::slice(1)|>
  ungroup()|>
  rename_with(~ paste0(.x, "_A"), -c(game_id, team_id))

teamB_train <- train_data|>
  group_by(game_id)|>
  arrange(team_id)|>
  dplyr::slice(2)|>
  ungroup()|>
  rename_with(~ paste0(.x, "_B"), -c(game_id, team_id))

# Merge A and B into training pairs
train_pairs <- teamA_train|>
  inner_join(teamB_train, by = "game_id")

# Compute A-B diffs
all_stats <- setdiff(names(train_data), c("team_id", "game_id", "season", "team_name", "result"))
base_stats <- all_stats  

# Compute A - B differences
for (stat in base_stats) {
  train_pairs[[paste0(stat, "_diff")]] <- train_pairs[[paste0(stat, "_A")]] - train_pairs[[paste0(stat, "_B")]]
}


# Prepare training matrix
X_train <- train_pairs|> select(ends_with("_diff"), rolling_rank_A, rolling_rank_B, -rolling_rank_diff)|> as.matrix()
y_train <- train_pairs$result_A

# Create our xgboost matrix
dtrain <- xgb.DMatrix(data = X_train, label = y_train)

# parameters for the model
params <- list(
  objective = "binary:logistic"
  , eval_metric = "logloss"
  , max_depth = 6
  , eta = 0.05
  , lambda = 1
  , alpha = 0.1
  , subsample = 0.7
  , colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = params
  , data = dtrain
  , nrounds = 100
  , verbose = 0
)

# 2025-only data
team_stats_2025 <- mens_combined_team_data|>
  filter(season == 2025)|>
  select(-score_gap) |> 
  group_by(team_id)|>
  summarise(across(where(is.numeric), mean, na.rm = TRUE), .groups = "drop") 

# Add features for Team A and B
team_combos_stats <- team_combos|>
  left_join(team_stats_2025, by = c("team_id_A" = "team_id"))|>
  rename_with(~ paste0(.x, "_A"), -c(team_id_A, team_id_B))|>
  left_join(team_stats_2025, by = c("team_id_B" = "team_id"))|>
  rename_with(~ paste0(.x, "_B"), -c(team_id_A, team_id_B, ends_with("_A")))

# Compute diffs
for (stat in base_stats) {
  team_combos_stats[[paste0(stat, "_diff")]] <- team_combos_stats[[paste0(stat, "_A")]] - team_combos_stats[[paste0(stat, "_B")]]
}

# Predict
X_pred <- team_combos_stats|>
  select(ends_with("_diff"), rolling_rank_A, rolling_rank_B, -rolling_rank_diff)|>
  as.matrix()

team_combos_stats$pred_prob_teamA_win <- predict(xgb_model, X_pred)

# Final result
predicted_combos <- team_combos_stats|>
  select(team_id_A, team_id_B, pred_prob_teamA_win)

# Add in a binary label
predicted_combos$pred_label <- ifelse(predicted_combos$pred_prob > 0.5, 1, 0)

# take a peak
head(predicted_combos)

# bring in team names
team_names <- mens_combined_team_data|>
  select(team_id, team_name)|>
  distinct()

predicted_combos_named <- predicted_combos|>
  left_join(team_names, by = c("team_id_A" = "team_id"))|>
  rename(team_A = team_name)|>
  left_join(team_names, by = c("team_id_B" = "team_id"))|>
  rename(team_B = team_name)

# Take a looksie
head(predicted_combos_named)

# perform feature importance analysis 
importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance)


