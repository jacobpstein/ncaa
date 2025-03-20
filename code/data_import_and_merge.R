###########################################################################
# This file implements initial data cleaning and merging
# for modeling of NCAA victories. All data come from
# https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data
# All files are stored in the data subfolder as .csv format
# Session Info:
# R version 4.4.2 (2024-10-31)
# Platform: aarch64-apple-darwin20
# Running under: macOS Sequoia 15.3.2
# Author: Jacob Patterson-Stein
###########################################################################

# Set seed
set.seed(3192025)

# Load necessary libraries
library(tidyverse) # for pretty much everything
library(janitor) # for cleaning up column names

# Load NCAA data from Kaggle---
# Get all CSV file paths
csv_files <- list.files(path = "data", pattern = "\\.csv$", full.names = TRUE)

# Read and assign each CSV file to a separate data frame based on the file name
walk(csv_files, ~ assign(
  x = tools::file_path_sans_ext(basename(.x)), 
  value = read_csv(.x) |> clean_names(), 
  envir = .GlobalEnv
))

# the data documentation can be found at the link above

# Step 1: create a data set that contains all of our key variables----
# at minimum we want 
# - seeding 
# - game outcome stats
# - box score stats by player and team

# for processing we also want to create the following:
# - Score gap
# - team continuity based on Jaccard Similarity

# overall, we should end up with team level stats by game and player level stats by game
# each dataset should contain the following:

# TeamID - a 4 digit id number, uniquely identifying each
# TeamName - a compact spelling of the team's college name

mens_combined__team_data <- 
  # start with Ws and Ls in the regular season by team
  MRegularSeasonCompactResults |> 
  # drop location column
  select(-w_loc) |> 
  # calculate the gap in score
  mutate(score_gap = w_score - l_score) |> 
  # rotate data from wide to long for easier analysis
  pivot_longer(cols = c(w_team_id, l_team_id, w_score, l_score)
               , names_to = c("result", ".value")
               , names_pattern = "([wl])_(.*)") |> 
  # create numeric, binary outcome
  mutate(result = ifelse(result == "w", 1, 0)
         # make score gape negative for losing teams
         , score_gap = ifelse(result == 0, -score_gap, score_gap)
         , tourney = 0) |> 
  # combine with tournament results
  bind_rows(
    MNCAATourneyCompactResults |> 
      # drop location
      select(-w_loc) |> 
      # calculate the gap in score
      mutate(score_gap = w_score - l_score) |> 
      # rotate data from wide to long for easier analysis
      pivot_longer(cols = c(w_team_id, l_team_id, w_score, l_score)
                   , names_to = c("result", ".value")
                   , names_pattern = "([wl])_(.*)") |> 
      # create numeric, binary outcome
      mutate(result = ifelse(result == "w", 1, 0)
             # make score gape negative for losing teams
             , score_gap = ifelse(result == 0, -score_gap, score_gap)
             , tourney = 1) 
  )


# little exploration to be deleted later
MRegularSeasonCompactResults |> 
  # drop location column
  select(-w_loc) |> 
  # calculate the gap in score
  mutate(score_gap = w_score - l_score) |> 
  # rotate data from wide to long for easier analysis
  pivot_longer(cols = c(w_team_id, l_team_id, w_score, l_score)
               , names_to = c("result", ".value")
               , names_pattern = "([wl])_(.*)") |> 
  # create numeric, binary outcome
  mutate(result = ifelse(result == "w", 1, 0)
         # make score gape negative for losing teams
         , score_gap = ifelse(result == 0, -score_gap, score_gap)) |> 
  filter(result ==1) |> 
  ggplot(aes(x = season, y = score_gap)) + 
  stat_summary(geom = "line") + stat_smooth()

