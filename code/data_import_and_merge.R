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
library(data.table) # for win streak calculation

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
# - box score stats by team

# for processing we also want to create the following:
# - Score gap

# overall, we should end up with team level stats by game and player level stats by game
# each dataset should contain the following:

# TeamID - a 4 digit id number, uniquely identifying each
# TeamName - a compact spelling of the team's college name

mens_regular_season_df0 <-
  MRegularSeasonDetailedResults |> 
  select(-w_loc) |> 
  # calculate the gap in score
  mutate(score_gap = w_score - l_score) |> 
  # rotate data from wide to long for easier analysis
  pivot_longer(cols = c(w_team_id, l_team_id, w_score, l_score, wfgm:lpf)
               , names_to = c("result", ".value")
               , names_pattern = "([wl])_?(.*)")  |> 
  mutate(
    result = ifelse(result == "w", 1, 0) # add binary result
    , score_gap = ifelse(result == 0, -score_gap, score_gap)  # negative for losing teams
    , fg_pct = ifelse(fga > 0, fgm / fga, NA)   # FG%
    , fg3_pct = ifelse(fga3 > 0, fgm3 / fga3, NA) # 3P%
    , ft_pct = ifelse(fta > 0, ftm / fta, NA)    # FT%
    , tourney = 0 # tournament dummy
  ) |> 
  # add overall season win percentage 
  group_by(season, team_id) |> 
  mutate(win_pct = mean(result)) |> 
  ungroup() |> 
  # add win streak
  arrange(season, day_num, team_id) |>  # ensure correct order
  group_by(team_id) |>  
  mutate(
    win_streak = ave(result, cumsum(result == 0), FUN = cumsum)  # this resets the streak once a team gets an L
    , games_played = row_number()  # count games played in the season
    , current_season_wp = cumsum(result) / games_played  # running win percentage
  ) |> 
  ungroup() 

final_regular_season_wp <- mens_regular_season_df0 |> 
  group_by(team_id, season) |> 
  summarize(prior_season_wp = max(current_season_wp, na.rm = TRUE), .groups = "drop") |> 
  mutate(season = season + 1)  |> # shift season forward to match next year
  ungroup()

# Merge prior season win percentage back into the dataset
mens_regular_season_df <- mens_regular_season_df0 |> 
  left_join(final_regular_season_wp, by = c("team_id", "season"))

# process tournament data
mens_tourney_df0 <-
  MNCAATourneyDetailedResults |> 
      select(-w_loc) |> 
      # calculate the gap in score
      mutate(score_gap = w_score - l_score) |> 
      # rotate data from wide to long for easier analysis
      pivot_longer(cols = c(w_team_id, l_team_id, w_score, l_score, wfgm:lpf)
                   , names_to = c("result", ".value")
                   , names_pattern = "([wl])_?(.*)")  |> 
      mutate(
        result = ifelse(result == "w", 1, 0) # Add binary result
        , score_gap = ifelse(result == 0, -score_gap, score_gap)  # Negative for losing teams
        , fg_pct = ifelse(fga > 0, fgm / fga, NA)   # FG%
        , fg3_pct = ifelse(fga3 > 0, fgm3 / fga3, NA) # 3P%
        , ft_pct = ifelse(fta > 0, ftm / fta, NA)    # FT%
        , tourney = 1 # tournament dummy
      ) |>  
      # add in win streak just for matrix completion purposes
      arrange(season, day_num, team_id) |>  # ensure correct order
      group_by(team_id) |>  
      mutate(
        win_streak = ave(result, cumsum(result == 0), FUN = cumsum)  # this resets the streak once a team gets an L
        , games_played = row_number()  # count games played in the season
        , current_season_wp = cumsum(result) / games_played  # running win percentage
      ) |> 
      ungroup() |> 
      # add in tournament seeding
      left_join(MNCAATourneySeeds |> select(team_id, season, seed)) |> 
  ungroup() 

# same as above
final_tourney_wp <- mens_tourney_df0 |> 
  group_by(team_id, season) |> 
  summarize(prior_season_wp = max(current_season_wp, na.rm = TRUE), .groups = "drop") |> 
  mutate(season = season + 1)  |> # shift season forward to match next year
  ungroup()

# Merge prior season win percentage back into the dataset
mens_tourney_df <- mens_tourney_df0 |> 
  left_join(final_tourney_wp, by = c("team_id", "season"))

# Bring together the regular season and tournament data frames
mens_combined_team_data <- mens_regular_season_df |> 
  bind_rows(mens_tourney_df) |> 
  # bring in coaches
  left_join(
   MTeamCoaches |> 
     # because there is more than one coach per season we need to collapse
    group_by(season, team_id) |> 
     # concatenate
    summarize(coach = paste(unique(coach_name), collapse = ", "), .groups = "drop") |> 
     ungroup() 
  ) |> 
  # bring in team names
  left_join(MTeams |> select(team_id, team_name)) |> 
  ungroup()
  


# little exploration to be deleted later
# mens_combined_team_data |> 
#   mutate(tourney = ifelse(tourney == 1, "Tourney", "Regular Season")) |> 
#   filter(result ==1) |> 
#   ggplot(aes(x = season, y = score_gap)) + 
#   stat_summary(geom = "pointrange") + stat_smooth() + 
#   facet_grid(~tourney) + usaidplot::usaid_plot() +
#   labs(x = "", y = "Score Gap")

# correlations
# cor(mens_combined_team_data[-c(30:32)], use = "complete.obs")
