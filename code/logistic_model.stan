data {
  int<lower=0> N; // Number of games
  int<lower=1> T; // Number of teams
  int<lower=1> P; // Number of players
  int<lower=0,upper=1> win[N]; // Binary outcome: Win/Loss
  int<lower=1,upper=T> team[N]; // Team index
  int<lower=1,upper=P> player[N]; // Player index
  real home_adv[N]; // Home advantage
  real xgb_prob[N]; // XGBoost probability
  real similarity[N]; // Team similarity score (0 to 1)
}

parameters {
  real alpha; // Global intercept
  vector[T] team_effect; // Team performance effect
  vector[P] player_effect; // Player performance effect
  real beta_home; // Effect of home advantage
  real beta_xgb; // Effect of XGBoost probability
  real beta_similarity; // Effect of team similarity score
  real<lower=0> sigma_team; // Team variance
  real<lower=0> sigma_player; // Player variance
}

model {
  // Priors
  alpha ~ normal(0, 5);
  team_effect ~ normal(0, sigma_team);
  player_effect ~ normal(0, sigma_player);
  beta_home ~ normal(0, 2);
  beta_xgb ~ normal(0, 2);
  beta_similarity ~ normal(0.5, 0.5); // Prior: Similarity increases win probability
  sigma_team ~ exponential(1);
  sigma_player ~ exponential(1);
  
  // Likelihood
  for (n in 1:N) {
    win[n] ~ bernoulli_logit(
      alpha + team_effect[team[n]] + player_effect[player[n]] +
      beta_home * home_adv[n] + beta_xgb * xgb_prob[n] +
      beta_similarity * similarity[n]
    );
  }
}

generated quantities {
  vector[N] win_prob;
  for (n in 1:N) {
    win_prob[n] = inv_logit(
      alpha + team_effect[team[n]] + player_effect[player[n]] +
      beta_home * home_adv[n] + beta_xgb * xgb_prob[n] +
      beta_similarity * similarity[n]
    );
  }
}