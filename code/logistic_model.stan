data {
  int<lower=0> N;                  // number of matchups
  int<lower=0> K;                  // number of features (stat differences)
  matrix[N, K] X;                  // matrix of team stat differences
  array[N] int<lower=0, upper=1> y; // outcome (1 = Team A wins, 0 = Team B wins)
}

parameters {
  real alpha;                      // intercept
  vector[K] beta;                  // coefficients for stat differences
  real<lower=0> sigma;             // standard deviation for prior strength effect
}

model {
  // Priors
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);

  // Logistic regression model
  for (i in 1:N) {
    y[i] ~ bernoulli_logit(alpha + dot_product(X[i], beta));
  }
}

generated quantities {
  vector[N] y_pred;

  // Predict the probability of Team A winning for each matchup
  for (i in 1:N) {
    y_pred[i] = bernoulli_logit_rng(alpha + dot_product(X[i], beta));
  }
}
