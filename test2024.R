# ── validate_year: leave-one-year-out cross-validation ───────────────────────
validate_year <- function(test_yr) {
  
  train_lo <- train_data_lean |> filter(model_df_clean$YEAR != test_yr)
  test_lo  <- train_data_lean |> filter(model_df_clean$YEAR == test_yr)
  
  X_train_lo <- train_lo |> select(-hSeed_won) |> as.matrix()
  X_test_lo  <- test_lo  |> select(-hSeed_won) |> as.matrix()
  y_train_lo <- ifelse(train_lo$hSeed_won == "favored", 1, 0)
  y_test_lo  <- ifelse(test_lo$hSeed_won  == "favored", 1, 0)
  
  dtrain_lo <- xgb.DMatrix(data = X_train_lo, label = y_train_lo)
  dtest_lo  <- xgb.DMatrix(data = X_test_lo,  label = y_test_lo)
  
  xgb_lo <- xgb.train(
    params = list(
      objective        = "binary:logistic",
      eval_metric      = "auc",
      eta              = 0.005,
      max_depth        = 2,
      min_child_weight = 10,
      subsample        = 0.6,
      colsample_bytree = 0.6,
      seed             = 42
    ),
    data                  = dtrain_lo,
    nrounds               = 1000,
    evals                 = list(train = dtrain_lo, test = dtest_lo),
    early_stopping_rounds = 100,
    verbose               = 0
    )
  
  lr_train <- tibble(
    xgb_prob  = predict(xgb_lo, dtrain_lo),
    seed_diff = X_train_lo[, "SEED_DIFF"],
    wab_diff  = X_train_lo[, "DIFF_WAB"],
    round     = X_train_lo[, "ROUND"],
    y         = y_train_lo
  )
  lr_test <- tibble(
    xgb_prob  = predict(xgb_lo, dtest_lo),
    seed_diff = X_test_lo[, "SEED_DIFF"],
    wab_diff  = X_test_lo[, "DIFF_WAB"],
    round     = X_test_lo[, "ROUND"]
  )
  
  lr_lo <- glm(y ~ xgb_prob + seed_diff + wab_diff + round,
               data = lr_train, family = binomial)
  
  final_probs <- predict(lr_lo, lr_test, type = "response")
  
  
  n_upsets    <- sum(y_test_lo == 0)
  upset_calls <- sum(ifelse(final_probs < 0.55, 1, 0) == 1 & y_test_lo == 0)
  
  summary_row <- tibble(
    year         = test_yr,
    games        = length(y_test_lo),
    accuracy     = round(mean(ifelse(final_probs >= 0.55, 1, 0) == y_test_lo), 3),
    upset_recall = round(ifelse(n_upsets > 0, upset_calls / n_upsets, NA), 3),
    brier        = round(mean((final_probs - y_test_lo)^2), 4),
    brier_naive  = round(mean((rep(0.70, length(y_test_lo)) - y_test_lo)^2), 4),
    beats_naive  = brier < brier_naive
  )
  
  game_results <- model_df_clean |>
    filter(YEAR == test_yr) |>
    select(hSeedTeam, lSeedTeam, hSeed, lSeed, `CURRENT ROUND`, hSeed_won) |>
    mutate(
      year          = test_yr,
      Pred_Prob     = round(final_probs, 4),
      Pred_Win      = ifelse(final_probs >= 0.55, 1, 0),
      Correct       = ifelse(Pred_Win == hSeed_won, 1, 0),
      Actual_Result = ifelse(hSeed_won == 1, "Favored Won", "Upset")
    )
  
  list(summary = summary_row, games = game_results)
}

all_years   <- model_df_clean |> distinct(YEAR) |> pull(YEAR)
all_results <- map(all_years, validate_year)

validation_summary <- map_dfr(all_results, "summary")
all_game_results   <- map_dfr(all_results, "games")

#**************************************************************************************************************************

results_2024 <- all_game_results |> filter(year == 2023)
print(results_2024)

results_2024 |>
  group_by(Correct) |> 
  summarise(
    count = n(),
  )

# ── Run for all years ─────────────────────────────────────────────────────────

print(validation_summary, n = 20)
cat("\nAvg accuracy:    ", round(mean(validation_summary$accuracy), 3), "\n")
cat("Avg upset recall:", round(mean(validation_summary$upset_recall), 3), "\n")
cat("Avg Brier:       ", round(mean(validation_summary$brier), 4), "\n")
cat("Beat naive:      ", sum(validation_summary$beats_naive),
    "out of", nrow(validation_summary), "years\n")

write.csv(validation_summary, "March-Madness/validation_summary.csv", row.names = FALSE)
write.csv(all_game_results,   "March-Madness/all_game_results.csv",   row.names = FALSE)
  
# ── Tuning grid ───────────────────────────────────────────────────────────────
# tune_grid <- expand.grid(
#   eta              = c(0.003, 0.005, 0.008),
#   max_depth        = c(2, 3),
#   min_child_weight = c(8, 10, 12),
#   subsample        = c(0.6, 0.7, 0.8),
#   colsample_bytree = c(0.6, 0.7, 0.8)
# )
# 
# cat("Total combinations:", nrow(tune_grid), "\n")
# 
# tune_results <- pmap_dfr(tune_grid, function(eta, max_depth, min_child_weight,
#                                              subsample, colsample_bytree) {
#   xgb_params <<- list(
#     objective        = "binary:logistic",
#     eval_metric      = "auc",
#     eta              = eta,
#     max_depth        = max_depth,
#     min_child_weight = min_child_weight,
#     subsample        = subsample,
#     colsample_bytree = colsample_bytree,
#     seed             = 42
#   )
#   
#   results <- map(all_years, validate_year)
#   summary <- map_dfr(results, "summary")
#   
#   tibble(
#     eta, max_depth, min_child_weight, subsample, colsample_bytree,
#     avg_brier        = round(mean(summary$brier), 4),
#     avg_accuracy     = round(mean(summary$accuracy), 3),
#     avg_upset_recall = round(mean(summary$upset_recall), 3),
#     beats_naive      = sum(summary$beats_naive)
#   )
# })
# 
# tune_results |> arrange(avg_brier) |> head(10)
# 
