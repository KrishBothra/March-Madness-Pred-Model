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
      eta              = 0.01,
      max_depth        = 3,
      min_child_weight = 8,    # was 5
      subsample        = 0.7,  # was 0.7
      colsample_bytree = 0.5,  # was 0.6
      seed             = 42,
      monotone_constraints = setNames(
        ifelse(colnames(X_train) == "DIFF_PAKE", 1, 0),
        colnames(X_train)
      )
    ),
    data                  = dtrain_lo,
    nrounds               = 1500,
    evals                 = list(train = dtrain_lo, test = dtest_lo),
    early_stopping_rounds = 25,
    verbose               = 0       # silenced — runs for every year
  )
  
  imp <- xgb.importance(model = xgb_lo)
  imp |> filter(Feature == "DIFF_PAKE")
  
  # ── Platt scaling (matches main training script) ──────────────────────────
  platt_train <- tibble(
    xgb_prob = predict(xgb_lo, dtrain_lo),
    y        = y_train_lo
  )
  platt_lo <- glm(y ~ xgb_prob, data = platt_train, family = binomial)
  
  final_probs <- predict(platt_lo,
                         tibble(xgb_prob = predict(xgb_lo, dtest_lo)),
                         type = "response")
  
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
all_years1 <- model_df_clean |> 
  distinct(YEAR) |> 
  pull(YEAR) |>
  setdiff(c(2017, 2023))
all_results <- map(all_years, validate_year)

validation_summary <- map_dfr(all_results, "summary")
all_game_results   <- map_dfr(all_results, "games")

# ── Results for a specific year ───────────────────────────────────────────────
results_2024 <- all_game_results |> filter(year == 2025)
print(results_2024)

results_2024 |>
  group_by(Correct) |>
  summarise(count = n())

# ── Summary across all years ──────────────────────────────────────────────────
print(validation_summary, n = 20)
cat("\nAvg accuracy:    ", round(mean(validation_summary$accuracy), 3), "\n")
cat("Avg upset recall:", round(mean(validation_summary$upset_recall), 3), "\n")
cat("Avg Brier:       ", round(mean(validation_summary$brier), 4), "\n")
cat("Beat naive:      ", sum(validation_summary$beats_naive),
    "out of", nrow(validation_summary), "years\n")

write.csv(validation_summary, "March-Madness/validation_summary.csv", row.names = FALSE)
write.csv(all_game_results,   "March-Madness/all_game_results.csv",   row.names = FALSE)

# tune_grid <- expand.grid(
#   eta              = c(0.005, 0.01, 0.02),
#   max_depth        = c(3, 4),
#   min_child_weight = c(3, 5, 8),
#   subsample        = c(0.6, 0.7, 0.8),
#   colsample_bytree = c(0.5, 0.6, 0.7)
# )
# 
# cat("Total combinations:", nrow(tune_grid), "\n")
# 
# tune_results <- pmap_dfr(tune_grid, function(eta, max_depth, min_child_weight,
#                                              subsample, colsample_bytree) {
# 
#   results <- map(all_years, function(test_yr) {
# 
#     train_lo <- train_data_lean |> filter(model_df_clean$YEAR != test_yr)
#     test_lo  <- train_data_lean |> filter(model_df_clean$YEAR == test_yr)
# 
#     X_train_lo <- train_lo |> select(-hSeed_won) |> as.matrix()
#     X_test_lo  <- test_lo  |> select(-hSeed_won) |> as.matrix()
#     y_train_lo <- ifelse(train_lo$hSeed_won == "favored", 1, 0)
#     y_test_lo  <- ifelse(test_lo$hSeed_won  == "favored", 1, 0)
# 
#     dtrain_lo <- xgb.DMatrix(data = X_train_lo, label = y_train_lo)
#     dtest_lo  <- xgb.DMatrix(data = X_test_lo,  label = y_test_lo)
# 
#     xgb_lo <- xgb.train(
#       params = list(
#         objective        = "binary:logistic",
#         eval_metric      = "auc",
#         eta              = eta,
#         max_depth        = max_depth,
#         min_child_weight = min_child_weight,
#         subsample        = subsample,
#         colsample_bytree = colsample_bytree,
#         seed             = 42
#       ),
#       data                  = dtrain_lo,
#       nrounds               = 1500,
#       evals                 = list(train = dtrain_lo, test = dtest_lo),
#       early_stopping_rounds = 100,
#       verbose               = 0
#     )
# 
#     platt_lo <- glm(
#       y ~ xgb_prob,
#       data   = tibble(xgb_prob = predict(xgb_lo, dtrain_lo), y = y_train_lo),
#       family = binomial
#     )
# 
#     final_probs <- predict(platt_lo,
#                            tibble(xgb_prob = predict(xgb_lo, dtest_lo)),
#                            type = "response")
# 
#     tibble(
#       brier       = mean((final_probs - y_test_lo)^2),
#       brier_naive = mean((rep(0.70, length(y_test_lo)) - y_test_lo)^2),
#       accuracy    = mean(ifelse(final_probs >= 0.55, 1, 0) == y_test_lo),
#       beats_naive = brier < brier_naive
#     )
#   })
# 
#   summary <- bind_rows(results)
# 
#   tibble(
#     eta, max_depth, min_child_weight, subsample, colsample_bytree,
#     avg_brier        = round(mean(summary$brier), 4),
#     avg_accuracy     = round(mean(summary$accuracy), 3),
#     beats_naive      = sum(summary$beats_naive),
#     avg_brier_naive  = round(mean(summary$brier_naive), 4)
#   )
# })
# 
# tune_results |> arrange(avg_brier) |> head(20) |> print()
# 
# # Best params
# best <- tune_results |> arrange(avg_brier) |> slice(1)
# cat("\nBest params:\n")
# print(best)