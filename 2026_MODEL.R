# ── Predict 2026 Matchups ─────────────────────────────────────────────────────
# Run this AFTER your full model training script (data prep + Steps 1-7).
# Requires: xgb_model_lean, lr_model, combined_data, key_stats, diff_clean, train_data_lean

# ── 1. Load the 2026 matchups CSV ─────────────────────────────────────────────
matchups_2026 <- read.csv("march-madness-data/2026_Potential_Matchups.csv",
                          check.names = FALSE)

# ── 2. Pull 2026 team stats ───────────────────────────────────────────────────
# ── Fix team name mismatches between combined_data and matchups CSV ────────────
name_fix <- c(
  "Iowa St."           = "Iowa St",
  "Ohio St."           = "Ohio St",
  "Michigan St."       = "Michigan St",
  "Saint Mary's"       = "St Mary's CA",
  "St. John's"         = "St John's",
  "Queens"             = "Queens NC",
  "Tennessee St."      = "Tennessee St",
  "North Dakota St."   = "N Dakota St",
  "Kennesaw St."       = "Kennesaw",
  "Wright St."         = "Wright St",
  "Utah St."           = "Utah St",
  "McNeese St."        = "McNeese St",
  "North Carolina St." = "NC State",
  "Saint Louis"        = "St Louis",
  "Prairie View A&M"   = "Prairie View"
)

combined_data <- combined_data |>
  mutate(TEAM = recode(TEAM, !!!name_fix))

stats_2026 <- combined_data |> filter(YEAR == 2026)
cat("Teams with 2026 stats:", nrow(stats_2026), "\n")

# ── 3. Prefix stats for higher seed (h_) and lower seed (l_) ─────────────────
h_stats_2026 <- stats_2026 |>
  rename_with(~ paste0("h_", .), .cols = -c(TEAM, YEAR))

l_stats_2026 <- stats_2026 |>
  rename_with(~ paste0("l_", .), .cols = -c(TEAM, YEAR))

matchups_2026_joined <- matchups_2026 |>
  left_join(h_stats_2026, by = c("HigherSeed" = "TEAM")) |>
  left_join(l_stats_2026, by = c("LowerSeed"  = "TEAM"))

# ── 4. Compute DIFF_ columns (h - l) for all shared stats ────────────────────
h_cols_2026 <- matchups_2026_joined |>
  select(starts_with("h_")) |>
  select(-matches("RANK|TEAM|SEED|ROUND|CONF|BID|QUAD|ID|YEAR|TYPE")) |>
  colnames()

l_cols_2026 <- matchups_2026_joined |>
  select(starts_with("l_")) |>
  select(-matches("RANK|TEAM|SEED|ROUND|CONF|BID|QUAD|ID|YEAR|TYPE")) |>
  colnames()

h_base_2026 <- str_remove(h_cols_2026, "^h_")
l_base_2026 <- str_remove(l_cols_2026, "^l_")
shared_2026 <- intersect(h_base_2026, l_base_2026)

diff_2026 <- map_dfc(shared_2026, function(stat) {
  h_vals <- as.numeric(matchups_2026_joined[[paste0("h_", stat)]])
  l_vals <- as.numeric(matchups_2026_joined[[paste0("l_", stat)]])
  tibble(!!paste0("DIFF_", stat) := h_vals - l_vals)
})

# ── 5. Add engineered features ────────────────────────────────────────────────
diff_2026 <- diff_2026 |>
  mutate(
    DIFF_SCORING_MARGIN = DIFF_PPPO - DIFF_PPPD,
    UPSET_PROFILE = case_when(
      matchups_2026$LowerSeedNum == 11 & DIFF_ELO < -20 ~ 1,
      matchups_2026$LowerSeedNum == 12 & DIFF_ELO < -15 ~ 1,
      TRUE ~ 0
    )
  )

# ── 6. Assemble feature matrix ────────────────────────────────────────────────
# ROUND = 1 is passed to both XGBoost and the LR stacker.
# All matchups in this file are potential first-round games, so this is
# correct and matches exactly how both models were trained.
surviving_diff_cols <- colnames(diff_clean)

predict_features_2026 <- bind_cols(
  tibble(SEED_DIFF = matchups_2026$LowerSeedNum - matchups_2026$HigherSeedNum),
  tibble(ROUND = 1L),
  diff_2026 |> select(UPSET_PROFILE),
  diff_2026 |> select(any_of(surviving_diff_cols)) |> select(any_of(key_stats))
) |>
  mutate(across(everything(), ~ replace_na(as.numeric(.), 0)))

# Align column order to training exactly (just drop hSeed_won)
train_feature_names <- colnames(train_data_lean |> select(-hSeed_won))

missing_cols <- setdiff(train_feature_names, colnames(predict_features_2026))
if (length(missing_cols) > 0) {
  cat("WARNING: Missing columns filled with 0:", paste(missing_cols, collapse = ", "), "\n")
  for (col in missing_cols) predict_features_2026[[col]] <- 0
}

predict_features_2026 <- predict_features_2026 |> select(all_of(train_feature_names))

# ── 7. Generate predictions ───────────────────────────────────────────────────
X_2026        <- as.matrix(predict_features_2026)
dpredict_2026 <- xgb.DMatrix(data = X_2026)

xgb_probs_2026 <- predict(xgb_model_lean, dpredict_2026)

# Apply Platt scaling instead of LR stacker
final_probs_2026 <- predict(platt_model, 
                            tibble(xgb_prob = xgb_probs_2026), 
                            type = "response")

matchups_2026$Predictions_Raw <- round(final_probs_2026, 4)


# ── 8. Injury adjustments (2026 confirmed absences) ──────────────────────────
# Adjustments applied to the HigherSeed win probability:
#   Injured HigherSeed -> subtract penalty (they're less likely to win)
#   Injured LowerSeed  -> add penalty    (higher seed now more likely to win)
# Probabilities clamped to [0.01, 0.99].
#
# Player          | Team          | Status               | Penalty
# --------------- | ------------- | -------------------- | -------
# Caleb Wilson    | North Carolina| Out for season       | -0.08
# JT Toppin       | Texas Tech    | Out (torn ACL)       | -0.10
# Aden Holloway   | Alabama       | Suspended (felony)   | -0.06
# Richie Saunders | BYU           | Out (torn ACL)       | -0.07
# Carter Welling  | Clemson       | Out                  | -0.05
# Caleb Foster    | Duke          | Out (fractured foot) | -0.04
# LJ Cason        | Michigan      | Out (torn ACL)       | -0.03

injury_map <- tribble(
  ~team,             ~penalty,
  "North Carolina",  0.08,
  "Texas Tech",      0.10,
  "Alabama",         0.06,
  "BYU",             0.07,
  "Clemson",         0.05,
  "Duke",            0.04,
  "Michigan",        0.03,
  "Louisville",      0.06
)

apply_injury_adj <- function(higher, lower, raw_prob) {
  adj <- 0
  for (i in seq_len(nrow(injury_map))) {
    team    <- injury_map$team[i]
    penalty <- injury_map$penalty[i]
    if (higher == team) adj <- adj - penalty
    if (lower  == team) adj <- adj + penalty
  }
  pmin(pmax(raw_prob + adj, 0.01), 0.99)
}

matchups_2026$Predictions <- round(
  mapply(
    apply_injury_adj,
    matchups_2026$HigherSeed,
    matchups_2026$LowerSeed,
    matchups_2026$Predictions_Raw
  ),
  4
)

# ── 9. Write output ───────────────────────────────────────────────────────────
write.csv(matchups_2026,
          "March-Madness/2026_Potential_Matchups_Predicted.csv",
          row.names = FALSE)

cat("\nDone! Predictions written to 2026_Potential_Matchups_Predicted.csv\n")

cat("\n1-seed win probabilities (sanity check):\n")
matchups_2026 |>
  filter(HigherSeedNum == 1, LowerSeedNum == 16) |>
  select(HigherSeed, LowerSeed, Predictions_Raw, Predictions) |>
  print()

cat("\nPreview of injury-adjusted matchups (raw vs adjusted):\n")
matchups_2026 |>
  filter(HigherSeed %in% injury_map$team | LowerSeed %in% injury_map$team) |>
  select(HigherSeed, HigherSeedNum, LowerSeed, LowerSeedNum,
         Predictions_Raw, Predictions) |>
  mutate(Adj = round(Predictions - Predictions_Raw, 4)) |>
  filter(Adj != 0) |>
  arrange(desc(abs(Adj))) |>
  head(20) |>
  print()

cat("\n--- Biggest potential upsets (model favors lower seed) ---\n")
matchups_2026 |>
  filter(Predictions < 0.5) |>
  arrange(Predictions) |>
  select(HigherSeed, HigherSeedNum, LowerSeed, LowerSeedNum, Predictions) |>
  head(15) |>
  print()