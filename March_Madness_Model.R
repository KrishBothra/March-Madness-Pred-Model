library(tidyverse)
library(caret)
library(xgboost)
library(dplyr)
library(vip)

# ── Step 1: Correlation filter ────────────────────────────────────────────────
diff_only  <- model_df_clean |> select(starts_with("DIFF_"))
cor_matrix <- cor(diff_only, use = "complete.obs")
high_cor   <- findCorrelation(cor_matrix, cutoff = 0.90)

diff_clean <- diff_only |> select(-all_of(colnames(diff_only)[high_cor]))

colnames(diff_clean) <- colnames(diff_clean) |>
  str_replace_all(" ", "_") |>
  str_replace_all("%", "PCT") |>
  str_replace_all("\\+", "PLUS")

# ── Step 2: Engineer new features ─────────────────────────────────────────────
model_df_clean <- model_df_clean |>
  mutate(
    DIFF_SCORING_MARGIN = DIFF_PPPO - DIFF_PPPD,
    UPSET_PROFILE = case_when(
      lSeed == 11 & DIFF_ELO < -20 ~ 1,
      lSeed == 12 & DIFF_ELO < -15 ~ 1,
      TRUE ~ 0
    )
  )

# ── Step 3: Key stats ─────────────────────────────────────────────────────────
key_stats <- c(
  # Core efficiency
  "DIFF_WAB", "DIFF_BADJ_O", "DIFF_KADJ_D", "DIFF_ELITE_SOS",
  "DIFF_ELO", "DIFF_RESUME", "DIFF_BADJ_T",
  "DIFF_NET_RPI", "DIFF_K_OFF", "DIFF_Z_RATING",
  "DIFF_Z_OFF", "DIFF_Z_DEF", "DIFF_R_SOS",
  "DIFF_PAKE",
  
  # Shooting — Four Factors
  "DIFF_EFGPCT",
  "DIFF_CLOSE_TWOS_FGPCT",
  "DIFF_THREES_FGPCT",
  "DIFF_FTR", "DIFF_FTPCT",
  "DIFF_3PTR", "DIFF_2PTR",
  "DIFF_PPPO", "DIFF_PPPD",
  
  # Four Factors continued
  "DIFF_TOV_PCT", "DIFF_TOV_PCTD",
  "DIFF_OREB_PCT", "DIFF_DREB_PCT",
  
  # Roster
  "DIFF_TALENT", "DIFF_EXP", "DIFF_AVG_HGT",
  
  # Record quality
  #"DIFF_Q1_W", "DIFF_Q3_Q4_L", "DIFF_PLUS_500",
  
  # New engineered
  "DIFF_SCORING_MARGIN"
)

# ── Step 4: Build lean training data ──────────────────────────────────────────
train_data_lean <- bind_cols(
  model_df_clean |> select(hSeed_won),
  model_df_clean |> transmute(SEED_DIFF = lSeed - hSeed),
  model_df_clean |> select(`CURRENT ROUND`),
  model_df_clean |> select(UPSET_PROFILE),
  diff_clean |> select(any_of(key_stats))
) |>
  rename(ROUND = `CURRENT ROUND`) |>
  mutate(hSeed_won = factor(hSeed_won, levels = c(0, 1),
                            labels = c("upset", "favored")))

# ── Step 5: Train/test split ──────────────────────────────────────────────────
set.seed(42)
train_idx <- createDataPartition(train_data_lean$hSeed_won, p = 0.8, list = FALSE)
train_set <- train_data_lean[train_idx, ]
test_set  <- train_data_lean[-train_idx, ]

# ── Step 6: Train XGBoost ─────────────────────────────────────────────────────
X_train <- train_set |> select(-hSeed_won) |> as.matrix()
X_test  <- test_set  |> select(-hSeed_won) |> as.matrix()
y_train <- ifelse(train_set$hSeed_won == "favored", 1, 0)
y_test  <- ifelse(test_set$hSeed_won  == "favored", 1, 0)

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

set.seed(42)
xgb_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.005,
  max_depth        = 2,
  min_child_weight = 10,
  subsample        = 0.6,
  colsample_bytree = 0.6,
  seed             = 42
)

xgb_model_lean <- xgb.train(
  params                = xgb_params,
  data                  = dtrain,
  nrounds               = 1000,
  evals                 = list(train = dtrain, test = dtest),
  early_stopping_rounds = 100,
  verbose               = 25
)

# ── Step 7: Stack XGBoost with logistic regression ───────────────────────────
xgb_probs_train <- predict(xgb_model_lean, dtrain)

lr_data_train <- tibble(
  xgb_prob  = xgb_probs_train,
  seed_diff = X_train[, "SEED_DIFF"],
  wab_diff  = X_train[, "DIFF_WAB"],
  round     = X_train[, "ROUND"],
  y         = y_train
)

lr_model <- glm(y ~ xgb_prob + seed_diff + wab_diff + round,
                data = lr_data_train, family = binomial)

# ── Step 8: Variable importance ───────────────────────────────────────────────
imp <- xgb.importance(model = xgb_model_lean)
xgb.plot.importance(imp)
#vip(xgb_model_lean, geom = "col")