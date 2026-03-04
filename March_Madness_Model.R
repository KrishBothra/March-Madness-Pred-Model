library(tidyverse)
library(ggthemes)
library(ranger)
library(vip)
library(caret)
library(xgboost)
library(dplyr)

# ── Step 1: Correlation filter ────────────────────────────────────────────────
diff_only <- model_df_clean |> select(starts_with("DIFF_"))

cor_matrix <- cor(diff_only, use = "complete.obs")
high_cor    <- findCorrelation(cor_matrix, cutoff = 0.90)

cat("Features before:", ncol(diff_only), "\n")
cat("Features removed (too correlated):", length(high_cor), "\n")
cat("Features remaining:", ncol(diff_only) - length(high_cor), "\n")

diff_clean <- diff_only |> select(-all_of(colnames(diff_only)[high_cor]))

colnames(diff_clean) <- colnames(diff_clean) |>
  str_replace_all(" ", "_") |>
  str_replace_all("%", "PCT") |>
  str_replace_all("\\+", "PLUS")

# ── Step 2: Define key stats BEFORE using them ────────────────────────────────
key_stats <- c(
  "DIFF_WAB",
  "DIFF_BADJ_O",
  "DIFF_KADJ_D",
  "DIFF_ELITE_SOS",
  "DIFF_TALENT",
  "DIFF_EXP",
  "DIFF_CLOSE_TWOS_FGPCT",
  "DIFF_STREM",
  "DIFF_ELO",
  "DIFF_RESUME",
  "DIFF_BADJ_T",
  "DIFF_STROE",
  "DIFF_NET_RPI",
  "DIFF_FTR",
  "DIFF_WINPCT",
  "DIFF_K_OFF",
  "DIFF_AVG_HGT",
  "DIFF_FTPCT",
  "DIFF_STRDE"
)
# Note: removed DIFF_EFG_PCT and duplicate DIFF_EXP from your original list

# ── Step 3: Verify all key stats exist ────────────────────────────────────────
missing_stats <- key_stats[!key_stats %in% colnames(diff_clean)]
cat("Missing stats:", length(missing_stats), "\n")
if(length(missing_stats) > 0) print(missing_stats)

# ── Step 4: Build lean training data ─────────────────────────────────────────
train_data_lean <- bind_cols(
  model_df_clean |> select(hSeed_won),
  model_df_clean |> transmute(SEED_DIFF = lSeed - hSeed),
  diff_clean |> select(any_of(key_stats))
) |>
  mutate(hSeed_won = factor(hSeed_won, levels = c(0,1),
                            labels = c("upset", "favored")))

cat("Features in lean model:", ncol(train_data_lean) - 1, "\n")
cat("Rows:", nrow(train_data_lean), "\n")

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
xgb_model_lean <- xgb.train(
  params = list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    eta              = 0.01,
    max_depth        = 2,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 10
  ),
  data    = dtrain,
  nrounds = 1000,
  evals   = list(train = dtrain, test = dtest),
  early_stopping_rounds = 100,
  verbose = 25
)

# ── Step 7: Evaluate ──────────────────────────────────────────────────────────
probs <- predict(xgb_model_lean, dtest)

results <- map_dfr(seq(0.3, 0.7, by = 0.05), function(t) {
  preds  <- factor(ifelse(probs > t, "favored", "upset"), levels = c("upset","favored"))
  actual <- factor(ifelse(y_test == 1, "favored", "upset"), levels = c("upset","favored"))
  cm <- confusionMatrix(preds, actual, positive = "favored")
  tibble(
    threshold    = t,
    accuracy     = cm$overall["Accuracy"],
    sensitivity  = cm$byClass["Sensitivity"],
    specificity  = cm$byClass["Specificity"],
    balanced_acc = cm$byClass["Balanced Accuracy"]
  )
})

print(results)

# ── Step 8: Variable importance ───────────────────────────────────────────────
imp <- xgb.importance(model = xgb_model_lean)
print(imp)
xgb.plot.importance(imp, top_n = 15)

# Use XGBoost probabilities as input to a logistic regression
xgb_probs_train <- predict(xgb_model_lean, dtrain)

lr_data_train <- tibble(
  xgb_prob  = xgb_probs_train,
  seed_diff = X_train[, "SEED_DIFF"],
  wab_diff  = X_train[, "DIFF_WAB"],
  y         = y_train
)

lr_data_test <- tibble(
  xgb_prob  = predict(xgb_model_lean, dtest),
  seed_diff = X_test[, "SEED_DIFF"],
  wab_diff  = X_test[, "DIFF_WAB"],
  y         = y_test
)

# Fit logistic regression on top
lr_model <- glm(y ~ xgb_prob + seed_diff + wab_diff, 
                data = lr_data_train, 
                family = binomial)

# Evaluate stacked model
stacked_probs <- predict(lr_model, lr_data_test, type = "response")

results_stacked <- map_dfr(seq(0.3, 0.7, by = 0.05), function(t) {
  preds  <- factor(ifelse(stacked_probs > t, "favored", "upset"), levels = c("upset","favored"))
  actual <- factor(ifelse(y_test == 1, "favored", "upset"),        levels = c("upset","favored"))
  cm <- confusionMatrix(preds, actual, positive = "favored")
  tibble(
    threshold    = t,
    accuracy     = cm$overall["Accuracy"],
    sensitivity  = cm$byClass["Sensitivity"],
    specificity  = cm$byClass["Specificity"],
    balanced_acc = cm$byClass["Balanced Accuracy"]
  )
})

print(results_stacked)

test_year <- 2024

train_lo <- train_data_lean |> 
  filter(model_df_clean$YEAR != test_year)

test_lo <- train_data_lean |> 
  filter(model_df_clean$YEAR == test_year)

cat("Test year games:", nrow(test_lo), "\n")
cat("Train games:", nrow(train_lo), "\n")