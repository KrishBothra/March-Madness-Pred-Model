library(tidyverse)
library(caret)
library(xgboost)
library(dplyr)

matchups <- read.csv("march-madness-data/Tournament Matchups.csv", check.names = FALSE)

matchups <- matchups |> 
  mutate(
    Game_ID    = (row_number() + 1) %/% 2,
    Team_Label = rep(c("A","B"), length.out = n())
  ) |> 
  select(-`BY YEAR NO`, -`TEAM NO`) |> 
  pivot_wider(
    id_cols    = c(Game_ID, YEAR, `CURRENT ROUND`),
    names_from = Team_Label,
    values_from = c(TEAM, SEED, SCORE, ROUND)
  ) |> 
  mutate(
    Winner = ifelse(SCORE_A > SCORE_B, TEAM_A, TEAM_B),
    Loser  = ifelse(SCORE_A > SCORE_B, TEAM_B, TEAM_A)
  )

matchups <- matchups |> 
  mutate(
    hSeedTeam  = ifelse(SEED_A <= SEED_B, TEAM_A, TEAM_B),
    lSeedTeam  = ifelse(SEED_A <= SEED_B, TEAM_B, TEAM_A),
    hSeed      = ifelse(SEED_A <= SEED_B, SEED_A, SEED_B),
    lSeed      = ifelse(SEED_A <= SEED_B, SEED_B, SEED_A),
    hSeedScore = ifelse(SEED_A <= SEED_B, SCORE_A, SCORE_B),
    lSeedScore = ifelse(SEED_A <= SEED_B, SCORE_B, SCORE_A)
  ) |> 
  select(-TEAM_A, -TEAM_B, -SEED_A, -SEED_B, -SCORE_A, -SCORE_B)

evanMiya      <- read.csv("march-madness-data/EvanMiya.csv",           check.names = FALSE)
kenpom_torvik <- read.csv("march-madness-data/KenPom Barttorvik.csv",  check.names = FALSE)
team_results  <- read.csv("march-madness-data/Team Results.csv",       check.names = FALSE)
shooting_splits <- read.csv("march-madness-data/Shooting Splits.csv",  check.names = FALSE)
rppf_ratings  <- read.csv("march-madness-data/RPPF Ratings.csv",       check.names = FALSE)
resumes       <- read.csv("march-madness-data/Resumes.csv",            check.names = FALSE)
zrating       <- read.csv("march-madness-data/Z Rating Teams.csv",            check.names = FALSE)

zrating <- zrating |> 
  filter(TYPE == 'NEW')

combined_data <- evanMiya |>
  left_join(kenpom_torvik,   by = c("TEAM", "YEAR")) |>
  left_join(shooting_splits, by = c("TEAM", "YEAR")) |>
  left_join(rppf_ratings,    by = c("TEAM", "YEAR")) |>
  left_join(resumes,         by = c("TEAM", "YEAR")) |>
  left_join(zrating,         by = c("TEAM", "YEAR")) |>
  left_join(
    team_results |> select(TEAM, PAKE, PASE, `WIN%`, F4, CHAMP, `F4%`, `CHAMP%`),
    by = "TEAM"
  )

matchups_enriched <- matchups |>
  left_join(combined_data, by = c("hSeedTeam" = "TEAM", "YEAR" = "YEAR")) |>
  rename_with(~ paste0("h_", .), .cols = !c(Game_ID, YEAR, `CURRENT ROUND`,
                                            hSeedTeam, lSeedTeam, hSeed, lSeed,
                                            hSeedScore, lSeedScore, Winner, Loser)) |>
  left_join(combined_data, by = c("lSeedTeam" = "TEAM", "YEAR" = "YEAR"),
            suffix = c("", "_l")) |>
  rename_with(~ paste0("l_", .), .cols = ends_with("_l"))

h_stat_cols <- matchups_enriched |>
  select(starts_with("h_")) |>
  select(-matches("RANK|TEAM|SEED|ROUND|CONF|BID|QUAD|ID")) |>
  colnames()

l_stat_cols <- matchups_enriched |>
  select(-starts_with("h_")) |>
  select(-any_of(c("Game_ID","YEAR","CURRENT ROUND","Winner","Loser",
                   "hSeedTeam","lSeedTeam","hSeed","lSeed",
                   "hSeedScore","lSeedScore","l_h_L"))) |>
  select(-matches("RANK|TEAM|SEED|ROUND|CONF|BID|QUAD|ID")) |>
  select(where(is.numeric)) |>
  colnames()

h_base       <- str_remove(h_stat_cols, "^h_")
shared_stats <- intersect(h_base, l_stat_cols)

diff_df <- map_dfc(shared_stats, function(stat) {
  h_vals <- as.numeric(matchups_enriched[[paste0("h_", stat)]])
  l_vals <- as.numeric(matchups_enriched[[stat]])
  tibble(!!paste0("DIFF_", stat) := h_vals - l_vals)
})

model_df <- bind_cols(
  matchups_enriched |> select(YEAR, Game_ID, `CURRENT ROUND`,
                              hSeedTeam, lSeedTeam,
                              hSeed, lSeed,
                              hSeedScore, lSeedScore, Winner),
  diff_df
) |>
  mutate(hSeed_won = ifelse(Winner == hSeedTeam, 1L, 0L))

cols_to_drop <- c(
  "DIFF_KILL SHOTS PER GAME",
  "DIFF_KILL SHOTS CONCEDED PER GAME",
  "DIFF_TOTAL KILL SHOTS",
  "DIFF_TOTAL KILL SHOTS CONCEDED",
  "DIFF_NPB RATING"
)

model_df_clean <- model_df |>
  select(-all_of(cols_to_drop)) |>
  filter(rowSums(is.na(across(starts_with("DIFF_")))) == 0)

saveRDS(model_df_clean, "model_df_clean.rds")