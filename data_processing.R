library(tidyverse)
library(ggthemes)
library(ranger)
library(vip)
library(caret)
library(xgboost)
library(dplyr)

# 1. Set the path to your folder

#folder_path <- "march-madness-data"
#file_paths <- list.files(path = folder_path, pattern = "\\.csv$", full.names = TRUE)
#file_names <- tools::file_path_sans_ext(basename(file_paths))
#clean_names <- gsub("[ -]", "_", file_names)
#clean_names <- make.names(clean_names) 
#my_data_list <- lapply(file_paths, read.csv)
#names(my_data_list) <- clean_names
#list2env(my_data_list, envir = .GlobalEnv)


matchups <- read.csv("march-madness-data/Tournament Matchups.csv", check.names = FALSE)

unique_teams_vector <- unique(matchups$TEAM)

write.csv(unique_teams_vector, "March-Madness/unique_teams_vector.csv", row.names = FALSE)

matchups <- matchups |> 
  mutate(
    Game_ID = (row_number() + 1) %/% 2,
    Team_Label = rep(c("A","B"), length.out = n())
  ) |> 
  select(-`BY YEAR NO`, -`TEAM NO`) |> 
  pivot_wider(
    id_cols = c(Game_ID, YEAR, `CURRENT ROUND`),
    names_from = Team_Label,
    values_from = c(TEAM, SEED, SCORE, ROUND)
  ) |> 
  mutate(
    Winner = ifelse(SCORE_A > SCORE_B, TEAM_A, TEAM_B),
    Loser  = ifelse(SCORE_A > SCORE_B, TEAM_B, TEAM_A)
  )

matchups<-matchups |> 
  mutate(
    hSeedTeam = ifelse(SEED_A <= SEED_B, TEAM_A, TEAM_B),
    lSeedTeam = ifelse(SEED_A <= SEED_B, TEAM_B, TEAM_A),
    
    hSeed        = ifelse(SEED_A <= SEED_B, SEED_A, SEED_B),
    lSeed         = ifelse(SEED_A <= SEED_B, SEED_B, SEED_A),
    
    hSeedScore  = ifelse(SEED_A <= SEED_B, SCORE_A, SCORE_B),
    lSeedScore  = ifelse(SEED_A <= SEED_B, SCORE_B, SCORE_A)
  ) |> 
  select(-TEAM_A, -TEAM_B, -SEED_A, -SEED_B, -SCORE_A, -SCORE_B)

#EVANMIYA

evanMiya <- read.csv("march-madness-data/EvanMiya.csv", check.names = FALSE)

unique_teams_vector_evanMiya <- unique(evanMiya$TEAM)

write.csv(unique_teams_vector_evanMiya, "March-Madness/unique_teams_vector_evanMiya.csv", row.names = FALSE)

#KENPOM

kenpom_torvik <- read.csv("march-madness-data/KenPom Barttorvik.csv", check.names = FALSE)

unique_teams_vector_kenpom <- unique(kenpom_torvik$TEAM)

write.csv(unique_teams_vector_kenpom, "March-Madness/unique_teams_vector_kenpom.csv", row.names = FALSE)

#Team Results

team_results <- read.csv("march-madness-data/Team Results.csv", check.names = FALSE)

unique_teams_vector_team_results <- unique(team_results$TEAM)

write.csv(unique_teams_vector_team_results, "March-Madness/unique_teams_vector_team_results.csv", row.names = FALSE)

#Shooting Splits

shooting_splits <- read.csv("march-madness-data/Shooting Splits.csv", check.names = FALSE)

unique_teams_vector_shooting_splits <- unique(shooting_splits$TEAM)

write.csv(unique_teams_vector_shooting_splits, "March-Madness/unique_teams_vector_shooting_splits.csv", row.names = FALSE)

#RPPF Ratings
rppf_ratings <- read.csv("march-madness-data/RPPF Ratings.csv", check.names = FALSE)

unique_teams_vector_rppf_ratings <- unique(rppf_ratings$TEAM)

write.csv(unique_teams_vector_rppf_ratings, "March-Madness/unique_teams_vector_rppf_ratings.csv", row.names = FALSE)

#Seed Results

seed_results <- read.csv("march-madness-data/Seed Results.csv", check.names = FALSE)

#Resumes

resumes <- read.csv("march-madness-data/Resumes.csv", check.names = FALSE)

unique_teams_vector_resumes <- unique(resumes$TEAM)

write.csv(unique_teams_vector_resumes, "March-Madness/unique_teams_vector_resumes.csv", row.names = FALSE)


#*************************************************************************************************************************************

combined_data <- evanMiya |>
  left_join(kenpom_torvik,   by = c("TEAM", "YEAR")) |>
  left_join(shooting_splits, by = c("TEAM", "YEAR")) |>
  left_join(rppf_ratings,    by = c("TEAM", "YEAR")) |>
  left_join(resumes,         by = c("TEAM", "YEAR")) |>
  left_join(
    team_results |> select(TEAM, PAKE, PASE, `WIN%`, F4, CHAMP, `F4%`, `CHAMP%`),
    by = "TEAM"
  )

write.csv(combined_data, "March-Madness/combined_data.csv", row.names = FALSE)


matchups_enriched <- matchups |>
  left_join(combined_data, by = c("hSeedTeam" = "TEAM", "YEAR" = "YEAR")) |>
  rename_with(~ paste0("h_", .), .cols = !c(Game_ID, YEAR, `CURRENT ROUND`, 
                                            hSeedTeam, lSeedTeam, hSeed, lSeed,
                                            hSeedScore, lSeedScore, Winner, Loser)) |>
  left_join(combined_data, by = c("lSeedTeam" = "TEAM", "YEAR" = "YEAR"),
            suffix = c("", "_l")) |>
  rename_with(~ paste0("l_", .), .cols = ends_with("_l"))

write.csv(matchups_enriched, "March-Madness/matchups_enriched.csv", row.names = FALSE)


# glimpse(matchups_enriched |> select(1:15))
# 
# # How many games per year?
# matchups_enriched |> count(YEAR)
# 
# # Confirm the target: Winner is a team name, we need to convert to 1/0
# matchups_enriched |> select(hSeedTeam, lSeedTeam, hSeed, lSeed, Winner) |> head(10)



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

# Strip h_ prefix to get base names, then find stats present for BOTH teams
h_base <- str_remove(h_stat_cols, "^h_")
shared_stats <- intersect(h_base, l_stat_cols)

cat("h_ stat cols:", length(h_stat_cols), "\n")
cat("l_ stat cols:", length(l_stat_cols), "\n")
cat("Shared (usable) stats:", length(shared_stats), "\n")


# Compute DIFF for each of the 92 shared stats
# DIFF = higher seed stat - lower seed stat
# Positive = higher seed is better in that stat

diff_df <- map_dfc(shared_stats, function(stat) {
  h_vals <- as.numeric(matchups_enriched[[paste0("h_", stat)]])
  l_vals <- as.numeric(matchups_enriched[[stat]])
  tibble(!!paste0("DIFF_", stat) := h_vals - l_vals)
})

# Build the target variable
# hSeed_won = 1 means the better seed won (expected outcome)
# hSeed_won = 0 means the worse seed won (upset)
model_df <- bind_cols(
  matchups_enriched |> select(YEAR, Game_ID, `CURRENT ROUND`,
                              hSeedTeam, lSeedTeam, 
                              hSeed, lSeed,
                              hSeedScore, lSeedScore, Winner),
  diff_df
) |>
  mutate(hSeed_won = ifelse(Winner == hSeedTeam, 1L, 0L))

cat("Rows:", nrow(model_df), "\n")
cat("Cols:", ncol(model_df), "\n")
cat("Higher seed win rate:", round(mean(model_df$hSeed_won), 3), "\n")
cat("Missing values in diff cols:", sum(is.na(diff_df)), "\n")

na_summary <- diff_df |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  pivot_longer(everything(), names_to = "col", values_to = "na_count") |>
  filter(na_count > 0) |>
  arrange(desc(na_count))

# Also check: are the NAs concentrated in certain years?
model_df |>
  mutate(has_na = rowSums(is.na(across(starts_with("DIFF_")))) > 0) |>
  count(YEAR, has_na) |>
  filter(has_na == TRUE)

# Step 1: See exactly which years have COMPLETE stat coverage
model_df |>
  mutate(has_na = rowSums(is.na(across(starts_with("DIFF_")))) > 0) |>
  count(YEAR, has_na) |>
  print(n = 40)


model_df_clean <- model_df |>
  filter(rowSums(is.na(across(starts_with("DIFF_")))) == 0)

cat("Rows remaining:", nrow(model_df_clean), "\n")
cat("Years remaining:\n")
model_df_clean |> count(YEAR) |> print()
cat("Missing values remaining:", sum(is.na(model_df_clean |> select(starts_with("DIFF_")))), "\n")
cat("Higher seed win rate:", round(mean(model_df_clean$hSeed_won), 3), "\n")

model_df |>
  filter(YEAR >= 2013 & YEAR <= 2022) |>
  select(starts_with("DIFF_")) |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  pivot_longer(everything(), names_to = "col", values_to = "na_count") |>
  filter(na_count > 0) |>
  arrange(desc(na_count)) |>
  print(n = 100)

# Drop the 5 problematic columns
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

cat("Rows remaining:", nrow(model_df_clean), "\n")
cat("Years remaining:\n")
model_df_clean |> count(YEAR) |> print()
cat("DIFF cols remaining:", sum(startsWith(colnames(model_df_clean), "DIFF_")), "\n")
cat("Missing values:", sum(is.na(model_df_clean |> select(starts_with("DIFF_")))), "\n")
cat("Higher seed win rate:", round(mean(model_df_clean$hSeed_won), 3), "\n")

saveRDS(model_df_clean, "model_df_clean.rds")

combined_data |> 
  filter(YEAR == 2025) |> 
  select(TEAM, PAKE, PASE, Q1_W = `Q1 W`) |> 
  filter(is.na(PAKE) | is.na(Q1_W))

# See all columns in your source datasets
# cat("=== Team Results ===\n");    print(colnames(team_results))
# cat("=== Resumes ===\n");         print(colnames(resumes))
# cat("=== KenPom/Torvik ===\n");   print(colnames(kenpom_torvik))
# cat("=== EvanMiya ===\n");        print(colnames(evanMiya))
