library(tidyverse)
library(ggthemes)
library(ranger)
library(vip)
library(caret)
library(xgboost)
library(dplyr)

# 1. Set the path to your folder

folder_path <- "march-madness-data"
file_paths <- list.files(path = folder_path, pattern = "\\.csv$", full.names = TRUE)
file_names <- tools::file_path_sans_ext(basename(file_paths))
clean_names <- gsub("[ -]", "_", file_names)
clean_names <- make.names(clean_names) 
my_data_list <- lapply(file_paths, read.csv)
names(my_data_list) <- clean_names
list2env(my_data_list, envir = .GlobalEnv)


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
  left_join(kenpom_torvik,      by = c("TEAM", "YEAR")) |>
  left_join(shooting_splits,    by = c("TEAM", "YEAR")) |>
  left_join(rppf_ratings,       by = c("TEAM", "YEAR")) |>
  left_join(resumes,            by = c("TEAM", "YEAR"))


matchups_enriched <- matchups |>
  left_join(combined_data, by = c("hSeedTeam" = "TEAM", "YEAR" = "YEAR")) |>
  rename_with(~ paste0("h_", .), .cols = !c(Game_ID, YEAR, `CURRENT ROUND`, 
                                            hSeedTeam, lSeedTeam, hSeed, lSeed,
                                            hSeedScore, lSeedScore, Winner, Loser)) |>
  left_join(combined_data, by = c("lSeedTeam" = "TEAM", "YEAR" = "YEAR"),
            suffix = c("", "_l")) |>
  rename_with(~ paste0("l_", .), .cols = ends_with("_l"))


