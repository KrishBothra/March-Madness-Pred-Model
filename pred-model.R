library(tidyverse)
library(dplyr)


# 1. Set the path to your folder
folder_path <- "March-Madness/march-madness-data"

# 2. Get a list of all the CSV file paths in that folder
file_paths <- list.files(path = folder_path, pattern = "\\.csv$", full.names = TRUE)

# 3. Extract just the file names (without the ".csv" part)
file_names <- tools::file_path_sans_ext(basename(file_paths))

# 4. Clean up the names (R doesn't like spaces or starting with numbers)
# This changes "538 Ratings" to "X538_Ratings" and "AP Poll Data" to "AP_Poll_Data"
clean_names <- gsub("[ -]", "_", file_names)
clean_names <- make.names(clean_names) 

# 5. Read all files into a single list of data frames using base R
# (If your files are huge, you can swap read.csv with readr::read_csv for speed)
my_data_list <- lapply(file_paths, read.csv)

# 6. Apply your clean names to the list
names(my_data_list) <- clean_names

# 7. Unpack the list into your Global Environment
# Now you will see them all pop up in your top-right RStudio pane!
list2env(my_data_list, envir = .GlobalEnv)