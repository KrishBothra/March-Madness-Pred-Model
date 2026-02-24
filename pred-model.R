install.packages("devtools")
devtools::install_github("andreweatherman/cbbdata")
library(cbbdata)

# to register
cbbdata::cbd_create_account(username = 'kbothra', email = 'messiharden1@gmail.com', password = 'fr0zenboi', confirm_password = 'fr0zenboi')
cbbdata::cbd_login(username = 'kbothra', password = 'fr0zenboi')

CBB_Data <- cbbdata::cbd_torvik_team_schedule()
