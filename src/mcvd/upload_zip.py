import wandb

# Set up your W&B account by logging in or using API key
wandb.login()

# Specify the path to your zip file
zip_file_path = "/scratch/ak11089/final-project//Unet/final_leaderboard_team_27.pt"

# Initialize a new W&B run
run = wandb.init(project="final-pred-upload")

# Specify the name you want to give to the uploaded file on W&B
uploaded_filename = "uploaded_file.zip"

# Upload the zip file using wandb.save
wandb.save(zip_file_path)

# Finish the run
run.finish()

