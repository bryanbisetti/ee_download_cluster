import pandas as pd
from utils_updated import download_in_parallel


# Load your data
df_flaring = pd.read_csv('df_flaring.csv') 

# Start the download process
download_in_parallel(
    df_flaring.iloc[750:1050],
    batch_size=30,
    max_workers=30, # The cluster can handle high concurrency
    period="W"
)
