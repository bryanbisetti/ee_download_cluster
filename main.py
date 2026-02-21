import pandas as pd
from utils_updated import download_in_parallel


# Load your data
df_flaring = pd.read_csv('df_flaring.csv') 

# Start the download process
results = download_in_parallel(
    df_flaring,
    batch_size=100,
    max_workers=100, # The cluster can handle high concurrency
    period="W"
)