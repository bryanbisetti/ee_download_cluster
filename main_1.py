import pandas as pd
from utils_updated import download_in_parallel


# Load your data
df_flaring = pd.read_csv('df_flaring.csv') 

# Start the download process
download_in_parallel(
    df_flaring.iloc[:500],
    batch_size=50,
    max_workers=50, # The cluster can handle high concurrency
    period="W"
)
