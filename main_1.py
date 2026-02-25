from utils_updated import download_in_parallel_urls
import pandas as pd

# Load your data
df_not_downloaded_flaring = pd.read_csv('df_not_downloaded_flaring.csv') 


# Start the download process
download_in_parallel_urls(
    df_not_downloaded_flaring,
    batch_size=30,
    max_workers=30, # The cluster can handle high concurrency
    period="W"
)
