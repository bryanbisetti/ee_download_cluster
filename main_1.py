from utils_updated import download_in_parallel
import pandas as pd

# Load your data
df_flaring = pd.read_csv('df_flaring.csv') 

#stable-healer-488213-f9 #:150
#ee-brybisetti-cluster #150:300
#tesi-isa-1 #300:450
#peppy-center-488409-p3 #450:600
# for the cluster hardy-unison-487923-t0 #600:900


df_url = download_in_parallel(
    df_flaring.iloc[600:900],
    batch_size=30,
    max_workers=30, # The cluster can handle high concurrency
    period="W",
    project = 'hardy-unison-487923-t0'
)

