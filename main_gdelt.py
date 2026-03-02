import pandas as pd
from utils_updated import download_csv

# to download links from Gdelt
weeks = pd.date_range(start="2016-01-01", end="2016-04-01", freq="W")
dates = weeks.strftime("%b %Y").tolist() 

for i in range(0,len(weeks)-1):
    days_param = 1
    print((weeks[i],weeks[i+1]))
    download_csv((weeks[i],weeks[i+1]), days_param)
