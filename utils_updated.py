import requests, time, ee, datetime, os, glob, rasterio, ast, re, math, pyproj, random

import pandas as pd
import numpy as np
import google.generativeai as genai
import geopandas as gpd
import statsmodels.formula.api as smf
import concurrent.futures as _cf


from tqdm import tqdm
from scipy.ndimage import median_filter
from PIL import Image
from pathlib import Path
from sklearn.neighbors import BallTree
from shapely.geometry import box
from shapely.ops import unary_union, transform as shp_transform
from rasterio.mask import mask
from datetime import timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# If you want to use your own lists:
from keywords import oil_keywords, geopolitics_keywords, examples


# One-time config (put your key in env var GEMINI_API_KEY)
import vertexai
from vertexai.generative_models import GenerativeModel


# ---- Earth Engine init (avoid re-initializing in every download call) ----
_EE_INITIALIZED = False

def init_ee(project=None):
    try:
        if project:
            ee.Initialize(project=project)   # ✅ call EE, not init_ee again
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def _to_date(x):
    # accepts "YYYY-MM", "YYYY-MM-DD", pandas Timestamp, datetime/date
    if x is None:
        return None

    if hasattr(x, "to_pydatetime"):  # pandas Timestamp
        x = x.to_pydatetime().date()

    if isinstance(x, datetime.datetime):
        return x.date()
    if isinstance(x, datetime.date):
        return x

    s = str(x)

    # "YYYY-MM" -> "YYYY-MM-01"
    if len(s) == 7 and s[4] == "-":
        s = s + "-01"

    return datetime.date.fromisoformat(s)

def month_starts(start_date, end_date):
    d0 = _to_date(start_date)
    d1 = _to_date(end_date)

    cur = datetime.date(d0.year, d0.month, 1)
    out = []
    while cur <= d1:
        out.append(cur)  # <-- store date, NOT string
        y = cur.year + (cur.month // 12)
        m = (cur.month % 12) + 1
        cur = datetime.date(y, m, 1)
    return out

def month_end(d):
    if isinstance(d, str):
        if len(d) == 7:
            d = d + "-01"
        d = datetime.date.fromisoformat(d)
    y, m = (d.year + (d.month // 12), (d.month % 12) + 1)
    return datetime.date(y, m, 1) - datetime.timedelta(days=1)

def period_starts(start_date, end_date, period: str = "M"):
    """Return list of period-start dates between start_date and end_date.
    period:
      - "M": calendar months (1st of each month)
      - "W": weeks starting Monday
    """
    d0 = _to_date(start_date)
    d1 = _to_date(end_date)
    if period.upper().startswith("M"):
        return month_starts(start_date, end_date)

    if period.upper().startswith("W"):
        # align to Monday
        cur = d0 - datetime.timedelta(days=d0.weekday())
        out = []
        while cur <= d1:
            out.append(cur)
            cur = cur + datetime.timedelta(days=7)
        return out

    raise ValueError(f"Unsupported period={period!r}. Use 'M' or 'W'.")

def period_end(start_date_dt: datetime.date, period: str = "M"):
    """Compute the inclusive end-date for a period starting at start_date_dt."""
    if period.upper().startswith("M"):
        return month_end(start_date_dt)
    if period.upper().startswith("W"):
        return start_date_dt + datetime.timedelta(days=6)
    raise ValueError(f"Unsupported period={period!r}. Use 'M' or 'W'.")

def period_tag(start_date_dt: datetime.date, period: str = "M") -> str:
    """Human-readable tag for a period."""
    if period.upper().startswith("M"):
        return start_date_dt.strftime("%Y-%m")
    if period.upper().startswith("W"):
        # ISO year-week, e.g., 2026-W05
        iso_year, iso_week, _ = start_date_dt.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    raise ValueError(f"Unsupported period={period!r}. Use 'M' or 'W'.")



# ---- small helpers to make URLs from ee.Image objects ----
def png_url(image, bands, scale, name, aoi):
    vis = image.select(bands).visualize(min=0, max=3000)
    return vis.getThumbURL({'region': aoi, 'scale': scale, 'format': 'png', 'name': name})

def geotiff_url(image, bands, scale, name, aoi):
    sel = image.select(bands)
    return sel.getDownloadURL({'region': aoi, 'scale': scale, 'format': 'GEO_TIFF', 'name': name})

def grab(url, out_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1<<20):
            if chunk: f.write(chunk)

def robust_percentile(x, q):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return np.percentile(x, q)

def download_url(location, lon, lat, start_date, end_date, period: str = "M", k_per_period: int = 3,
             radius_m: int = 1000, max_cloud_pct: int = 100, collect: str = "L1C", sleep_s: float = 0.5):
    
    # print('---------------------------------------------------------------------------------------------------------------------------------------------------')
    # print(f'Starting Download! Plant name: {location} | Coordinates: {lon}, {lat} | Date range: from {start_date} to {end_date}')
    # print('---------------------------------------------------------------------------------------------------------------------------------------------------')

    # ---- tunable parameters ----
        # ---- tunable parameters ----
    # (you can override these via function args)
    radius_m = int(radius_m)        # AOI buffer radius
    k_per_period = int(k_per_period)  # how many scenes per period (month/week)
    max_cloud_pct = float(max_cloud_pct)  # 100 = take everything; no masking
    collect = str(collect)              # 'L1C' (TOA) or 'SR' (surface reflectance)

    pt  = ee.Geometry.Point([lon, lat])
    aoi = pt.buffer(radius_m)

    # choose S2 collection (no masking)
    if collect == 'L1C':
        s2 = (ee.ImageCollection('COPERNICUS/S2')
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct)))
    else:
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct)))
        
    periods = period_starts(start_date, end_date, period=period)

    print(f'------- Starting URLs downloads for plant {location}! -------')
    rows = []
    for count, ps in enumerate(periods):
        ps = _to_date(ps)
        pe = period_end(ps, period=period)
        # overall end date (convert "YYYY-MM" to end-of-month if monthly)
        d1 = pd.to_datetime(end_date).date()
        if period == "M":
            # if your end_date is stored as month start (YYYY-MM-01), include the whole month
            d1 = (pd.Timestamp(d1) + pd.offsets.MonthEnd(0)).date()
        # clamp
        if pe > d1:
            pe = d1
        # guard: skip invalid/empty ranges
        if pe <= ps:
            continue

        ee_end = pe + timedelta(days=1)
        col_m = (s2.filterDate(ps.isoformat(), ee_end.isoformat()).sort("CLOUDY_PIXEL_PERCENTAGE"))

        n = int(col_m.size().getInfo())
        if n == 0:
            print(f'No images fount for plant {location} for period {ps} - {pe} -> ({count}/{len(periods)})')
            continue
        print(f'Number of images fount for plant {location} for period {ps} - {pe} : {n} images -> ({count}/{len(periods)})')

        k = min(k_per_period, n)
        lst = col_m.toList(n)
        for i in range(k):
            img = ee.Image(lst.get(i))
            date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            tag = period_tag(ps, period=period)
            cloud = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
            base = f"S2_{collect}_{tag}_{date_str.replace('-','')}"

            rgb = png_url(img, ['B4','B3','B2'], 10, base+'_RGB_10m', aoi)
            swr = png_url(img, ['B12','B11','B8A'], 20, base+'_SWIR_20m', aoi)
            tif = geotiff_url(img, ['B2','B3','B4','B8','B8A','B11','B12'], 20, base+'_RAW_20m', aoi)
            cir  = png_url(img, ['B8','B4','B3'], 10, base+'_CIR_10m', aoi)       # NIR/Red/Green
            b11  = png_url(img, ['B11'], 20, base+'_B11_20m', aoi)
            b12  = png_url(img, ['B12'], 20, base+'_B12_20m', aoi)

            rows.append({
                'period': tag,
                'date': date_str,
                'cloudy_pct': cloud,
                'rgb_png': rgb,
                'swir_png': swr,
                'raw_tiff': tif,
                'cir_png': cir,
                'b11': b11,
                'b12': b12
            })
    print(rows)

    save_dir = f"s2_downloads_{period}/{location}/"
    os.makedirs(save_dir, exist_ok=True)
    out_csv_urls = os.path.join(save_dir, "urls.csv")
    df_url = pd.DataFrame(rows).sort_values(['period','date']).reset_index(drop=True)
    df_url.to_csv(out_csv_urls)
    print(f'------- URLs download finished for plant {location}! -------')
    print(f"------- URLs info for plant {location} --> Total rows: {len(df_url)} | Total S2 scenes in window: {s2.size().getInfo()} -------")



def download_url_1(location, lon, lat, start_date, end_date, period: str = "M", k_per_period: int = 3,
             radius_m: int = 1000, max_cloud_pct: int = 100, collect: str = "L1C", sleep_s: float = 0.5):
    
    # init_ee(project="stable-healer-488213-f9") # old project 'gen-lang-client-0273914738'
    # print('---------------------------------------------------------------------------------------------------------------------------------------------------')
    # print(f'Starting Download! Plant name: {location} | Coordinates: {lon}, {lat} | Date range: from {start_date} to {end_date}')
    # print('---------------------------------------------------------------------------------------------------------------------------------------------------')

    # ---- tunable parameters ----
        # ---- tunable parameters ----
    # (you can override these via function args)
    radius_m = int(radius_m)        # AOI buffer radius
    k_per_period = int(k_per_period)  # how many scenes per period (month/week)
    max_cloud_pct = float(max_cloud_pct)  # 100 = take everything; no masking
    collect = str(collect)              # 'L1C' (TOA) or 'SR' (surface reflectance)

    pt  = ee.Geometry.Point([lon, lat])
    aoi = pt.buffer(radius_m)

    # choose S2 collection (no masking)
    if collect == 'L1C':
        s2 = (ee.ImageCollection('COPERNICUS/S2')
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct)))
    else:
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct)))
        
    periods = period_starts(start_date, end_date, period=period)

    print(f'------- Starting URLs downloads for plant {location}! -------')
    rows = []
    for count, ps in enumerate(periods):
        ps = _to_date(ps)
        pe = period_end(ps, period=period)
        # overall end date (convert "YYYY-MM" to end-of-month if monthly)
        d1 = pd.to_datetime(end_date).date()
        if period == "M":
            # if your end_date is stored as month start (YYYY-MM-01), include the whole month
            d1 = (pd.Timestamp(d1) + pd.offsets.MonthEnd(0)).date()
        # clamp
        if pe > d1:
            pe = d1
        # guard: skip invalid/empty ranges
        if pe <= ps:
            continue

        ee_end = pe + timedelta(days=1)
        col_m = (s2.filterDate(ps.isoformat(), ee_end.isoformat()).sort("CLOUDY_PIXEL_PERCENTAGE"))

        n = int(col_m.size().getInfo())
        if n == 0:
            print(f'No images fount for plant {location} for period {ps} - {pe} -> ({count}/{len(periods)})')
            continue
        print(f'Number of images fount for plant {location} for period {ps} - {pe} : {n} images -> ({count}/{len(periods)})')

        k = min(k_per_period, n)
        lst = col_m.toList(n)
        for i in range(k):
            img = ee.Image(lst.get(i))
            date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            tag = period_tag(ps, period=period)
            cloud = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
            base = f"S2_{collect}_{tag}_{date_str.replace('-','')}"

            rgb = png_url(img, ['B4','B3','B2'], 10, base+'_RGB_10m', aoi)
            swr = png_url(img, ['B12','B11','B8A'], 20, base+'_SWIR_20m', aoi)
            tif = geotiff_url(img, ['B2','B3','B4','B8','B8A','B11','B12'], 20, base+'_RAW_20m', aoi)
            cir  = png_url(img, ['B8','B4','B3'], 10, base+'_CIR_10m', aoi)       # NIR/Red/Green
            b11  = png_url(img, ['B11'], 20, base+'_B11_20m', aoi)
            b12  = png_url(img, ['B12'], 20, base+'_B12_20m', aoi)

            rows.append({
                'period': tag,
                'date': date_str,
                'cloudy_pct': cloud,
                'rgb_png': rgb,
                'swir_png': swr,
                'raw_tiff': tif,
                'cir_png': cir,
                'b11': b11,
                'b12': b12
            })
    print(rows)

    save_dir = f"s2_downloads_{period}/{location}/"
    os.makedirs(save_dir, exist_ok=True)
    out_csv_urls = os.path.join(save_dir, "urls.csv")
    df_url = pd.DataFrame(rows).sort_values(['period','date']).reset_index(drop=True)
    df_url.to_csv(out_csv_urls)
    print(f'------- URLs download finished for plant {location}! -------')
    print(f"------- URLs info for plant {location} --> Total rows: {len(df_url)} | Total S2 scenes in window: {s2.size().getInfo()} -------")
    
    print(f'------- Starting Image Download for plant {location}! RGB, TIF, CIR, SWIR -------')
    url_dir = f"s2_downloads_{period}/{location}/urls.csv"
    save_dir = f"s2_downloads_{period}/{location}/"
    
    df_url = pd.read_csv(url_dir)
    # Download RGB PNG and the raw GeoTIFF for each row (adjust as you like)
    for i, row in df_url.iterrows():
        print(f'Download images for plant {location}  -> ({i}/{len(df_url)})')
        base = f"{row['period']}_{row['date'].replace('-','')}"
        rgb_path = os.path.join(save_dir, base + "_RGB_10m.png")
        tif_path = os.path.join(save_dir, base + "_RAW_20m.tif")
        cir_path = os.path.join(save_dir, base + "_CIR_10m.tif")
        swir_path = os.path.join(save_dir, base + "_SWIR_20m.tif")
        grab(row["rgb_png"], rgb_path)
        grab(row["raw_tiff"], tif_path)
        grab(row["cir_png"], cir_path)
        grab(row["swir_png"], swir_path)

    print(f"Download done for plant {location}! Files in:", os.path.abspath(save_dir))

 
    

def download_images(period, location):
    print(f'------- Starting Image Download for plant {location}! RGB, TIF, CIR, SWIR -------')
    url_dir = f"s2_downloads_{period}/{location}/urls.csv"
    save_dir = f"s2_downloads_{period}/{location}/"
    
    df_url = pd.read_csv(url_dir)
    # Download RGB PNG and the raw GeoTIFF for each row (adjust as you like)
    for i, row in df_url.iterrows():
        print(f'Download images for plant {location}  -> ({i}/{len(df_url)})')
        base = f"{row['period']}_{row['date'].replace('-','')}"
        rgb_path = os.path.join(save_dir, base + "_RGB_10m.png")
        tif_path = os.path.join(save_dir, base + "_RAW_20m.tif")
        cir_path = os.path.join(save_dir, base + "_CIR_10m.tif")
        swir_path = os.path.join(save_dir, base + "_SWIR_20m.tif")
        grab(row["rgb_png"], rgb_path)
        grab(row["raw_tiff"], tif_path)
        grab(row["cir_png"], cir_path)
        grab(row["swir_png"], swir_path)

    print(f"Download done for plant {location}! Files in:", os.path.abspath(save_dir))


def make_thermal_palette():
    # Return a 256*3 list for a fire/thermal palette (black→purple→red→orange→yellow→white).
    stops = np.array([
        [0.00,   0,   0,   0],   # black
        [0.10,  35,   0,  70],   # deep purple
        [0.25, 120,   0, 120],   # magenta
        [0.40, 180,   0,   0],   # red
        [0.60, 220,  80,   0],   # orange
        [0.80, 255, 160,   0],   # amber
        [0.95, 255, 255,   0],   # yellow
        [1.00, 255, 255, 255],   # white
    ], dtype=float)

    x  = np.linspace(0, 1, 256)
    r  = np.interp(x, stops[:,0], stops[:,1])
    g  = np.interp(x, stops[:,0], stops[:,2])
    b  = np.interp(x, stops[:,0], stops[:,3])
    pal = np.vstack([r,g,b]).T.clip(0,255).astype(np.uint8).reshape(-1)
    return pal.tolist()

def to_indexed_png(arr, p_lo=2, p_hi=98, palette=None, nan_transparent=True):
    
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        idx = np.zeros_like(arr, dtype=np.uint8)
    else:
        lo, hi = np.percentile(v, [p_lo, p_hi])
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        x = np.clip((arr - lo) / (hi - lo), 0, 1)
        idx = (x * 255).astype(np.uint8)

    img = Image.fromarray(idx, mode="P")
    if palette is None:
        palette = make_thermal_palette()
    img.putpalette(palette)

    if nan_transparent:
        nan_mask = ~np.isfinite(arr)
        if nan_mask.any():
            arr_idx = np.array(img)
            arr_idx[nan_mask] = 0
            img = Image.fromarray(arr_idx, mode="P")
            img.putpalette(palette)
        img.info["transparency"] = 0

    return img

def TAI_calculation(location, period):
    save_dir = f"s2_downloads_{period}/{location}"
    b8a_min = 10.0                       # DN guard; use 0.001 if reflectance 0..1
    apply_median3=True                 # denoise TAI a bit
    tifs = sorted(glob.glob(os.path.join(save_dir, "*_RAW_20m.tif")))
    print(f"Found {len(tifs)} RAW files")

    print('----------------------------------------------------')
    print('Generating Infrared images based on TAI calculation!')
    print('----------------------------------------------------')

    for src_path in tqdm(tifs):
        base = os.path.splitext(src_path)[0]
        tai_tif = base + "_TAI_20m.tif"
        tai_png = base + "_TAI_20m.png"

        with rasterio.open(src_path) as ds:
            # 1=B2, 2=B3, 3=B4, 4=B8, 5=B8A, 6=B11, 7=B12
            B8A = ds.read(5).astype(np.float32)
            B11 = ds.read(6).astype(np.float32)
            B12 = ds.read(7).astype(np.float32)

            # per-pixel TAI with denominator guard
            denom = np.where(np.isfinite(B8A) & (B8A > b8a_min), B8A, b8a_min).astype(np.float32)
            TAI = (B12 - B11) / denom
            TAI[~(np.isfinite(B8A) & np.isfinite(B11) & np.isfinite(B12))] = np.nan

            if apply_median3:
                # keep NaNs as NaNs after filtering
                mask = np.isfinite(TAI)
                filtered = median_filter(np.nan_to_num(TAI, nan=0.0), size=3)
                counts  = median_filter(mask.astype(np.uint8), size=3)
                TAI = np.where(counts > 0, filtered, np.nan)

            # --- save GeoTIFF (float32, NaN nodata) ---
            profile = ds.profile.copy()
            profile.update({
                "count": 1,
                "dtype": "float32",
                "nodata": np.nan,
                "compress": "LZW",
                "predictor": 2,
                "tiled": True
            })
            with rasterio.open(tai_tif, "w", **profile) as out:
                out.write(TAI, 1)

            p_lo=2
            p_hi=98
            thermal_pal = make_thermal_palette()

            # ---- Save thermal-style PNG preview ----
            png_img = to_indexed_png(TAI, p_lo=p_lo, p_hi=p_hi, palette=thermal_pal, nan_transparent=True)
            png_img.save(tai_png)

    print('------ Download finished! ------')


def generate(prompt: str):
    vertexai.init(project="project-7c6ab91d-2219-4c30-98a", location="us-central1")
    model = GenerativeModel("gemini-2.5-flash")
    # Keep outputs deterministic & simple to parse
    generation_config = {
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 60000,
        # If you want strict JSON back, uncomment:
        # "response_mime_type": "application/json",
    }

    # You can also pass safety_settings here if you wish; omitted for simplicity.
    resp = model.generate_content(
        prompt,
        generation_config=generation_config,
        stream=False
    )
    # Return plain text (use resp.text). If you set response_mime_type to JSON, do json.loads(resp.text)
    return resp.text


def prompt_add_variables_call(input_list):
    with open("prompt_add_variables_1.txt", "r") as file:
        prompt = file.read()

    prompt = prompt.replace('{input_list_of_name_and_coordinates}', str(input_list))

    response = generate(prompt)
    print('------ Response: ------')
    print(response)
    print('-----------------------')

    # response_cleaned = response.strip().replace("json","").replace("```", "").replace("python\n", "").strip()
    m = re.search(r"```json(.*?)```", response, flags=re.DOTALL)
    response_cleaned = m.group(1).replace("python\n", "").replace("null", "None").replace("true", "True").replace("false", "False").strip()
    dict_add_variables = ast.literal_eval(str(response_cleaned))
    return dict_add_variables

def create_input(df):
    list_plants = []
    for i in range(len(df)):
        add = {'ID': int(df['ID'].iloc[i]), 'Plant name' : df['Plant name'].iloc[i], 'Coordinates' : (float(df['Latitude'].iloc[i]),float(df['Longitude'].iloc[i])), 'Wiki URL': df['Wiki URL'].iloc[i]}
        list_plants.append(add)
    return list_plants


def tai_stats_to_csv(location):
    """
    Scan recursively for TAI GeoTIFFs and write per-image statistics to CSV.

    Columns:
      folder, file, date(YYYYMMDD if found), valid_px, nan_px, pct_valid,
      minTAI, maxTAI, meanTAI, stdTAI, p1TAI, p5TAI, p50TAI, p95TAI, p99TAI, p99_5TAI,
      + for each threshold t: px_above_t, frac_above_t, area_m2_above_t
    """
    print('----------------------------')
    print('Calculating TAI statistics!')
    print('----------------------------')

    out_csv="tai_stats_per_image.csv"
    thresholds=(0.25, 0.30, 0.35)   # TAI thresholds to evaluate
    save_dir = f"s2_downloads/{location}"
    tifs = sorted(glob.glob(os.path.join(save_dir, "*_TAI_20m.tif")))
    if not tifs:
        print("No TAI files found.")
        return

    rows = []
    for path in tqdm(tifs):
        fname  = os.path.basename(path)

        # best-effort date parse (YYYYMMDD anywhere in the name)
        m = re.search(r"(20\d{6})", fname)
        date_tag = m.group(1) if m else None

        with rasterio.open(path) as ds:
            a = ds.read(1).astype(np.float32)     # TAI values, NaN nodata
            valid = np.isfinite(a)
            n_all   = a.size
            n_valid = int(valid.sum())
            n_nan   = int(n_all - n_valid)
            pct_valid = 100.0 * n_valid / n_all if n_all else 0.0

            if n_valid:
                v = a[valid]
                vmin = float(np.nanmin(v))
                vmax = float(np.nanmax(v))
                vmean = float(np.nanmean(v))
                vstd  = float(np.nanstd(v))
                p1, p5, p50, p95, p99, p995 = np.percentile(v, [1,5,50,95,99,99.5])
            else:
                vmin = vmax = vmean = vstd = np.nan
                p1 = p5 = p50 = p95 = p99 = p995 = np.nan

            # pixel area (m^2) if in a projected CRS (e.g., UTM). If degrees, we leave area as NaN.
            try:
                rx, ry = ds.res
                # if CRS units look like meters, area is |rx*ry|
                is_metre = (ds.crs is not None) and ("unit=m" in ds.crs.to_wkt().lower() or "metre" in ds.crs.to_wkt().lower())
                pix_area_m2 = abs(rx * ry) if is_metre else np.nan
            except Exception:
                pix_area_m2 = np.nan

            rec = {
                "file": fname,
                "date": date_tag,
                "valid_px": n_valid,
                "nan_px": n_nan,
                "pct_valid": round(pct_valid, 2),
                "minTAI": round(vmin, 6) if np.isfinite(vmin) else np.nan,
                "maxTAI": round(vmax, 6) if np.isfinite(vmax) else np.nan,
                "meanTAI": round(vmean, 6) if np.isfinite(vmean) else np.nan,
                "stdTAI": round(vstd, 6) if np.isfinite(vstd) else np.nan,
                "p1TAI":  round(p1, 6) if np.isfinite(p1) else np.nan,
                "p5TAI":  round(p5, 6) if np.isfinite(p5) else np.nan,
                "p50TAI": round(p50, 6) if np.isfinite(p50) else np.nan,
                "p95TAI": round(p95, 6) if np.isfinite(p95) else np.nan,
                "p99TAI": round(p99, 6) if np.isfinite(p99) else np.nan,
                "p99_5TAI": round(p995, 6) if np.isfinite(p995) else np.nan,
                "pixel_area_m2": pix_area_m2 if np.isfinite(pix_area_m2) else np.nan,
            }

            # threshold metrics
            for t in thresholds:
                if n_valid:
                    n_above = int((a > t).sum())
                    frac = n_above / n_valid if n_valid else 0.0
                    area = n_above * pix_area_m2 if np.isfinite(pix_area_m2) else np.nan
                else:
                    n_above = 0
                    frac = 0.0
                    area = np.nan
                rec[f"px_above_{t}"]   = n_above
                rec[f"frac_above_{t}"] = round(frac, 6)
                rec[f"area_m2_above_{t}"] = area if (area is np.nan or np.isfinite(area)) else np.nan

            rows.append(rec)

    out_dir = os.path.join(save_dir, out_csv)
    df = pd.DataFrame(rows).sort_values(["date","file"]).reset_index(drop=True)
    df.to_csv(out_dir, index=False)
    print(f"------ Saved {len(df)} rows to {out_dir} ------")
    return df



# -----------------------------
# Helpers
# -----------------------------
def _find_col(cols, patterns):
    for pat in patterns:
        r = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if r.search(str(c)):
                return c
    return None

def infer_years_from_filename(path):
    name = Path(path).name
    # range like 2012-2016
    m = re.search(r"((19|20)\d{2})\s*[-_]\s*((19|20)\d{2})", name)
    if m:
        y0, y1 = int(m.group(1)), int(m.group(3))
        if y0 <= y1:
            return list(range(y0, y1 + 1))
    # single year
    m = re.search(r"(19|20)\d{2}", name)
    return [int(m.group(0))] if m else []

def clean_coords(d, lat_col="Latitude", lon_col="Longitude"):
    d = d.copy()
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")

    # fix obvious swaps
    swap = (d[lat_col].abs() > 90) & (d[lon_col].abs() <= 90)
    d.loc[swap, [lat_col, lon_col]] = d.loc[swap, [lon_col, lat_col]].values
    return d

# -----------------------------
# Load VIIRS flare points from a folder
# -----------------------------
def load_viirs_folder_points(viirs_folder="viirs", verbose=True):
    """
    Loads ALL .xlsx in viirs_folder, finds the sheet containing lat/lon (flare points),
    and concatenates them into one flare-point df: flare_lat, flare_lon, year.
    Skips files that contain only country summaries (no coordinates).
    """
    paths = sorted(glob.glob(str(Path(viirs_folder) / "*.xlsx")))
    if not paths:
        raise FileNotFoundError(f"No .xlsx files found in: {viirs_folder}")

    all_rows = []
    skipped = []

    for p in paths:
        years = infer_years_from_filename(p)
        if not years:
            skipped.append((p, "cannot infer year from filename"))
            continue

        xls = pd.ExcelFile(p)
        found = False

        for sh in xls.sheet_names:
            tmp = pd.read_excel(p, sheet_name=sh)

            lat_col = _find_col(tmp.columns, [r"^latitude$", r"^lat$", r"lat"])
            lon_col = _find_col(tmp.columns, [r"^longitude$", r"^lon$", r"long", r"lon"])

            if lat_col is None or lon_col is None:
                continue  # not the point sheet

            pts = pd.DataFrame({
                "flare_lat": pd.to_numeric(tmp[lat_col], errors="coerce"),
                "flare_lon": pd.to_numeric(tmp[lon_col], errors="coerce"),
            }).dropna(subset=["flare_lat", "flare_lon"])

            if pts.empty:
                continue

            # duplicate points for each year in the file (needed for 2012-2016)
            for y in years:
                yy = pts.copy()
                yy["year"] = y
                all_rows.append(yy)

            found = True
            if verbose:
                print(f"Loaded points from: {Path(p).name} | sheet='{sh}' | years={years} | n={len(pts)}")
            break

        if not found:
            # this is exactly your 2018 file case: it is a summary table with no coords
            skipped.append((p, "no lat/lon point sheet found (likely country summary)"))

    if not all_rows:
        msg = "No flare-point coordinates found in any Excel file. You likely downloaded summary tables.\n"
        msg += "Fix: use the KML downloads or the Excel files that include flare site coordinates."
        raise ValueError(msg)

    flares_all = pd.concat(all_rows, ignore_index=True).drop_duplicates().reset_index(drop=True)

    if verbose and skipped:
        print("\nSkipped files:")
        for p, reason in skipped:
            print(f" - {Path(p).name}: {reason}")

    return flares_all

# -----------------------------
# Plant active years
# -----------------------------
def add_active_years(plants_df, start_col="Start_production", end_col="End_production"):
    d = plants_df.copy()
    d[start_col] = pd.to_datetime(d[start_col], errors="coerce")
    d[end_col] = pd.to_datetime(d[end_col], errors="coerce")
    d["start_year"] = d[start_col].dt.year
    d["end_year"] = d[end_col].dt.year
    return d

# -----------------------------
# Label flaring during active period
# -----------------------------
def label_flaring_active_period(plants_df, flares_df, radius_km=3.0, min_years_active=1,
                                plant_name_col="Plant name",
                                lat_col="Latitude", lon_col="Longitude"):

    EARTH_RADIUS_KM = 6371.0088

    plants = clean_coords(plants_df, lat_col, lon_col).copy()
    plants = plants.dropna(subset=[lat_col, lon_col, "start_year"]).reset_index(drop=True)

    flares = flares_df.dropna(subset=["flare_lat","flare_lon","year"]).copy().reset_index(drop=True)

    # Fill missing end_year with latest VIIRS year
    max_viirs_year = int(flares["year"].max())
    plants["end_year"] = plants["end_year"].fillna(max_viirs_year).astype(int)
    plants["start_year"] = plants["start_year"].astype(int)

    flare_rad = np.deg2rad(np.c_[flares["flare_lat"].values, flares["flare_lon"].values])
    plant_rad = np.deg2rad(np.c_[plants[lat_col].values, plants[lon_col].values])

    tree = BallTree(flare_rad, metric="haversine")
    r = radius_km / EARTH_RADIUS_KM

    ind, dist = tree.query_radius(plant_rad, r=r, return_distance=True, sort_results=True)

    plants["nearest_flare_km"] = [
        (ds[0] * EARTH_RADIUS_KM) if len(ds) else np.inf for ds in dist
    ]
    plants["flare_points_within_radius_anyyear"] = [len(ix) for ix in ind]

    active_year_counts = []
    active_points_counts = []

    for i, ix in enumerate(ind):
        if len(ix) == 0:
            active_year_counts.append(0)
            active_points_counts.append(0)
            continue

        sy = plants.loc[i, "start_year"]
        ey = plants.loc[i, "end_year"]

        yrs = flares.loc[ix, "year"].values
        mask = (yrs >= sy) & (yrs <= ey)

        active_points_counts.append(int(mask.sum()))
        active_year_counts.append(int(len(np.unique(yrs[mask]))))

    plants["flare_points_within_radius_active"] = active_points_counts
    plants["flare_years_count_active"] = active_year_counts
    plants["flaring_in_active_period"] = plants["flare_years_count_active"] >= int(min_years_active)

    return plants[[plant_name_col,
                   "flaring_in_active_period",
                   "flare_years_count_active",
                   "flare_points_within_radius_active",
                   "nearest_flare_km",
                   "flare_points_within_radius_anyyear"]]

# ROI

def parse_date(path):
    """
    Example: 2022-01_20220109_RAW_20m_TAI_20m.tif
    Returns (month='2022-01', date='20220109') if found, else (None, None)
    """
    base = os.path.basename(path)
    m = re.search(r"(?P<date>\d{8})", base)
    if not m:
        return None, None
    date = m.group("date")
    return date[:4], date[4:6]


def utm_epsg_from_lonlat(lon, lat):    
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def build_rois_projected(plants_gdf_wgs84, raster_crs, r_sig=1000, r_bg_in=1500, r_bg_out=3000, raster_bounds=None):
    """
    Builds ROI polygons in raster_crs, assuming raster_crs is projected (meters).
    Returns a GeoDataFrame with roi_sig and roi_bg_clean as shapely polygons (NOT as geometry).
    """
    g = plants_gdf_wgs84.to_crs(raster_crs).copy()

    g["roi_sig"] = g.geometry.buffer(r_sig)
    bg_outer = g.geometry.buffer(r_bg_out)
    bg_inner = g.geometry.buffer(r_bg_in)
    g["roi_bg"] = bg_outer.difference(bg_inner)

    # Remove any signal ROI from background ring to avoid contaminating background with hot pixels
    union_sig = unary_union(g["roi_sig"].values)
    g["roi_bg_clean"] = g["roi_bg"].difference(union_sig)

    # Optional clip to raster bounds
    if raster_bounds is not None:
        raster_box = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)
        g["roi_sig"] = g["roi_sig"].intersection(raster_box)
        g["roi_bg_clean"] = g["roi_bg_clean"].intersection(raster_box)

    return g


def build_rois_geographic(plants_gdf_wgs84, out_crs, r_sig=1000, r_bg_in=1500, r_bg_out=3000, raster_bounds=None):
    """
    If raster CRS is geographic (degrees), buffer in local UTM meters per point,
    then reproject polygons to out_crs.
    """
    g = plants_gdf_wgs84.to_crs("EPSG:4326").copy()
    crs_wgs84 = pyproj.CRS("EPSG:4326")
    crs_out   = pyproj.CRS.from_user_input(out_crs)

    roi_sig_list, roi_bg_list = [], []

    for pt in g.geometry:
        lon, lat = pt.x, pt.y
        epsg_utm = utm_epsg_from_lonlat(lon, lat)
        crs_utm  = pyproj.CRS.from_epsg(epsg_utm)

        wgs84_to_utm = pyproj.Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True).transform
        utm_to_out   = pyproj.Transformer.from_crs(crs_utm, crs_out,   always_xy=True).transform

        pt_utm = shp_transform(wgs84_to_utm, pt)
        sig_utm = pt_utm.buffer(r_sig)
        ring_utm = pt_utm.buffer(r_bg_out).difference(pt_utm.buffer(r_bg_in))

        roi_sig_list.append(shp_transform(utm_to_out, sig_utm))
        roi_bg_list.append(shp_transform(utm_to_out, ring_utm))

    g_out = g.to_crs(out_crs).copy()
    g_out["roi_sig"] = roi_sig_list
    g_out["roi_bg"] = roi_bg_list

    union_sig = unary_union(g_out["roi_sig"].values)
    g_out["roi_bg_clean"] = g_out["roi_bg"].difference(union_sig)

    if raster_bounds is not None:
        raster_box = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)
        g_out["roi_sig"] = g_out["roi_sig"].intersection(raster_box)
        g_out["roi_bg_clean"] = g_out["roi_bg_clean"].intersection(raster_box)

    return g_out


def rois_long_format(g_rois, tif_path, month, year):
    """
    Convert wide ROI columns into a long GeoDataFrame:
    one row per (plant, tif, roi_type) with geometry set accordingly.
    """
    rows = []

    for _, r in g_rois.iterrows():
        base_meta = {
            "Plant name": r.get("Plant name"),
            "tif_path": tif_path,
            "month": month,
            "year": year,
        }

        rows.append({**base_meta, "roi_type": "signal", "geometry": r["roi_sig"]})
        rows.append({**base_meta, "roi_type": "background", "geometry": r["roi_bg_clean"]})

    g_long = gpd.GeoDataFrame(rows, geometry="geometry", crs=g_rois.crs)
    return g_long


def extract_proxy_from_tif(tif_path, roi_sig_geom, roi_bg_geom, rois_crs="EPSG:4326", top_q=0.95):
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        # --- reproject ROIs to raster CRS ---
        sig_gdf = gpd.GeoDataFrame(geometry=[roi_sig_geom], crs=rois_crs).to_crs(raster_crs)
        bg_gdf  = gpd.GeoDataFrame(geometry=[roi_bg_geom],  crs=rois_crs).to_crs(raster_crs)

        roi_sig = sig_gdf.geometry.iloc[0]
        roi_bg  = bg_gdf.geometry.iloc[0]

        # Safety: if still no overlap, skip
        if roi_sig.is_empty or roi_bg.is_empty:
            return {"tif_path": tif_path, "n_sig": 0, "n_bg": 0,
                    "bg_med": np.nan, "p99_sig": np.nan, "tail_mean": np.nan, "tail_sum": np.nan}

        # --- mask ---
        sig_arr, _ = mask(src, [roi_sig], crop=True, filled=True, nodata=np.nan)
        bg_arr, _  = mask(src, [roi_bg],  crop=True, filled=True, nodata=np.nan)

    sig = sig_arr[0].astype("float64")
    bg  = bg_arr[0].astype("float64")
    if nodata is not None:
        sig[sig == nodata] = np.nan
        bg[bg == nodata] = np.nan

    sig_v = sig[np.isfinite(sig)]
    bg_v  = bg[np.isfinite(bg)]

    out = {"tif_path": tif_path, "n_sig": int(sig_v.size), "n_bg": int(bg_v.size)}

    if sig_v.size == 0 or bg_v.size == 0:
        out.update({"bg_med": np.nan, "p99_sig": np.nan, "tail_mean": np.nan, "tail_sum": np.nan})
        return out

    bg_med = np.nanmedian(bg_v)
    out["bg_med"] = float(bg_med)
    out["p99_sig"] = float(np.nanpercentile(sig_v, 99))

    excess = sig_v - bg_med
    excess = excess[excess > 0]

    if excess.size == 0:
        out["tail_mean"] = 0.0
        out["tail_sum"]  = 0.0
        return out

    thr = np.nanquantile(excess, top_q)
    tail = excess[excess >= thr]

    out["tail_mean"] = float(np.nanmean(tail))
    out["tail_sum"]  = float(np.nansum(tail))
    return out


def extract_proxy_from_tif(tif_path, roi_sig_geom, roi_bg_geom, rois_crs="EPSG:4326", top_q=0.95):
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        # --- reproject ROIs to raster CRS ---
        sig_gdf = gpd.GeoDataFrame(geometry=[roi_sig_geom], crs=rois_crs).to_crs(raster_crs)
        bg_gdf  = gpd.GeoDataFrame(geometry=[roi_bg_geom],  crs=rois_crs).to_crs(raster_crs)

        roi_sig = sig_gdf.geometry.iloc[0]
        roi_bg  = bg_gdf.geometry.iloc[0]

        # Safety: if still no overlap, skip
        if roi_sig.is_empty or roi_bg.is_empty:
            return {"tif_path": tif_path, "n_sig": 0, "n_bg": 0,
                    "bg_med": np.nan, "p99_sig": np.nan, "tail_mean": np.nan, "tail_sum": np.nan}

        # --- mask ---
        sig_arr, _ = mask(src, [roi_sig], crop=True, filled=True, nodata=np.nan)
        bg_arr, _  = mask(src, [roi_bg],  crop=True, filled=True, nodata=np.nan)

    sig = sig_arr[0].astype("float64")
    bg  = bg_arr[0].astype("float64")
    if nodata is not None:
        sig[sig == nodata] = np.nan
        bg[bg == nodata] = np.nan

    sig_v = sig[np.isfinite(sig)]
    bg_v  = bg[np.isfinite(bg)]

    out = {"tif_path": tif_path, "n_sig": int(sig_v.size), "n_bg": int(bg_v.size)}

    if sig_v.size == 0 or bg_v.size == 0:
        out.update({"bg_med": np.nan, "p99_sig": np.nan, "tail_mean": np.nan, "tail_sum": np.nan})
        return out

    bg_med = np.nanmedian(bg_v)
    out["bg_med"] = float(bg_med)
    out["p99_sig"] = float(np.nanpercentile(sig_v, 99))

    excess = sig_v - bg_med
    excess = excess[excess > 0]

    if excess.size == 0:
        out["tail_mean"] = 0.0
        out["tail_sum"]  = 0.0
        return out

    thr = np.nanquantile(excess, top_q)
    tail = excess[excess >= thr]

    out["tail_mean"] = float(np.nanmean(tail))
    out["tail_sum"]  = float(np.nansum(tail))
    return out


def parse_month_date(path):
    m = re.search(r"(?P<month>\d{4}-\d{2})_(?P<date>\d{8})", path)
    if not m:
        return pd.Series({"month_str": None, "date_str": None})
    return pd.Series({"month_str": m.group("month"), "date_str": m.group("date")})


def nasa_power_daily_t2m(lat, lon, start_yyyymmdd, end_yyyymmdd):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN",
        "community": "RE",
        "latitude": lat,
        "longitude": lon,
        "start": start_yyyymmdd,
        "end": end_yyyymmdd,
        "format": "JSON",
    }
    js = requests.get(url, params=params, timeout=60).json()
    d = js["properties"]["parameter"]
    df = pd.DataFrame({
        "date": pd.to_datetime(list(d["T2M"].keys())),
        "t2m": list(d["T2M"].values()),
        "t2m_max": list(d["T2M_MAX"].values()),
        "t2m_min": list(d["T2M_MIN"].values()),
    })
    return df


# ----------------------------
# 0) Prep: ensure required columns + transforms
# ----------------------------
def prep_panel(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Required columns in monthly_df:
      Plant name, month, OilProxy_raw, TempNorm, Age_years, Extension, Capacity (MW), Machinery, Region
    """
    d = monthly_df.copy()

    d["month"] = pd.to_datetime(d["month"])
    d["Capacity_MW"] = pd.to_numeric(d["Capacity_MW"], errors="coerce")
    d["Extension"]   = pd.to_numeric(d["Extension"], errors="coerce")
    d["Age_years"]   = pd.to_numeric(d["Age_years"], errors="coerce")
    d["TempNorm"]    = pd.to_numeric(d["TempNorm"], errors="coerce")

    d["Machinery"] = d["Machinery"].astype(str).str.strip().str.lower()
    d["Region"] = d["Region"].astype(str).str.strip()

    d["log_proxy"] = np.log1p(pd.to_numeric(d["OilProxy_raw"], errors="coerce"))
    d["log_ext"] = np.log(d["Extension"].where(d["Extension"] > 0))
    d["log_cap"] = np.log(d["Capacity_MW"].where(d["Capacity_MW"] > 0))

    # keep only usable rows
    d = d.dropna(subset=[
        "Plant name","month","log_proxy","TempNorm","Age_years","log_ext","log_cap","Machinery","Region"
    ]).copy()

    return d


# ----------------------------
# 1) Temperature validation: persistent sites should show minimal seasonality after adjustment
# ----------------------------
def select_persistent(df: pd.DataFrame, min_months=9, top_q=0.80) -> pd.DataFrame:
    counts = df.groupby("Plant name")["month"].nunique()
    eligible = counts[counts >= min_months].index
    tmp = df[df["Plant name"].isin(eligible)].copy()

    mproxy = tmp.groupby("Plant name")["log_proxy"].mean()
    persistent = mproxy[mproxy >= mproxy.quantile(top_q)].index
    return tmp[tmp["Plant name"].isin(persistent)].copy()

def seasonality_r2_by_moy(df: pd.DataFrame, value_col: str) -> float:
    """
    For each plant: regress value ~ month-of-year dummies, return mean R^2 across plants.
    Lower is better (less seasonality).
    """
    d = df.copy()
    d["moy"] = d["month"].dt.month.astype(int)

    r2s = []
    for plant, g in d.groupby("Plant name"):
        if g["moy"].nunique() < 6 or len(g) < 6:
            continue
        fit = smf.ols(f"{value_col} ~ C(moy)", data=g).fit()
        r2s.append(fit.rsquared)
    return float(np.mean(r2s)) if r2s else np.inf

def calibrate_beta_T(df: pd.DataFrame, grid=None) -> float:
    """
    Grid-search beta_T to minimize seasonality on persistent plants.
    """
    if grid is None:
        grid = np.linspace(-2.0, 2.0, 161)

    persist = select_persistent(df)
    best_b, best_s = None, np.inf

    for b in grid:
        tmp = persist.copy()
        tmp["log_T_adj"] = tmp["log_proxy"] - b * tmp["TempNorm"]
        s = seasonality_r2_by_moy(tmp, "log_T_adj")
        if s < best_s:
            best_s, best_b = s, b

    return float(best_b)


# ----------------------------
# 2) Distribution shift metric without SciPy (quantile distance)
# ----------------------------
def quantile_distance(x, y, qs=np.linspace(0.05, 0.95, 19)) -> float:
    x = np.asarray(x); y = np.asarray(y)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan
    return float(np.mean(np.abs(np.quantile(x, qs) - np.quantile(y, qs))))

def month_to_month_shift(df: pd.DataFrame, value_col: str, bin_col: str) -> float:
    """
    Average quantile-distance between consecutive months *within each bin*.
    Lower is better.
    """
    d = df.sort_values("month").copy()

    total, n = 0.0, 0
    for b, gb in d.groupby(bin_col):
        months = sorted(gb["month"].unique())  # IMPORTANT: months per-bin, not global
        if len(months) < 3:
            continue

        for m1, m2 in zip(months[:-1], months[1:]):
            x = gb.loc[gb["month"] == m1, value_col].values
            y = gb.loc[gb["month"] == m2, value_col].values
            dist = quantile_distance(x, y)
            if np.isfinite(dist):
                total += dist
                n += 1

    return total / n if n else np.inf



# ----------------------------
# 3) Calibrate Age/Extension/Capacity (sequential) to minimize shifts within bins
# ----------------------------
def add_bins(df: pd.DataFrame, q=3) -> pd.DataFrame:
    d = df.copy()
    d["age_bin"] = pd.qcut(d["Age_years"], q=q, duplicates="drop")
    d["ext_bin"] = pd.qcut(d["log_ext"], q=q, duplicates="drop")
    d["cap_bin"] = pd.qcut(d["log_cap"], q=q, duplicates="drop")
    return d

def apply_continuous_adjustment(df, beta_T, beta_age, beta_ext, beta_cap):
    out = df.copy()
    out["log_adj_cont"] = (
        out["log_proxy"]
        - beta_T  * out["TempNorm"]
        - beta_age * out["Age_years"]
        - beta_ext * out["log_ext"]
        - beta_cap * out["log_cap"]
    )
    return out

def calibrate_beta_continuous(df: pd.DataFrame, beta_T: float, q_bins=3,
                              grid_age=None, grid_ext=None, grid_cap=None):
    d = add_bins(df, q=q_bins)

    if grid_age is None: grid_age = np.linspace(-0.5, 0.5, 101)
    if grid_ext is None: grid_ext = np.linspace(-1.0, 1.0, 161)
    if grid_cap is None: grid_cap = np.linspace(-1.0, 1.0, 161)

    # Age
    best_age, best_s = 0.0, np.inf
    for b in grid_age:
        tmp = apply_continuous_adjustment(d, beta_T, b, 0.0, 0.0)
        s = month_to_month_shift(tmp, "log_adj_cont", "age_bin")
        if np.isfinite(s) and s < best_s:
            best_s, best_age = s, b

    # Extension
    best_ext, best_s = 0.0, np.inf
    for b in grid_ext:
        tmp = apply_continuous_adjustment(d, beta_T, best_age, b, 0.0)
        s = month_to_month_shift(tmp, "log_adj_cont", "ext_bin")
        if np.isfinite(s) and s < best_s:
            best_s, best_ext = s, b

    # Capacity
    best_cap, best_s = 0.0, np.inf
    for b in grid_cap:
        tmp = apply_continuous_adjustment(d, beta_T, best_age, best_ext, b)
        s = month_to_month_shift(tmp, "log_adj_cont", "cap_bin")
        if np.isfinite(s) and s < best_s:
            best_s, best_cap = s, b

    return float(best_age), float(best_ext), float(best_cap)


# ----------------------------
# 4) Machinery validation via matched strata regression
# ----------------------------
def calibrate_machinery_effects(df: pd.DataFrame) -> pd.Series:
    """
    Fit: log_adj_cont ~ C(stratum) + C(Machinery)
    where stratum matches on Region + age/ext/cap bins.
    """
    d = add_bins(df, q=3).copy()
    d["stratum"] = (
        d["Region"].astype(str) + " | "
        + d["age_bin"].astype(str) + " | "
        + d["ext_bin"].astype(str) + " | "
        + d["cap_bin"].astype(str)
    )

    fit = smf.ols("log_adj_cont ~ C(stratum) + C(Machinery)", data=d).fit(cov_type="HC3")
    # return only machinery coefficients (relative to reference)
    return fit.params[fit.params.index.str.startswith("C(Machinery)")]


def apply_machinery_adjustment(df: pd.DataFrame, mach_params: pd.Series) -> pd.DataFrame:
    """
    Subtract the machinery effect (relative to reference category).
    """
    d = df.copy()
    d["mach_effect"] = 0.0
    for k, v in mach_params.items():
        # k like: C(Machinery)[T.internal combustion]
        level = k.split("[T.")[-1].rstrip("]")
        d.loc[d["Machinery"] == level, "mach_effect"] = float(v)

    d["log_OilProxy_adj"] = d["log_adj_cont"] - d["mach_effect"]
    d["OilProxy_adj"] = np.expm1(d["log_OilProxy_adj"])
    return d


# ----------------------------
# 5) One-shot: validate + calibrate + adjust + report
# ----------------------------
def validate_and_adjust(monthly_df: pd.DataFrame):
    panel = prep_panel(monthly_df)

    # A) temperature
    beta_T = calibrate_beta_T(panel)

    # B) continuous covariates (age/ext/cap)
    beta_Age, beta_Ext, beta_Cap = calibrate_beta_continuous(panel, beta_T, q_bins=3)

    # apply continuous adjustment
    cont = apply_continuous_adjustment(panel, beta_T, beta_Age, beta_Ext, beta_Cap)

    # C) machinery
    mach_params = calibrate_machinery_effects(cont)
    adj = apply_machinery_adjustment(cont, mach_params)

    # D) validation summary
    persist = select_persistent(panel)
    persist_raw = seasonality_r2_by_moy(persist, "log_proxy")

    persist_tmp = persist.copy()
    persist_tmp["log_T_adj"] = persist_tmp["log_proxy"] - beta_T * persist_tmp["TempNorm"]
    persist_T = seasonality_r2_by_moy(persist_tmp, "log_T_adj")

    # temperature correlation before/after
    corr_raw_T = float(panel["log_proxy"].corr(panel["TempNorm"]))
    corr_adj_T = float(adj["log_OilProxy_adj"].corr(adj["TempNorm"]))

    # bin shift before/after (using same bins)
    binned_raw = add_bins(panel, q=3)
    binned_adj = add_bins(adj, q=3)

    report = {
        "beta_T": beta_T,
        "beta_Age": beta_Age,
        "beta_Ext": beta_Ext,
        "beta_Cap": beta_Cap,
        "seasonality_R2_persistent_raw": persist_raw,
        "seasonality_R2_persistent_T_adj": persist_T,
        "corr_raw_vs_TempNorm": corr_raw_T,
        "corr_adj_vs_TempNorm": corr_adj_T,
        "shift_raw_age_bins": month_to_month_shift(binned_raw, "log_proxy", "age_bin"),
        "shift_adj_age_bins": month_to_month_shift(binned_adj, "log_OilProxy_adj", "age_bin"),
        "shift_raw_ext_bins": month_to_month_shift(binned_raw, "log_proxy", "ext_bin"),
        "shift_adj_ext_bins": month_to_month_shift(binned_adj, "log_OilProxy_adj", "ext_bin"),
        "shift_raw_cap_bins": month_to_month_shift(binned_raw, "log_proxy", "cap_bin"),
        "shift_adj_cap_bins": month_to_month_shift(binned_adj, "log_OilProxy_adj", "cap_bin"),
        "n_plants": int(panel["Plant name"].nunique()),
        "n_rows": int(len(panel)),
    }

    return adj, mach_params, pd.Series(report)

def download_in_parallel(
    df_flaring,
    batch_size: int = 20,
    max_workers: int = 20,
    period: str = "M",
    project = 'stable-healer-488213-f9',
    *,
    location_col: str = "Plant name",
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    start_col: str = "Start_production",
    end_col: str = "End_production",
    **download_kwargs,
):
    """Run `download(...)` concurrently, batch-by-batch.

    - Concurrency is capped by `max_workers`.
    - Batches are processed sequentially to avoid overwhelming APIs / rate limits.
    - Returns a list of results: dicts with keys: location, ok, error (optional).
    """
    init_ee(project = project)  #stable-healer-488213-f9 #ee-brybisetti-cluster #tesi-isa-1 #peppy-center-488409-p3  and cluster is running on #hardy-unison-487923-t0

    results = []
    n = len(df_flaring)
    for b0 in range(0, n, batch_size):
        b1 = min(b0 + batch_size, n)
        batch = df_flaring.iloc[b0:b1]

        print(f"=== Batch {b0//batch_size + 1} | rows {b0}..{b1-1} | period={period} | workers={max_workers} ===")

        with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {}
            for _, row in batch.iterrows():
                location = row[location_col]
                lon = float(row[lon_col])
                lat = float(row[lat_col])
                start_date = row[start_col]
                end_date = row[end_col]
                fut = ex.submit(download_url_1, location, lon, lat, start_date, end_date, period, **download_kwargs)
                futs[fut] = location

            for fut in _cf.as_completed(futs):
                location = futs[fut]
                try:
                    fut.result()
                    results.append({"location": location, "ok": True})
                except Exception as e:
                    results.append({"location": location, "ok": False, "error": repr(e)})
                    print(f"[ERROR] {location}: {e}")

    return results
    


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def fetch_one_keyword(
    keyword: str,
    start_str: str,
    end_str: str,
    BASE_URL: str,
    MODE: str,
    FORMAT: str,
    MAXRECORDS: int,
    max_retries: int = 8,
    base_sleep: float = 1.5
):
    """
    One keyword -> one GDELT request -> returns a DataFrame or None.
    Retries on 429 with exponential backoff.
    """
    query = f'"{keyword}"'
    params = {
        "query": query,
        "mode": MODE,
        "format": FORMAT,
        "maxrecords": MAXRECORDS,
        "startdatetime": start_str,
        "enddatetime": end_str
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=40)
        except Exception:
            resp = None

        # Success
        if resp is not None and resp.status_code == 200 and resp.text.strip():
            try:
                df = pd.read_csv(StringIO(resp.text))
                return df
            except Exception:
                return None

        # Rate limited
        if resp is not None and resp.status_code == 429:
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)
            continue

        # Other errors (temporary)
        time.sleep(0.5 + random.uniform(0, 0.5))

    return None


# ---------------------------------------------------------
# Main downloader (parallel keyword requests in batches)
# ---------------------------------------------------------
def download_csv(date: str, days_param: int):
    """
    date format must be: 'Aug 2022' (Mon YYYY)
    days_param = window size in days for each GDELT call window (e.g. 1, 3, 7)
    """

    # --- parse "Aug 2022" into numeric month/year ---
    try:
        mon_str, yr_str = date.split()
        month = datetime.datetime.strptime(mon_str, "%b").month
        year  = int(yr_str)
    except Exception:
        raise ValueError(f"date must be 'Mon YYYY', got {date!r}")

    # month boundaries
    if month == 12:
        next_month, next_year = 1, year + 1
    else:
        next_month, next_year = month + 1, year

    period_start = datetime.datetime(year, month, 1, 0, 0)
    period_end   = datetime.datetime(next_year, next_month, 1, 0, 0)

    print("----------------------------------------------------------")
    print(f"Period Start: {period_start.date()} --- Period End: {period_end.date()}")
    print("----------------------------------------------------------")

    # --- GDELT constants ---
    BASE_URL    = "https://api.gdeltproject.org/api/v2/doc/doc"
    FORMAT      = "CSV"
    MODE        = "artList"
    MAXRECORDS  = 250

    # --- keywords (choose ONE) ---
    # keywords = oil_keywords + geopolitics_keywords
    # keywords = examples
    keywords = examples

    # parallel settings
    KEYWORD_BATCH_SIZE = 20   # <= batches of 20 keywords
    MAX_WORKERS = 20          # <= 20 concurrent requests (reduce if too many 429s)

    start_time = period_start
    combined_all = pd.DataFrame()

    while start_time < period_end:
        end_time = start_time + timedelta(days=days_param)
        if end_time > period_end:
            end_time = period_end

        start_str = start_time.strftime("%Y%m%d%H%M%S")
        end_str   = end_time.strftime("%Y%m%d%H%M%S")

        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Querying for period: {start_time.date()} → {end_time.date()}")
        print("+++++++++++++++++++++++++++++++++++++++++++++++")

        all_batches = []

        # Process keywords in batches of 20, each batch runs up to 20 threads
        for kw_batch in chunks(keywords, KEYWORD_BATCH_SIZE):
            print(f"--- Parallel batch of {len(kw_batch)} keywords ---")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        fetch_one_keyword,
                        kw, start_str, end_str,
                        BASE_URL, MODE, FORMAT, MAXRECORDS
                    ): kw
                    for kw in kw_batch
                }

                for fut in as_completed(futures):
                    kw = futures[fut]
                    df_out = fut.result()

                    if df_out is not None and len(df_out) > 0:
                        print(f"  ✓ {kw}: {len(df_out)} articles")
                        all_batches.append(df_out)
                    else:
                        print(f"  - {kw}: 0 / failed")

        # Move the window forward
        start_time = end_time

        # Concatenate this date-window results
        if all_batches:
            combined_date = pd.concat(all_batches, ignore_index=True)
            print(f"Number of articles found in this date range: {len(combined_date)}")
            combined_all = pd.concat([combined_all, combined_date], ignore_index=True)
        else:
            print("Number of articles found in this date range: 0")

    # Save output
    out_path = f"articles/{period_start.date()}_{period_end.date()}.csv"
    combined_all.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | rows={len(combined_all)}")

    return combined_all


analyzer = SentimentIntensityAnalyzer()

def score_text_vader(x: str):
    if pd.isna(x):
        return np.nan, np.nan
    x = str(x).strip()
    if not x:
        return np.nan, np.nan
    s = analyzer.polarity_scores(x)
    # compound in [-1, 1]
    compound = s["compound"]
    label = "pos" if compound >= 0.05 else ("neg" if compound <= -0.05 else "neu")
    return compound, label


