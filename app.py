from __future__ import annotations

import json
import re
import shutil
import warnings as python_warnings
import zipfile
from pathlib import Path
from typing import Any

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import streamlit as st
from pandas.errors import PerformanceWarning
from shapely.errors import GEOSException
from streamlit_folium import st_folium

python_warnings.filterwarnings("ignore", category=PerformanceWarning)


# ---------------------------------------------------------------------------
# Basic project settings
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
DEPLOY_DATA_DIR = BASE_DIR / "deploy_data"
MAX_MAP_FEATURES = 5000
APP_VERSION = "2026-05-06 vote-scale fix"

STATE_CONFIG = {
    "California": {
        "abbr": "CA",
        "folder": "CA",
        "center": [36.7783, -119.4179],
        "zoom": 5,
        "district_zip": "ca_cong_adopted_2025.zip",
        "precinct_zip": "ca_2024_gen_prec.zip",
        "l2_zip": "CA_l2_2024_gen_stats_2020block.zip",
        "cvap_zip": "ca_cvap_2023_b.zip",
        "cities": [
            ("Los Angeles", 34.0522, -118.2437),
            ("San Diego", 32.7157, -117.1611),
            ("San Jose", 37.3382, -121.8863),
            ("San Francisco", 37.7749, -122.4194),
            ("Sacramento", 38.5816, -121.4944),
            ("Fresno", 36.7378, -119.7871),
            ("Oakland", 37.8044, -122.2712),
            ("Bakersfield", 35.3733, -119.0187),
        ],
    },
    "Louisiana": {
        "abbr": "LA",
        "folder": "LA",
        "center": [30.9843, -91.9623],
        "zoom": 7,
        "district_zip": "la_cong_adopted_2024.zip",
        "precinct_zip": "la_2024_gen_prec.zip",
        "l2_zip": "LA_l2_2024_gen_stats_2020block.zip",
        "cvap_zip": "la_cvap_2023_b.zip",
        "cities": [
            ("New Orleans", 29.9511, -90.0715),
            ("Baton Rouge", 30.4515, -91.1871),
            ("Shreveport", 32.5252, -93.7502),
            ("Lafayette", 30.2241, -92.0198),
            ("Lake Charles", 30.2266, -93.2174),
            ("Alexandria", 31.3113, -92.4451),
            ("Monroe", 32.5093, -92.1193),
        ],
    },
}

LAYER_DEFINITIONS = [
    {
        "category": "Votes",
        "label": "Partisan result (blue D / red R)",
        "column": "dem_share",
        "description": "Blue districts voted more Democratic; red districts voted more Republican. Hover for exact percentages.",
    },
    {
        "category": "Votes",
        "label": "Republican share",
        "column": "rep_share",
        "description": "Two-party Republican vote share: Republican votes divided by Democratic plus Republican votes.",
    },
    {
        "category": "Votes",
        "label": "D + R votes",
        "column": "total_dr_votes",
        "description": "Democratic plus Republican votes. This is not total population.",
    },
    {
        "category": "Turnout",
        "label": "Overall turnout",
        "column": "turnout_rate",
        "description": "Votes divided by registered voters when those fields are available.",
    },
    {
        "category": "Turnout",
        "label": "Young turnout",
        "column": "young_voter_turnout",
        "description": "Estimated turnout for young voter age groups, usually 18-24, when L2 age fields are detected.",
    },
    {
        "category": "CVAP demographics",
        "label": "Black CVAP",
        "column": "black_cvap_pct",
        "description": "Black Citizen Voting Age Population as a share of total CVAP.",
    },
    {
        "category": "CVAP demographics",
        "label": "Latino CVAP",
        "column": "latino_cvap_pct",
        "description": "Latino Citizen Voting Age Population as a share of total CVAP.",
    },
    {
        "category": "CVAP demographics",
        "label": "Asian CVAP",
        "column": "asian_cvap_pct",
        "description": "Asian Citizen Voting Age Population as a share of total CVAP.",
    },
    {
        "category": "CVAP demographics",
        "label": "White CVAP",
        "column": "white_cvap_pct",
        "description": "White Citizen Voting Age Population as a share of total CVAP.",
    },
    {
        "category": "CVAP demographics",
        "label": "Minority CVAP",
        "column": "minority_cvap_pct",
        "description": "Non-white CVAP share, calculated as one minus White CVAP share.",
    },
]
LAYER_OPTIONS = [(layer["label"], layer["column"]) for layer in LAYER_DEFINITIONS]
LAYER_BY_COLUMN = {layer["column"]: layer for layer in LAYER_DEFINITIONS}

PERCENT_COLUMNS = {
    "dem_share",
    "rep_share",
    "turnout_rate",
    "black_cvap_pct",
    "latino_cvap_pct",
    "asian_cvap_pct",
    "white_cvap_pct",
    "minority_cvap_pct",
    "young_voter_turnout",
}


# ---------------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------------


def inject_dark_theme_css() -> None:
    """Apply a compact dark theme on top of Streamlit's base styling."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }

        .stApp {
            background: #0b1120;
            color: #e5e7eb;
        }

        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #253044;
        }

        [data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f8fafc;
        }

        h1, h2, h3 {
            color: #f8fafc;
        }

        div[data-testid="stAlert"] {
            background: #172033;
            border: 1px solid #334155;
            color: #e5e7eb;
        }

        div[data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #253044;
            border-radius: 8px;
            padding: 0.8rem;
        }

        div[data-testid="stExpander"] {
            background: #111827;
            border: 1px solid #253044;
            border-radius: 8px;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: #1f2937;
            color: #f8fafc;
            border: 1px solid #475569;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: #334155;
            border-color: #94a3b8;
            color: #ffffff;
        }

        iframe {
            border-radius: 8px;
            border: 1px solid #253044;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clean_name(value: Any) -> str:
    """Normalize a column name so flexible matching is easier."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numbers while keeping missing and messy cells safe."""
    if series is None:
        return pd.Series(dtype="float64")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two numeric series and return NaN instead of crashing on zero."""
    num = safe_numeric(numerator)
    den = safe_numeric(denominator)
    den = den.replace(0, np.nan)
    return num / den


def normalize_district_value(value: Any) -> str | None:
    """Turn values such as 'District 06' or '6.0' into a simple join key."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    number_match = re.search(r"\d+", text)
    if number_match:
        return str(int(number_match.group(0)))
    return clean_name(text)


def normalize_block_value(value: Any) -> str | None:
    """Keep only digits from a Census block GEOID-like value."""
    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) >= 15:
        return digits[:15]
    return digits or None


def has_usable_numeric_data(df: pd.DataFrame, column: str) -> bool:
    """True when a map/chart column exists and has at least one numeric value."""
    return column in df.columns and safe_numeric(df[column]).notna().any()


def format_number(value: Any) -> str:
    number = safe_numeric(pd.Series([value])).iloc[0]
    if pd.isna(number):
        return "Not available"
    return f"{number:,.0f}"


def format_percent(value: Any) -> str:
    number = safe_numeric(pd.Series([value])).iloc[0]
    if pd.isna(number):
        return "Not available"
    if number > 1.5:
        number = number / 100
    return f"{number:.1%}"


# ---------------------------------------------------------------------------
# Data discovery and loading
# ---------------------------------------------------------------------------


def safe_extract_zip(archive: zipfile.ZipFile, extract_dir: Path) -> None:
    """Extract a ZIP only if every member stays inside the target folder."""
    target_root = extract_dir.resolve()
    for member in archive.infolist():
        member_target = (target_root / member.filename).resolve()
        if not member_target.is_relative_to(target_root):
            raise ValueError(f"Unsafe ZIP member path: {member.filename}")
    archive.extractall(extract_dir)


def unzip_data(
    state_folder: Path,
    warnings: list[str] | None = None,
    zip_names: list[str] | None = None,
) -> dict[str, Path]:
    """
    Extract every ZIP file in a state folder into a clean generated folder.

    Each archive gets a sibling folder named like:
    ca_2024_gen_prec_unzipped/
    """
    warnings = warnings if warnings is not None else []
    extracted: dict[str, Path] = {}

    if not state_folder.exists():
        warnings.append(f"State data folder was not found: {state_folder}")
        return extracted

    wanted = set(zip_names or [])
    for zip_path in sorted(state_folder.glob("*.zip")):
        if wanted and zip_path.name not in wanted:
            continue
        extract_dir = state_folder / f"{zip_path.stem}_unzipped"
        marker = extract_dir / ".source_zip_mtime"
        try:
            source_mtime = str(zip_path.stat().st_mtime_ns)
        except OSError as exc:
            warnings.append(f"Could not inspect ZIP file {zip_path.name}: {exc}")
            continue

        if marker.exists() and marker.read_text().strip() == source_mtime:
            extracted[zip_path.name] = extract_dir
            continue

        try:
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path) as archive:
                safe_extract_zip(archive, extract_dir)
            marker.write_text(source_mtime)
            extracted[zip_path.name] = extract_dir
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            warnings.append(f"Could not unzip {zip_path.name}: {exc}")
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)

    return extracted


def find_geospatial_files(search_root: Path | None) -> list[Path]:
    """Find geospatial and useful table files below an extracted data folder."""
    allowed = {".shp", ".gpkg", ".geojson", ".json", ".csv", ".dbf"}
    if not search_root or not search_root.exists():
        return []
    return sorted(
        path
        for path in search_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed
        and not path.name.startswith(".")
    )


def score_file(path: Path, kind: str, want: str) -> int:
    """Rank files so the app can pick the best likely file automatically."""
    name = clean_name(path.stem)
    suffix = path.suffix.lower()
    score = 0

    if want == "geo":
        score += {".gpkg": 80, ".geojson": 75, ".json": 55, ".shp": 70}.get(suffix, -100)
    elif want == "table":
        score += {".csv": 80, ".dbf": 35}.get(suffix, -100)

    if kind == "district":
        score += 20 if any(word in name for word in ["cong", "congress", "district", "ab604", "act2"]) else 0
    elif kind == "district_assignment":
        score += 30 if suffix == ".csv" else 0
        score += 20 if any(word in name for word in ["cong", "congress", "ab604", "act2"]) else 0
        score -= 50 if "report" in name or "readme" in name else 0
    elif kind == "precinct":
        score += 30 if "prec" in name or "vtd" in name else 0
        score += 15 if "congprec" in name else 0
        score += 5 if "allprec" in name else 0
    elif kind == "cvap":
        score += 30 if "cvap" in name else 0
    elif kind == "l2":
        score += 30 if "l2" in name or "stats" in name else 0

    return score


def choose_best_file(files: list[Path], kind: str, want: str) -> Path | None:
    """Return the highest-ranked file for a dataset, or None if none exists."""
    candidates = [path for path in files if score_file(path, kind, want) > 0]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (score_file(path, kind, want), path.stat().st_size), reverse=True)[0]


def geofile_columns(path: Path | None) -> list[str]:
    """Read geospatial field names without loading the whole dataset."""
    if path is None:
        return []
    try:
        info = pyogrio.read_info(path)
        fields = list(info.get("fields") or [])
        return [str(field) for field in fields]
    except Exception:
        try:
            sample = gpd.read_file(path, rows=1, engine="pyogrio")
            return [column for column in sample.columns if column != "geometry"]
        except Exception:
            return []


def needed_precinct_attribute_columns(path: Path | None) -> list[str] | None:
    """Choose only the precinct fields needed for calculations and debugging."""
    columns = geofile_columns(path)
    if not columns:
        return None

    detected = detect_columns(pd.DataFrame(columns=columns))
    needed: list[str] = []
    for key in ["district_id", "precinct_id", "county", "total_votes", "registered_voters", "turnout"]:
        column = detected.get(key)
        if column in columns:
            needed.append(column)
    needed.extend(column for column in detected.get("dem_vote_cols", []) if column in columns)
    needed.extend(column for column in detected.get("rep_vote_cols", []) if column in columns)
    return list(dict.fromkeys(needed)) or None


def missing_shapefile_sidecars(path: Path) -> list[str]:
    """Return required shapefile companion files that are missing."""
    if path.suffix.lower() != ".shp":
        return []
    required_suffixes = [".shx", ".dbf"]
    return [path.with_suffix(suffix).name for suffix in required_suffixes if not path.with_suffix(suffix).exists()]


def ensure_wgs84(gdf: gpd.GeoDataFrame, label: str, warnings: list[str]) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame in EPSG:4326 without crashing on CRS problems."""
    gdf = gdf.copy()
    if gdf.crs is None:
        warnings.append(f"{label} did not include a CRS. The app assumes EPSG:4326.")
        return gdf.set_crs(4326, allow_override=True)

    try:
        return gdf.to_crs(4326)
    except Exception as exc:
        warnings.append(f"{label} could not be reprojected to EPSG:4326: {exc}")
        return gdf


def load_geodata(path: Path | None, label: str, warnings: list[str]) -> gpd.GeoDataFrame | None:
    """Load a shapefile, GeoPackage, or GeoJSON into a GeoDataFrame."""
    if path is None:
        warnings.append(f"No geospatial file was found for {label}.")
        return None

    missing_sidecars = missing_shapefile_sidecars(path)
    if missing_sidecars:
        warnings.append(f"{label} shapefile is missing required sidecar file(s): {', '.join(missing_sidecars)}")

    try:
        try:
            gdf = gpd.read_file(path, engine="pyogrio")
        except Exception:
            gdf = gpd.read_file(path)
    except Exception as exc:
        warnings.append(f"Could not load {label} from {path.name}: {exc}")
        return None

    if gdf.empty:
        warnings.append(f"{label} loaded from {path.name}, but it has no rows.")
        return None

    return ensure_wgs84(gdf, label, warnings)


def load_geodata_sample(
    path: Path | None,
    label: str,
    warnings: list[str],
    max_features: int | None = None,
) -> gpd.GeoDataFrame | None:
    """Load geodata, optionally limiting rows for faster Folium rendering."""
    if path is None:
        warnings.append(f"No geospatial file was found for {label}.")
        return None

    missing_sidecars = missing_shapefile_sidecars(path)
    if missing_sidecars:
        warnings.append(f"{label} shapefile is missing required sidecar file(s): {', '.join(missing_sidecars)}")

    columns = needed_precinct_attribute_columns(path)
    read_kwargs: dict[str, Any] = {}
    if columns:
        read_kwargs["columns"] = columns
    if max_features:
        read_kwargs["max_features"] = max_features

    try:
        gdf = pyogrio.read_dataframe(path, read_geometry=True, **read_kwargs)
    except Exception:
        try:
            gdf = gpd.read_file(path, engine="pyogrio", **read_kwargs)
        except Exception as exc:
            warnings.append(f"Could not load {label} from {path.name}: {exc}")
            return None

    if gdf.empty:
        warnings.append(f"{label} loaded from {path.name}, but it has no rows.")
        return None
    if max_features:
        warnings.append(
            f"{label}: showing the first {len(gdf):,} features for speed. "
            "Use congressional district view for full-state summaries."
        )
    return ensure_wgs84(gdf, label, warnings)


def load_geodata_attributes(path: Path | None, label: str, warnings: list[str]) -> pd.DataFrame | None:
    """Load only geospatial attributes, not geometry. This is much faster for huge GeoJSONs."""
    if path is None:
        warnings.append(f"No geospatial attribute file was found for {label}.")
        return None

    columns = needed_precinct_attribute_columns(path)
    try:
        table = pyogrio.read_dataframe(path, read_geometry=False, columns=columns)
    except Exception:
        try:
            table = gpd.read_file(path, engine="pyogrio", ignore_geometry=True, columns=columns)
        except Exception as exc:
            warnings.append(f"Could not load attributes for {label} from {path.name}: {exc}")
            return None

    if table.empty:
        warnings.append(f"{label} attributes loaded from {path.name}, but there are no rows.")
        return None
    return pd.DataFrame(table)


def source_signature(paths: list[Path | None]) -> str:
    """Create a small fingerprint so cached cleaned data updates when sources change."""
    parts = []
    for path in paths:
        if path and path.exists():
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        elif path:
            parts.append(f"{path.name}:missing")
    return "|".join(parts)


def district_cache_paths(state_name: str) -> tuple[Path, Path]:
    """Return cache data and metadata paths for a state's district layer."""
    abbr = STATE_CONFIG[state_name]["abbr"]
    return CACHE_DIR / f"{abbr}_districts_fast.geojson", CACHE_DIR / f"{abbr}_districts_fast_meta.json"


def deploy_district_path(state_name: str) -> Path:
    """Return the small congressional district file committed for Streamlit Cloud."""
    abbr = STATE_CONFIG[state_name]["abbr"]
    return DEPLOY_DATA_DIR / f"{abbr}_districts_fast.geojson"


def state_source_zip_paths(state_name: str) -> list[Path]:
    """Return the source ZIP paths expected for a state."""
    config = STATE_CONFIG[state_name]
    state_folder = DATA_DIR / config["folder"]
    return [
        state_folder / config["district_zip"],
        state_folder / config["precinct_zip"],
        state_folder / config["cvap_zip"],
        state_folder / config["l2_zip"],
    ]


def has_state_source_data(state_name: str) -> bool:
    """True when the local/raw ZIP data is available for a state."""
    return any(path.exists() for path in state_source_zip_paths(state_name))


def load_district_cache(
    state_name: str,
    signature: str,
    warnings: list[str],
) -> gpd.GeoDataFrame | None:
    """Load a processed district GeoJSON cache when it matches current source files."""
    geojson_path, meta_path = district_cache_paths(state_name)
    if not geojson_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if meta.get("source_signature") != signature:
        return None

    try:
        gdf = gpd.read_file(geojson_path, engine="pyogrio")
    except Exception as exc:
        warnings.append(f"Could not load fast district cache for {state_name}: {exc}")
        return None

    return ensure_wgs84(gdf, f"{state_name} fast district cache", warnings)


def load_deploy_district_data(state_name: str, warnings: list[str]) -> gpd.GeoDataFrame | None:
    """Load the compact district dataset used by Streamlit Cloud deployments."""
    path = deploy_district_path(state_name)
    if not path.exists():
        return None

    try:
        gdf = gpd.read_file(path, engine="pyogrio")
    except Exception as exc:
        warnings.append(f"Could not load deploy-ready district data for {state_name}: {exc}")
        return None

    warnings.append(
        f"{state_name}: using compact deploy-ready congressional district data. "
        "Full precinct/source ZIP layers are only available when the local data folder is present."
    )
    return ensure_wgs84(gdf, f"{state_name} deploy-ready district data", warnings)


def write_district_cache(state_name: str, signature: str, district_gdf: gpd.GeoDataFrame | None) -> None:
    """Save a small processed district cache for faster future starts."""
    if district_gdf is None or district_gdf.empty:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    geojson_path, meta_path = district_cache_paths(state_name)
    cleaned_export_gdf(district_gdf).to_file(geojson_path, driver="GeoJSON")
    meta_path.write_text(json.dumps({"source_signature": signature}, indent=2))


def load_table_data(path: Path | None, label: str, warnings: list[str]) -> pd.DataFrame | None:
    """Load CSV or DBF-like attribute tables with a few fallbacks."""
    if path is None:
        warnings.append(f"No table file was found for {label}.")
        return None

    try:
        if path.suffix.lower() == ".dbf":
            table = gpd.read_file(path).drop(columns="geometry", errors="ignore")
        else:
            table = pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8-sig")
            if table.shape[1] == 1:
                table = pd.read_csv(path, dtype=str, sep=r"\s+", engine="python", encoding="utf-8-sig")

            if looks_like_headerless_assignment(table):
                table = read_headerless_assignment(path)
    except Exception as exc:
        warnings.append(f"Could not load {label} from {path.name}: {exc}")
        return None

    if table.empty:
        warnings.append(f"{label} loaded from {path.name}, but it has no rows.")
        return None

    return table


def read_headerless_assignment(path: Path) -> pd.DataFrame:
    """Read a two-column block-to-district assignment file with no header."""
    with path.open(encoding="utf-8-sig", errors="ignore") as file:
        sample = file.readline()
    if "," in sample:
        return pd.read_csv(
            path,
            dtype=str,
            header=None,
            names=["block_id", "district_id"],
            encoding="utf-8-sig",
        )
    return pd.read_csv(
        path,
        dtype=str,
        header=None,
        names=["block_id", "district_id"],
        sep=r"\s+",
        engine="python",
        encoding="utf-8-sig",
    )


def looks_like_headerless_assignment(table: pd.DataFrame) -> bool:
    """Detect block-to-district assignment CSVs where the first row became a header."""
    if table.shape[1] != 2:
        return False
    col0, col1 = [str(column).strip() for column in table.columns[:2]]
    return bool(re.fullmatch(r"\d{12,15}", col0) and re.fullmatch(r"\d{1,3}", col1))


# ---------------------------------------------------------------------------
# Flexible column detection
# ---------------------------------------------------------------------------


def find_first_column(
    columns: list[str],
    exact: list[str] | None = None,
    contains_all: list[str] | None = None,
    contains_any: list[str] | None = None,
    regexes: list[str] | None = None,
    avoid: list[str] | None = None,
) -> str | None:
    """Find one likely column using exact, contains, and regex clues."""
    exact = [clean_name(item) for item in exact or []]
    contains_all = [clean_name(item) for item in contains_all or []]
    contains_any = [clean_name(item) for item in contains_any or []]
    avoid = [clean_name(item) for item in avoid or []]
    regexes = regexes or []

    normalized = {column: clean_name(column) for column in columns}

    for exact_name in exact:
        for column, name in normalized.items():
            if name == exact_name and not any(bad in name for bad in avoid):
                return column

    for column, name in normalized.items():
        if any(bad in name for bad in avoid):
            continue
        if contains_all and not all(part in name for part in contains_all):
            continue
        if contains_any and not any(part in name for part in contains_any):
            continue
        if regexes and not any(re.search(pattern, name) for pattern in regexes):
            continue
        return column

    return None


def find_columns_by_regex(columns: list[str], regexes: list[str], avoid: list[str] | None = None) -> list[str]:
    """Find all columns matching one of several normalized regexes."""
    avoid = [clean_name(item) for item in avoid or []]
    matches = []
    for column in columns:
        name = clean_name(column)
        if any(bad in name for bad in avoid):
            continue
        if any(re.search(pattern, name) for pattern in regexes):
            matches.append(column)
    return matches


def find_block_column(df: pd.DataFrame) -> str | None:
    """Detect a Census block GEOID field by name and 15-digit-looking values."""
    likely_names = ["geoid20", "blockid", "blockgeoid", "geoid", "block"]
    columns = list(df.columns)
    named_candidates = [
        column for column in columns if clean_name(column) in likely_names or "block" in clean_name(column)
    ]
    candidates = named_candidates + [column for column in columns if column not in named_candidates]

    for column in candidates:
        sample = df[column].dropna().astype(str).head(100)
        if sample.empty:
            continue
        block_like = sample.map(lambda value: bool(re.fullmatch(r"\D*\d{15}\D*", value.strip())))
        if block_like.mean() >= 0.75:
            return column
    return None


def detect_vote_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Detect Democratic and Republican election vote columns."""
    columns = list(df.columns)
    normalized = {column: clean_name(column) for column in columns}

    dem_pres = [
        column
        for column, name in normalized.items()
        if re.match(r"g\d{2}pred", name) or "harris" in name or "biden" in name
    ]
    rep_pres = [
        column
        for column, name in normalized.items()
        if re.match(r"g\d{2}prer", name) or "trump" in name
    ]

    dem_generic = [
        column
        for column, name in normalized.items()
        if (
            ("demvotes" in name or "democraticvotes" in name or "votesdem" in name or name == "dem")
            and "pct" not in name
            and "share" not in name
        )
    ]
    rep_generic = [
        column
        for column, name in normalized.items()
        if (
            ("repvotes" in name or "republicanvotes" in name or "votesrep" in name or name == "rep")
            and "pct" not in name
            and "share" not in name
        )
    ]

    dem_congress = find_columns_by_regex(columns, [r"^gcon\d{1,2}d"], avoid=["pct", "share"])
    rep_congress = find_columns_by_regex(columns, [r"^gcon\d{1,2}r"], avoid=["pct", "share"])

    detected: dict[str, Any] = {}
    if "dem_votes" in df.columns:
        detected["dem_vote_cols"] = ["dem_votes"]
    elif dem_pres:
        detected["dem_vote_cols"] = dem_pres
    elif dem_generic:
        detected["dem_vote_cols"] = dem_generic
    elif dem_congress:
        detected["dem_vote_cols"] = dem_congress

    if "rep_votes" in df.columns:
        detected["rep_vote_cols"] = ["rep_votes"]
    elif rep_pres:
        detected["rep_vote_cols"] = rep_pres
    elif rep_generic:
        detected["rep_vote_cols"] = rep_generic
    elif rep_congress:
        detected["rep_vote_cols"] = rep_congress

    return detected


def detect_columns(df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect likely columns without assuming exact names.

    The returned dictionary is shown in the Streamlit debug panel.
    """
    columns = list(df.columns)
    detected: dict[str, Any] = {}

    detected["district_id"] = find_first_column(
        columns,
        exact=[
            "cong_dist",
            "congressional_district",
            "district",
            "district_id",
            "district_i",
            "districtid",
            "districti",
            "dist",
            "distid",
            "cd",
        ],
        contains_all=["cong", "dist"],
    )
    detected["precinct_id"] = find_first_column(
        columns,
        exact=["unique_id", "precinct", "precinct_id", "geoid20", "name20", "vtd", "vtdst"],
        contains_any=["precinct", "vtd"],
    )
    detected["county"] = find_first_column(
        columns,
        exact=["county", "county_name", "parish", "parish_name", "countyfp"],
        contains_any=["county", "parish"],
    )
    detected["block_id"] = find_block_column(df)

    detected.update(detect_vote_columns(df))

    detected["total_votes"] = find_first_column(
        columns,
        exact=["total_votes", "totvote", "totvotes", "voted_all", "ballots", "votes_total"],
        contains_any=["totalvote", "totvote", "votedall", "ballot"],
        avoid=["dr", "dem", "rep", "pct", "share"],
    )
    detected["registered_voters"] = find_first_column(
        columns,
        exact=["registered_voters", "totreg", "total_registered", "reg_all", "registration"],
        contains_any=["registered", "totreg", "regall", "registration"],
        avoid=["pct", "share", "party", "gender", "age"],
    )
    detected["turnout"] = find_first_column(
        columns,
        exact=["turnout_rate", "pct_voted_all", "turnout", "voter_turnout"],
        contains_any=["turnout", "pctvotedall"],
        avoid=["age", "party", "gender"],
    )

    detected["black_cvap"] = find_first_column(
        columns,
        exact=["black_cvap", "cvap_bla23", "cvap_blk23"],
        contains_all=["cvap"],
        contains_any=["bla", "black", "blk"],
    )
    detected["latino_cvap"] = find_first_column(
        columns,
        exact=["latino_cvap", "hispanic_cvap", "cvap_hsp23"],
        contains_all=["cvap"],
        contains_any=["hsp", "hisp", "latino", "latinx"],
    )
    detected["asian_cvap"] = find_first_column(
        columns,
        exact=["asian_cvap", "cvap_asi23"],
        contains_all=["cvap"],
        contains_any=["asi", "asian"],
    )
    detected["white_cvap"] = find_first_column(
        columns,
        exact=["white_cvap", "cvap_wht23"],
        contains_all=["cvap"],
        contains_any=["wht", "white"],
    )
    detected["total_cvap"] = find_first_column(
        columns,
        exact=["total_cvap", "cvap_tot23", "cvaptot23"],
        contains_all=["cvap"],
        contains_any=["tot", "total"],
    )

    age_vote_cols = [
        column
        for column in columns
        if "pct" not in clean_name(column) and re.search(r"votedage(18|20|21|22|23|24)", clean_name(column))
    ]
    age_reg_cols = [column for column in columns if re.search(r"regage(18|20|21|22|23|24)", clean_name(column))]
    detected["young_voted_cols"] = age_vote_cols
    detected["young_registered_cols"] = age_reg_cols

    return {key: value for key, value in detected.items() if value}


# ---------------------------------------------------------------------------
# Calculated columns and aggregation
# ---------------------------------------------------------------------------


def calculate_vote_columns(df: pd.DataFrame, detected: dict[str, Any], warnings: list[str], label: str) -> pd.DataFrame:
    """Create canonical election columns when enough source columns are present."""
    df = df.copy()

    dem_cols = [column for column in detected.get("dem_vote_cols", []) if column in df.columns]
    rep_cols = [column for column in detected.get("rep_vote_cols", []) if column in df.columns]

    if dem_cols and "dem_votes" not in df.columns:
        df["dem_votes"] = sum(safe_numeric(df[column]) for column in dem_cols)
    if rep_cols and "rep_votes" not in df.columns:
        df["rep_votes"] = sum(safe_numeric(df[column]) for column in rep_cols)

    if "dem_votes" in df.columns and "rep_votes" in df.columns:
        df["total_dr_votes"] = safe_numeric(df["dem_votes"]) + safe_numeric(df["rep_votes"])
        df["dem_share"] = safe_divide(df["dem_votes"], df["total_dr_votes"])
        df["rep_share"] = safe_divide(df["rep_votes"], df["total_dr_votes"])
    else:
        warnings.append(f"{label}: Democratic and Republican vote columns were not both detected.")

    total_votes_col = detected.get("total_votes")
    registered_col = detected.get("registered_voters")
    turnout_col = detected.get("turnout")

    if total_votes_col in df.columns and "total_votes" not in df.columns:
        df["total_votes"] = safe_numeric(df[total_votes_col])
    if registered_col in df.columns and "registered_voters" not in df.columns:
        df["registered_voters"] = safe_numeric(df[registered_col])

    if "total_votes" in df.columns and "registered_voters" in df.columns:
        df["turnout_rate"] = safe_divide(df["total_votes"], df["registered_voters"])
    elif turnout_col in df.columns and "turnout_rate" not in df.columns:
        turnout = safe_numeric(df[turnout_col])
        df["turnout_rate"] = np.where(turnout > 1.5, turnout / 100, turnout)
    else:
        warnings.append(f"{label}: turnout rate could not be calculated from available columns.")

    return df


def calculate_cvap_columns(df: pd.DataFrame, detected: dict[str, Any], warnings: list[str], label: str) -> pd.DataFrame:
    """Create canonical CVAP percentage columns when enough source columns exist."""
    df = df.copy()
    total_col = detected.get("total_cvap")

    if total_col not in df.columns:
        warnings.append(f"{label}: total CVAP was not detected, so CVAP percentage layers are unavailable.")
        return df

    source_to_output = {
        "black_cvap": "black_cvap_pct",
        "latino_cvap": "latino_cvap_pct",
        "asian_cvap": "asian_cvap_pct",
        "white_cvap": "white_cvap_pct",
    }
    for source_key, output_col in source_to_output.items():
        source_col = detected.get(source_key)
        if source_col in df.columns:
            df[output_col] = safe_divide(df[source_col], df[total_col])

    if "white_cvap_pct" in df.columns:
        df["minority_cvap_pct"] = 1 - safe_numeric(df["white_cvap_pct"])

    if not any(column in df.columns for column in source_to_output.values()):
        warnings.append(f"{label}: race or ethnicity CVAP columns were not detected.")

    return df


def calculate_l2_columns(df: pd.DataFrame, detected: dict[str, Any], warnings: list[str], label: str) -> pd.DataFrame:
    """Create turnout and young-voter turnout fields from L2-style columns."""
    df = df.copy()
    total_votes = detected.get("total_votes")
    registered = detected.get("registered_voters")

    if "turnout_rate" not in df.columns and total_votes in df.columns and registered in df.columns:
        df["turnout_rate"] = safe_divide(df[total_votes], df[registered])

    young_voted_cols = [column for column in detected.get("young_voted_cols", []) if column in df.columns]
    young_reg_cols = [column for column in detected.get("young_registered_cols", []) if column in df.columns]

    if young_voted_cols and young_reg_cols:
        young_voted = sum(safe_numeric(df[column]) for column in young_voted_cols)
        young_registered = sum(safe_numeric(df[column]) for column in young_reg_cols)
        df["young_voter_turnout"] = young_voted / young_registered.replace(0, np.nan)
    else:
        warnings.append(f"{label}: young voter age columns were not detected from available columns.")

    return df


def numeric_count_columns(table: pd.DataFrame) -> list[str]:
    """Columns that can be safely summed when aggregating block-level tables."""
    numeric_cols = []
    for column in table.columns:
        name = clean_name(column)
        if (
            name in {"geometry"}
            or name.startswith("_")
            or "pct" in name
            or "rate" in name
            or "share" in name
            or "geoid" in name
            or "block" in name
        ):
            continue
        converted = safe_numeric(table[column])
        if converted.notna().any():
            numeric_cols.append(column)
    return numeric_cols


def aggregate_block_table_to_district(
    table: pd.DataFrame | None,
    assignment: pd.DataFrame | None,
    label: str,
    warnings: list[str],
) -> pd.DataFrame | None:
    """Aggregate block-level CSV data to congressional district IDs."""
    if table is None or assignment is None:
        warnings.append(f"{label}: missing block table or district assignment table.")
        return None

    table_block_col = find_block_column(table)
    assignment_detected = detect_columns(assignment)
    assignment_block_col = assignment_detected.get("block_id") or find_block_column(assignment)
    assignment_district_col = assignment_detected.get("district_id")

    if not assignment_district_col and assignment.shape[1] >= 2:
        assignment_district_col = assignment.columns[1]

    if not table_block_col or not assignment_block_col or not assignment_district_col:
        warnings.append(f"{label}: could not detect block or district columns for aggregation.")
        return None

    table = table.copy()
    assignment = assignment.copy()
    table["_block_join"] = table[table_block_col].map(normalize_block_value)
    assignment["_block_join"] = assignment[assignment_block_col].map(normalize_block_value)
    assignment["_district_join"] = assignment[assignment_district_col].map(normalize_district_value)

    numeric_cols = numeric_count_columns(table)
    if not numeric_cols:
        warnings.append(f"{label}: no numeric columns were found to aggregate.")
        return None

    merged = table[["_block_join"] + numeric_cols].merge(
        assignment[["_block_join", "_district_join"]].dropna().drop_duplicates(),
        on="_block_join",
        how="inner",
    )
    if merged.empty:
        warnings.append(f"{label}: no matching block IDs were found between data and district assignments.")
        return None

    for column in numeric_cols:
        merged[column] = safe_numeric(merged[column])

    aggregated = merged.groupby("_district_join", dropna=True)[numeric_cols].sum(min_count=1).reset_index()
    return aggregated


def aggregate_precincts_to_districts(
    precinct_gdf: pd.DataFrame | None,
    district_gdf: gpd.GeoDataFrame | None,
    warnings: list[str],
) -> pd.DataFrame | None:
    """Aggregate precinct election totals to congressional districts."""
    if precinct_gdf is None or district_gdf is None:
        return None

    precinct_detected = detect_columns(precinct_gdf.drop(columns="geometry", errors="ignore"))
    district_col = precinct_detected.get("district_id")
    if district_col not in precinct_gdf.columns:
        warnings.append("Precinct data: congressional district ID was not detected for district vote aggregation.")
        return None

    vote_cols = [
        column
        for column in ["dem_votes", "rep_votes", "total_dr_votes", "total_votes", "registered_voters"]
        if column in precinct_gdf.columns
    ]
    if not vote_cols:
        warnings.append("Precinct data: no calculated vote columns were available for district aggregation.")
        return None

    temp = precinct_gdf[[district_col] + vote_cols].copy()
    temp["_district_join"] = temp[district_col].map(normalize_district_value)
    for column in vote_cols:
        temp[column] = safe_numeric(temp[column])

    aggregated = temp.groupby("_district_join", dropna=True)[vote_cols].sum(min_count=1).reset_index()
    if "dem_votes" in aggregated.columns and "rep_votes" in aggregated.columns:
        aggregated["total_dr_votes"] = safe_numeric(aggregated["dem_votes"]) + safe_numeric(aggregated["rep_votes"])
        aggregated["dem_share"] = safe_divide(aggregated["dem_votes"], aggregated["total_dr_votes"])
        aggregated["rep_share"] = safe_divide(aggregated["rep_votes"], aggregated["total_dr_votes"])
    if "total_votes" in aggregated.columns and "registered_voters" in aggregated.columns:
        aggregated["turnout_rate"] = safe_divide(aggregated["total_votes"], aggregated["registered_voters"])

    return aggregated


def merge_district_metrics(
    district_gdf: gpd.GeoDataFrame,
    metrics: pd.DataFrame | None,
    warnings: list[str],
    label: str,
) -> gpd.GeoDataFrame:
    """Merge district-level metrics into district boundary geometry."""
    if metrics is None or metrics.empty:
        return district_gdf

    detected = detect_columns(district_gdf.drop(columns="geometry", errors="ignore"))
    district_col = detected.get("district_id")
    if district_col not in district_gdf.columns:
        warnings.append(f"{label}: district boundary ID was not detected, so metrics could not be merged.")
        return district_gdf

    district_gdf = district_gdf.copy()
    district_gdf["_district_join"] = district_gdf[district_col].map(normalize_district_value)
    merged = district_gdf.merge(metrics, on="_district_join", how="left", suffixes=("", "_metric"))
    matched = merged[[column for column in metrics.columns if column != "_district_join"]].notna().any(axis=1).sum()
    if matched == 0:
        warnings.append(f"{label}: no district IDs matched the boundary file.")
    return merged


def process_geodataframe(
    gdf: gpd.GeoDataFrame | None,
    label: str,
    warnings: list[str],
    warn_missing: bool = True,
) -> tuple[gpd.GeoDataFrame | None, dict[str, Any]]:
    """Run all available calculations directly on a GeoDataFrame."""
    if gdf is None:
        return None, {}
    warning_target = warnings if warn_missing else []
    detected = detect_columns(gdf.drop(columns="geometry", errors="ignore"))
    gdf = calculate_vote_columns(gdf, detected, warning_target, label)
    detected = detect_columns(gdf.drop(columns="geometry", errors="ignore"))
    gdf = calculate_cvap_columns(gdf, detected, warning_target, label)
    detected = detect_columns(gdf.drop(columns="geometry", errors="ignore"))
    gdf = calculate_l2_columns(gdf, detected, warning_target, label)
    detected = detect_columns(gdf.drop(columns="geometry", errors="ignore"))
    return gdf, detected


@st.cache_data(show_spinner=False)
def load_state_data(
    state_name: str,
    geography: str = "Congressional Districts",
    max_precinct_features: int = MAX_MAP_FEATURES,
    load_precinct_elections: bool = True,
    app_version: str = APP_VERSION,
) -> dict[str, Any]:
    """Load and process all data for one selected state."""
    del app_version
    config = STATE_CONFIG[state_name]
    state_folder = DATA_DIR / config["folder"]
    warnings: list[str] = []
    debug: dict[str, Any] = {"state": state_name, "files": {}, "detected_columns": {}}
    source_zip_paths = state_source_zip_paths(state_name)
    source_data_available = any(path.exists() for path in source_zip_paths)

    signature = source_signature(source_zip_paths)
    debug["source_signature"] = signature
    if geography == "Congressional Districts":
        cached_districts = load_district_cache(state_name, signature, warnings)
        cache_has_elections = cached_districts is not None and has_usable_numeric_data(cached_districts, "dem_share")
        if cached_districts is not None and (not load_precinct_elections or cache_has_elections):
            debug["cache"] = "used processed congressional district cache"
            debug["files"] = {"district_cache": str(district_cache_paths(state_name)[0])}
            debug["detected_columns"]["district_boundaries"] = detect_columns(
                cached_districts.drop(columns="geometry", errors="ignore")
            )
            return {
                "districts": cached_districts,
                "precincts": None,
                "precinct_attributes": None,
                "warnings": warnings,
                "debug": debug,
            }
        if cached_districts is not None and load_precinct_elections and not cache_has_elections:
            debug["cache"] = "ignored cache because it does not include precinct election vote-share columns"

    if not source_data_available:
        deploy_districts = load_deploy_district_data(state_name, warnings)
        if deploy_districts is not None:
            debug["mode"] = "deploy-ready compact data"
            debug["files"] = {
                "deploy_districts": str(deploy_district_path(state_name)),
                "local_source_data": "not found",
            }
            debug["detected_columns"]["district_boundaries"] = detect_columns(
                deploy_districts.drop(columns="geometry", errors="ignore")
            )
            if geography == "Precincts":
                warnings.append(
                    "Precinct view needs the full local source ZIP files and is not available in this deployed version."
                )
            return {
                "districts": deploy_districts,
                "precincts": None,
                "precinct_attributes": None,
                "warnings": warnings,
                "debug": debug,
            }

    required_zips = [config["district_zip"]]
    if geography == "Precincts":
        required_zips.append(config["precinct_zip"])
    elif geography == "Congressional Districts":
        required_zips.extend([config["cvap_zip"], config["l2_zip"]])
        if load_precinct_elections:
            required_zips.append(config["precinct_zip"])
        else:
            warnings.append(
                f"{state_name}: skipped full precinct election loading for a faster start. "
                "Turn on the precinct election option in the sidebar to calculate vote-share layers."
            )

    extracted = unzip_data(state_folder, warnings, required_zips)

    district_files = find_geospatial_files(extracted.get(config["district_zip"]))
    precinct_files = find_geospatial_files(extracted.get(config["precinct_zip"]))
    cvap_files = find_geospatial_files(extracted.get(config["cvap_zip"]))
    l2_files = find_geospatial_files(extracted.get(config["l2_zip"]))

    district_path = choose_best_file(district_files, "district", "geo")
    district_assignment_path = choose_best_file(district_files, "district_assignment", "table")
    precinct_path = choose_best_file(precinct_files, "precinct", "geo")
    cvap_path = choose_best_file(cvap_files, "cvap", "table")
    l2_path = choose_best_file(l2_files, "l2", "table")

    debug["files"] = {
        "district_boundaries": str(district_path) if district_path else None,
        "district_block_assignment": str(district_assignment_path) if district_assignment_path else None,
        "precincts": str(precinct_path) if precinct_path else None,
        "cvap_table": str(cvap_path) if cvap_path else None,
        "l2_table": str(l2_path) if l2_path else None,
    }

    district_gdf = load_geodata(district_path, f"{state_name} congressional districts", warnings)
    precinct_gdf = None
    precinct_attributes = None
    if geography == "Precincts":
        precinct_gdf = load_geodata_sample(
            precinct_path,
            f"{state_name} precincts",
            warnings,
            max_features=max_precinct_features,
        )
    elif load_precinct_elections:
        precinct_attributes = load_geodata_attributes(
            precinct_path,
            f"{state_name} precincts",
            warnings,
        )

    district_assignment = load_table_data(district_assignment_path, f"{state_name} district block assignment", warnings)
    cvap_table = load_table_data(cvap_path, f"{state_name} CVAP table", warnings)
    l2_table = load_table_data(l2_path, f"{state_name} L2 table", warnings)

    if precinct_gdf is not None:
        precinct_gdf, precinct_detected = process_geodataframe(
            precinct_gdf,
            f"{state_name} precincts",
            warnings,
            warn_missing=False,
        )
    elif precinct_attributes is not None:
        precinct_attributes, precinct_detected = process_geodataframe(
            precinct_attributes,
            f"{state_name} precinct attributes",
            warnings,
            warn_missing=False,
        )
    else:
        precinct_detected = {}

    if district_gdf is not None:
        district_gdf, district_detected = process_geodataframe(
            district_gdf,
            f"{state_name} congressional districts",
            warnings,
            warn_missing=False,
        )

        precinct_source = precinct_attributes if precinct_attributes is not None else precinct_gdf
        precinct_metrics = aggregate_precincts_to_districts(precinct_source, district_gdf, warnings)
        district_gdf = merge_district_metrics(district_gdf, precinct_metrics, warnings, "Precinct election aggregation")

        cvap_metrics = aggregate_block_table_to_district(cvap_table, district_assignment, "CVAP district aggregation", warnings)
        if cvap_metrics is not None:
            district_gdf = merge_district_metrics(district_gdf, cvap_metrics, warnings, "CVAP district aggregation")
            detected = detect_columns(district_gdf.drop(columns="geometry", errors="ignore"))
            district_gdf = calculate_cvap_columns(district_gdf, detected, warnings, f"{state_name} district CVAP")

        l2_metrics = aggregate_block_table_to_district(l2_table, district_assignment, "L2 district aggregation", warnings)
        if l2_metrics is not None:
            district_gdf = merge_district_metrics(district_gdf, l2_metrics, warnings, "L2 district aggregation")
            detected = detect_columns(district_gdf.drop(columns="geometry", errors="ignore"))
            district_gdf = calculate_l2_columns(district_gdf, detected, warnings, f"{state_name} district L2")

        district_detected = detect_columns(district_gdf.drop(columns="geometry", errors="ignore"))
        if geography == "Congressional Districts":
            try:
                write_district_cache(state_name, signature, district_gdf)
                debug["cache"] = "wrote processed congressional district cache"
            except Exception as exc:
                warnings.append(f"Could not write fast district cache: {exc}")
    else:
        district_detected = {}

    if cvap_table is not None:
        debug["detected_columns"]["cvap_table"] = detect_columns(cvap_table)
    if l2_table is not None:
        debug["detected_columns"]["l2_table"] = detect_columns(l2_table)
    if district_assignment is not None:
        debug["detected_columns"]["district_block_assignment"] = detect_columns(district_assignment)
    debug["detected_columns"]["district_boundaries"] = district_detected
    debug["detected_columns"]["precincts"] = precinct_detected

    return {
        "districts": district_gdf,
        "precincts": precinct_gdf,
        "precinct_attributes": precinct_attributes,
        "warnings": warnings,
        "debug": debug,
    }


# ---------------------------------------------------------------------------
# Map and display helpers
# ---------------------------------------------------------------------------


def layer_colormap(layer_column: str, values: pd.Series) -> cm.LinearColormap:
    """Choose a color palette for the selected map layer."""
    numeric = safe_numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        vmin, vmax = 0, 1
    elif layer_column in {"dem_share", "rep_share"}:
        vmin, vmax = 0, 1
    elif layer_column in PERCENT_COLUMNS:
        vmin = max(0, float(numeric.quantile(0.05)))
        vmax = min(1, float(numeric.quantile(0.95)))
        if vmin == vmax:
            spread = 0.05 if vmin == 0 else max(0.02, abs(vmin) * 0.08)
            vmin = max(0, vmin - spread)
            vmax = min(1, vmax + spread)
    else:
        vmin = float(numeric.quantile(0.02))
        vmax = float(numeric.quantile(0.98))
        if vmin == vmax:
            vmin, vmax = float(numeric.min()), float(numeric.max() or 1)
    if vmin == vmax:
        vmax = vmin + 1

    if layer_column in {"dem_share", "rep_share"}:
        tick_labels = [0, 0.5, 1]
    elif layer_column in PERCENT_COLUMNS:
        tick_labels = [round(vmin, 2), round((vmin + vmax) / 2, 2), round(vmax, 2)]
    else:
        tick_labels = [round(vmin), round((vmin + vmax) / 2), round(vmax)]

    if layer_column == "dem_share":
        color_map = cm.LinearColormap(
            ["#7f0000", "#d73027", "#f7f7f7", "#2b83ba", "#08306b"],
            index=[0, 0.4, 0.5, 0.6, 1],
            vmin=0,
            vmax=1,
            tick_labels=tick_labels,
            max_labels=3,
            text_color="#f8fafc",
        )
    elif layer_column == "rep_share":
        color_map = cm.LinearColormap(
            ["#08306b", "#2b83ba", "#f7f7f7", "#d73027", "#7f0000"],
            index=[0, 0.4, 0.5, 0.6, 1],
            vmin=0,
            vmax=1,
            tick_labels=tick_labels,
            max_labels=3,
            text_color="#f8fafc",
        )
    elif layer_column == "total_dr_votes":
        color_map = cm.LinearColormap(
            ["#fff7bc", "#fec44f", "#f03b20", "#800026"],
            vmin=vmin,
            vmax=vmax,
            tick_labels=tick_labels,
            max_labels=3,
            text_color="#f8fafc",
        )
    elif layer_column == "turnout_rate":
        color_map = cm.LinearColormap(
            ["#f7fcf5", "#c7e9c0", "#41ab5d", "#005a32"],
            vmin=vmin,
            vmax=vmax,
            tick_labels=tick_labels,
            max_labels=3,
            text_color="#f8fafc",
        )
    elif "cvap" in layer_column:
        color_map = cm.LinearColormap(
            ["#f7fcfd", "#bfd3e6", "#8c96c6", "#88419d"],
            vmin=vmin,
            vmax=vmax,
            tick_labels=tick_labels,
            max_labels=3,
            text_color="#f8fafc",
        )
    elif layer_column == "young_voter_turnout":
        color_map = cm.LinearColormap(
            ["#ffffcc", "#a1dab4", "#41b6c4", "#225ea8"],
            vmin=vmin,
            vmax=vmax,
            tick_labels=tick_labels,
            max_labels=3,
            text_color="#f8fafc",
        )
    else:
        color_map = cm.LinearColormap(
            ["#440154", "#31688e", "#35b779", "#fde725"],
            vmin=vmin,
            vmax=vmax,
            tick_labels=tick_labels,
            max_labels=3,
            text_color="#f8fafc",
        )

    if layer_column == "dem_share":
        color_map.caption = "Partisan result: darker blue = more Democratic, darker red = more Republican"
    elif layer_column == "rep_share":
        color_map.caption = "Republican share: darker red = higher Republican vote share"
    elif layer_column == "turnout_rate":
        color_map.caption = "Overall turnout: darker green = higher turnout"
    elif layer_column == "total_dr_votes":
        color_map.caption = "D + R votes: darker color = more Democratic + Republican votes"
    else:
        color_map.caption = LAYER_BY_COLUMN.get(layer_column, {}).get("label", layer_column)
    color_map.width = 360
    color_map.height = 74
    return color_map


def legend_formatter_script(layer_column: str | None) -> str:
    """Format Folium legend tick labels so they are readable in the browser."""
    if layer_column in PERCENT_COLUMNS:
        mode = "percent"
    elif layer_column == "total_dr_votes":
        mode = "votes"
    else:
        mode = ""
    if not mode:
        return ""

    return f"""
    <script>
    (function() {{
        function formatLegendTicks() {{
            document.querySelectorAll(".legend .tick text").forEach(function(label) {{
                var raw = label.textContent.replace(/,/g, "").trim();
                var value = Number(raw);
                if (!Number.isFinite(value)) {{
                    return;
                }}
                if ("{mode}" === "percent") {{
                    label.textContent = Math.round(value * 100) + "%";
                }} else if ("{mode}" === "votes") {{
                    label.textContent = value >= 1000000
                        ? (value / 1000000).toFixed(1) + "M"
                        : Math.round(value / 1000) + "k";
                }}
            }});
        }}
        setTimeout(formatLegendTicks, 250);
        setTimeout(formatLegendTicks, 900);
        setTimeout(formatLegendTicks, 1600);
    }})();
    </script>
    """


def prepare_map_gdf(
    gdf: gpd.GeoDataFrame,
    layer_column: str | None,
    simplify: bool,
    tooltip_columns: list[str],
) -> gpd.GeoDataFrame:
    """Trim, project, simplify, and make geometry safe for Folium."""
    if "geometry" not in gdf.columns:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=4326)

    keep_columns = [column for column in tooltip_columns if column in gdf.columns]
    if layer_column and layer_column in gdf.columns:
        keep_columns.append(layer_column)
    keep_columns = list(dict.fromkeys(keep_columns))

    map_gdf = gdf[keep_columns + ["geometry"]].copy()
    if map_gdf.crs is None:
        map_gdf = map_gdf.set_crs(4326, allow_override=True)
    try:
        map_gdf = map_gdf.to_crs(4326)
    except Exception as exc:
        st.warning(f"Could not convert map geometry to EPSG:4326. Showing the original coordinates. Details: {exc}")
    map_gdf = map_gdf[map_gdf.geometry.notna() & ~map_gdf.geometry.is_empty]

    try:
        map_gdf["geometry"] = map_gdf.geometry.make_valid()
    except (AttributeError, GEOSException, ValueError):
        pass

    if simplify:
        map_gdf["geometry"] = map_gdf.geometry.simplify(0.01, preserve_topology=True)

    if layer_column and layer_column in map_gdf.columns:
        map_gdf[layer_column] = safe_numeric(map_gdf[layer_column])

    for column in keep_columns:
        if column in map_gdf.columns and column != "geometry":
            map_gdf[column] = map_gdf[column].replace([np.inf, -np.inf], np.nan)

    return map_gdf


def geojson_dict(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """Serialize a GeoDataFrame for Folium with safe nulls and strings."""
    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}

    serializable = gdf.copy()
    for column in serializable.columns:
        if column == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].astype(str).replace({"NaT": None, "NaN": None})
        elif pd.api.types.is_object_dtype(serializable[column]) or pd.api.types.is_string_dtype(serializable[column]):
            serializable[column] = serializable[column].map(lambda value: None if pd.isna(value) else str(value))

    return json.loads(serializable.to_json(na="null"))


def format_tooltip_fields(
    gdf: gpd.GeoDataFrame,
    geography: str,
) -> tuple[gpd.GeoDataFrame, list[str], list[str], list[str]]:
    """Add formatted tooltip columns and return field names and aliases."""
    detected = detect_columns(gdf.drop(columns="geometry", errors="ignore"))
    formatted_gdf = gdf.copy()
    fields = []
    aliases = []

    def add_tooltip_value(alias: str, values: pd.Series) -> None:
        field = f"tooltip_{len(fields)}"
        formatted_gdf[field] = values
        fields.append(field)
        aliases.append(alias)

    id_col = detected.get("district_id") if geography == "Congressional Districts" else detected.get("precinct_id")
    if id_col in gdf.columns:
        id_label = "District" if geography == "Congressional Districts" else "Precinct"
        add_tooltip_value(id_label, gdf[id_col].map(lambda value: "Not available" if pd.isna(value) else str(value)))
    if detected.get("county") in gdf.columns:
        add_tooltip_value(
            "County/Parish",
            gdf[detected["county"]].map(lambda value: "Not available" if pd.isna(value) else str(value)),
        )

    if "dem_share" in gdf.columns and "rep_share" in gdf.columns:
        dem_share = safe_numeric(gdf["dem_share"])
        rep_share = safe_numeric(gdf["rep_share"])
        winner = pd.Series(np.nan, index=gdf.index, dtype="object")
        winner_share = pd.Series(np.nan, index=gdf.index, dtype="float64")
        valid = dem_share.notna() & rep_share.notna()
        dem_won = valid & (dem_share >= rep_share)
        rep_won = valid & (rep_share > dem_share)
        winner.loc[dem_won] = "Democratic"
        winner.loc[rep_won] = "Republican"
        winner_share.loc[dem_won] = dem_share.loc[dem_won]
        winner_share.loc[rep_won] = rep_share.loc[rep_won]
        add_tooltip_value("Vote winner", winner.fillna("Not available"))
        add_tooltip_value("Winner share", winner_share.map(format_percent))

    standard_specs = [
        ("Democratic votes", "dem_votes", "number"),
        ("Republican votes", "rep_votes", "number"),
        ("D + R votes", "total_dr_votes", "number"),
        ("Democratic share", "dem_share", "percent"),
        ("Republican share", "rep_share", "percent"),
        ("Overall turnout", "turnout_rate", "percent"),
        ("Black CVAP", "black_cvap_pct", "percent"),
        ("Latino CVAP", "latino_cvap_pct", "percent"),
        ("Asian CVAP", "asian_cvap_pct", "percent"),
        ("White CVAP", "white_cvap_pct", "percent"),
        ("Minority CVAP", "minority_cvap_pct", "percent"),
        ("Young turnout", "young_voter_turnout", "percent"),
    ]

    for alias, source_col, kind in standard_specs:
        if source_col not in gdf.columns:
            continue
        if kind == "number":
            values = formatted_gdf[source_col].map(format_number)
        elif kind == "percent":
            values = formatted_gdf[source_col].map(format_percent)
        else:
            values = formatted_gdf[source_col].map(
                lambda value: "Not available" if pd.isna(value) else str(value)
            )
        add_tooltip_value(alias, values)

    return formatted_gdf, fields, aliases, fields


def add_city_markers(map_object: folium.Map, state_name: str) -> None:
    """Add simple markers for major cities."""
    for city, lat, lon in STATE_CONFIG[state_name]["cities"]:
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            popup=city,
            tooltip=city,
            color="#0f172a",
            fill=True,
            fill_color="#facc15",
            fill_opacity=1,
            weight=2.5,
        ).add_to(map_object)
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                icon_size=(140, 24),
                icon_anchor=(-8, 10),
                class_name="city-label",
                html=f"<span>{city}</span>",
            ),
        ).add_to(map_object)


def add_district_overlay(map_object: folium.Map, district_gdf: gpd.GeoDataFrame | None, simplify: bool) -> None:
    """Draw congressional district boundaries on top of the active map."""
    if district_gdf is None or district_gdf.empty:
        return
    overlay = prepare_map_gdf(district_gdf, None, simplify, [])
    if overlay.empty:
        return
    folium.GeoJson(
        data=geojson_dict(overlay),
        name="Congressional District Boundary Overlay",
        interactive=False,
        style_function=lambda _: {
            "fillOpacity": 0,
            "color": "#f8fafc",
            "weight": 2.2,
            "opacity": 0.9,
        },
    ).add_to(map_object)


def create_folium_map(
    gdf: gpd.GeoDataFrame,
    state_name: str,
    geography: str,
    layer_column: str | None,
    show_district_overlay: bool,
    district_gdf: gpd.GeoDataFrame | None,
    show_cities: bool,
    simplify: bool,
) -> folium.Map:
    """Build the interactive Folium map."""
    config = STATE_CONFIG[state_name]
    map_object = folium.Map(
        location=config["center"],
        zoom_start=config["zoom"],
        tiles="cartodbdark_matter",
        control_scale=True,
    )

    map_object.get_root().html.add_child(folium.Element(
        """
        <style>
        .leaflet-container {
            background: #0b1120;
        }
        .legend {
            background: rgba(15, 23, 42, 0.92) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(148, 163, 184, 0.45) !important;
            border-radius: 6px !important;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35) !important;
            padding: 8px !important;
        }
        .legend text,
        .legend span,
        .legend label {
            color: #f8fafc !important;
            fill: #f8fafc !important;
        }
        .leaflet-control-layers,
        .leaflet-control-zoom a,
        .leaflet-control-scale-line {
            background: rgba(15, 23, 42, 0.92) !important;
            color: #f8fafc !important;
            border-color: rgba(148, 163, 184, 0.5) !important;
        }
        .leaflet-control-attribution {
            background: rgba(15, 23, 42, 0.78) !important;
            color: #cbd5e1 !important;
        }
        .leaflet-control-attribution a {
            color: #93c5fd !important;
        }
        .leaflet-tooltip {
            background: #0f172a !important;
            color: #f8fafc !important;
            border: 1px solid #475569 !important;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35) !important;
        }
        .city-label {
            pointer-events: none;
            z-index: 900 !important;
        }
        .city-label span {
            display: inline-block;
            padding: 2px 7px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(250, 204, 21, 0.75);
            color: #fefce8;
            font-size: 12px;
            font-weight: 700;
            line-height: 1.2;
            text-shadow: 0 1px 2px #000000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
            white-space: nowrap;
        }
        </style>
        """
    ))

    tooltip_gdf, tooltip_fields, tooltip_aliases, tooltip_columns = format_tooltip_fields(gdf, geography)
    map_gdf = prepare_map_gdf(tooltip_gdf, layer_column, simplify, tooltip_columns)

    if map_gdf.empty:
        st.warning("No valid geometry was available for the selected map.")
        return map_object

    if layer_column and layer_column in map_gdf.columns:
        color_map = layer_colormap(layer_column, map_gdf[layer_column])

        def style_function(feature: dict[str, Any]) -> dict[str, Any]:
            value = feature["properties"].get(layer_column)
            numeric = pd.to_numeric(value, errors="coerce")
            if pd.isna(numeric):
                fill_color = "#d9d9d9"
            else:
                fill_color = color_map(float(numeric))
            return {
                "fillColor": fill_color,
                "color": "#f8fafc",
                "weight": 0.65,
                "fillOpacity": 0.9,
                "opacity": 0.85,
            }

        color_map.add_to(map_object)
        formatter = legend_formatter_script(layer_column)
        if formatter:
            map_object.get_root().html.add_child(folium.Element(formatter))
    else:
        style_function = lambda _: {
            "fillColor": "#334155",
            "color": "#cbd5e1",
            "weight": 0.5,
            "fillOpacity": 0.55,
            "opacity": 0.72,
        }

    tooltip = None
    if tooltip_fields:
        tooltip = folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            sticky=True,
            labels=True,
            max_width=360,
        )
    popup = None
    if tooltip_fields:
        popup = folium.GeoJsonPopup(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            labels=True,
            max_width=380,
        )

    folium.GeoJson(
        data=geojson_dict(map_gdf),
        name=geography,
        style_function=style_function,
        tooltip=tooltip,
        popup=popup,
        highlight_function=lambda _: {"weight": 2.8, "color": "#f8fafc", "fillOpacity": 0.9},
    ).add_to(map_object)

    if show_district_overlay:
        add_district_overlay(map_object, district_gdf, simplify)
    if show_cities:
        add_city_markers(map_object, state_name)

    folium.LayerControl(collapsed=True).add_to(map_object)
    return map_object


# ---------------------------------------------------------------------------
# Outputs, summaries, and charts
# ---------------------------------------------------------------------------


def cleaned_export_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep export columns useful and serializable."""
    export_gdf = gdf.copy()
    for column in export_gdf.columns:
        if column != "geometry" and pd.api.types.is_object_dtype(export_gdf[column]):
            export_gdf[column] = export_gdf[column].astype(str).replace({"nan": None, "None": None})
    if export_gdf.crs is None:
        export_gdf = export_gdf.set_crs(4326, allow_override=True)
    return export_gdf.to_crs(4326)


def summarize_metrics(gdf: gpd.GeoDataFrame | None) -> dict[str, Any]:
    """Build simple statewide/current-geography summary metrics."""
    if gdf is None or gdf.empty:
        return {}

    summary: dict[str, Any] = {"rows": len(gdf)}
    if "dem_votes" in gdf.columns:
        summary["total_democratic_votes"] = safe_numeric(gdf["dem_votes"]).sum()
    if "rep_votes" in gdf.columns:
        summary["total_republican_votes"] = safe_numeric(gdf["rep_votes"]).sum()
    if "dem_votes" in gdf.columns and "rep_votes" in gdf.columns:
        dem_total = safe_numeric(gdf["dem_votes"]).sum()
        rep_total = safe_numeric(gdf["rep_votes"]).sum()
        dr_total = dem_total + rep_total
        summary["statewide_democratic_share"] = dem_total / dr_total if dr_total else np.nan
        summary["statewide_republican_share"] = rep_total / dr_total if dr_total else np.nan
    if "turnout_rate" in gdf.columns:
        summary["average_turnout_rate"] = safe_numeric(gdf["turnout_rate"]).mean()
    return summary


def save_outputs(state_name: str, state_data: dict[str, Any]) -> list[str]:
    """Save cleaned GeoJSONs and summary CSVs into outputs/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    abbr = STATE_CONFIG[state_name]["abbr"]
    messages = []

    for key, label in [("districts", "congressional_districts"), ("precincts", "precincts")]:
        gdf = state_data.get(key)
        if gdf is None or gdf.empty:
            continue

        geojson_path = OUTPUT_DIR / f"{abbr}_{label}_cleaned.geojson"
        summary_path = OUTPUT_DIR / f"{abbr}_{label}_summary.csv"

        if not geojson_path.exists():
            try:
                cleaned_export_gdf(gdf).to_file(geojson_path, driver="GeoJSON")
                messages.append(f"Saved {geojson_path.name}")
            except Exception as exc:
                messages.append(f"Could not save {geojson_path.name}: {exc}")
        else:
            messages.append(f"{geojson_path.name} already exists")

        summary_columns = [
            column
            for column in [
                "dem_votes",
                "rep_votes",
                "total_dr_votes",
                "dem_share",
                "rep_share",
                "turnout_rate",
                "black_cvap_pct",
                "latino_cvap_pct",
                "asian_cvap_pct",
                "white_cvap_pct",
                "minority_cvap_pct",
                "young_voter_turnout",
            ]
            if column in gdf.columns
        ]
        detected = detect_columns(gdf.drop(columns="geometry", errors="ignore"))
        id_column = detected.get("district_id") if key == "districts" else detected.get("precinct_id")
        if id_column in gdf.columns:
            summary_columns = [id_column] + summary_columns
        gdf.drop(columns="geometry", errors="ignore")[summary_columns].to_csv(summary_path, index=False)
        messages.append(f"Saved {summary_path.name}")

    overall_summary = []
    for key, label in [("districts", "congressional_districts"), ("precincts", "precincts")]:
        summary = summarize_metrics(state_data.get(key))
        for metric, value in summary.items():
            overall_summary.append({"geography": label, "metric": metric, "value": value})
    if overall_summary:
        path = OUTPUT_DIR / f"{abbr}_summary.csv"
        pd.DataFrame(overall_summary).to_csv(path, index=False)
        messages.append(f"Saved {path.name}")

    return messages


def display_summary_metrics(gdf: gpd.GeoDataFrame) -> None:
    """Show high-level vote and turnout numbers."""
    summary = summarize_metrics(gdf)
    columns = st.columns(5)

    columns[0].metric("Total Democratic Votes", format_number(summary.get("total_democratic_votes")))
    columns[1].metric("Total Republican Votes", format_number(summary.get("total_republican_votes")))
    columns[2].metric("Statewide Democratic Share", format_percent(summary.get("statewide_democratic_share")))
    columns[3].metric("Statewide Republican Share", format_percent(summary.get("statewide_republican_share")))
    columns[4].metric("Average Turnout Rate", format_percent(summary.get("average_turnout_rate")))


def display_charts(gdf: gpd.GeoDataFrame, geography: str) -> None:
    """Draw the requested top-10 bar charts."""
    detected = detect_columns(gdf.drop(columns="geometry", errors="ignore"))
    id_column = detected.get("district_id") if geography == "Congressional Districts" else detected.get("precinct_id")
    if id_column not in gdf.columns:
        id_column = gdf.index.name or "index"
        chart_df = gdf.reset_index().rename(columns={"index": id_column})
    else:
        chart_df = gdf.copy()

    left, right = st.columns(2)

    with left:
        st.subheader("Top 10 by D + R Vote Total")
        if has_usable_numeric_data(chart_df, "total_dr_votes"):
            top_votes = chart_df[[id_column, "total_dr_votes"]].copy()
            top_votes["total_dr_votes"] = safe_numeric(top_votes["total_dr_votes"])
            top_votes = top_votes.nlargest(10, "total_dr_votes").set_index(id_column)
            st.bar_chart(top_votes)
        else:
            st.info("D + R vote total was not available for this geography.")

    with right:
        st.subheader("Top 10 by Turnout Rate")
        if has_usable_numeric_data(chart_df, "turnout_rate"):
            top_turnout = chart_df[[id_column, "turnout_rate"]].copy()
            top_turnout["turnout_rate"] = safe_numeric(top_turnout["turnout_rate"])
            top_turnout = top_turnout.dropna().nlargest(10, "turnout_rate").set_index(id_column)
            st.bar_chart(top_turnout)
        else:
            st.info("Turnout rate was not available for this geography.")


def download_current_data(gdf: gpd.GeoDataFrame, state_name: str, geography: str) -> None:
    """Add a CSV download button for the current selected data."""
    csv_data = gdf.drop(columns="geometry", errors="ignore").to_csv(index=False).encode("utf-8")
    filename = f"{STATE_CONFIG[state_name]['abbr']}_{geography.lower().replace(' ', '_')}_current_data.csv"
    st.download_button(
        label="Download Current Data as CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
    )


def available_layer_definitions(gdf: gpd.GeoDataFrame) -> list[dict[str, str]]:
    """Only show layer definitions that can actually be drawn."""
    return [
        layer
        for layer in LAYER_DEFINITIONS
        if layer["column"] in gdf.columns and has_usable_numeric_data(gdf, layer["column"])
    ]


def layer_categories(layers: list[dict[str, str]]) -> list[str]:
    """Preserve the configured layer category order."""
    return list(dict.fromkeys(layer["category"] for layer in layers))


def map_status_text(state_name: str, geography: str, layer_label: str, layer_column: str | None) -> str:
    """Write a plain-language sentence above the map."""
    base = f"Showing {state_name} {geography.lower()}."
    if layer_column == "dem_share":
        return (
            f"{base} Darker blue means a higher Democratic share; darker red means a higher "
            "Republican share. Hover or click a district for exact percentages."
        )
    if layer_column == "rep_share":
        return f"{base} Darker red means a higher Republican vote share. Hover or click for exact percentages."
    if layer_column == "total_dr_votes":
        return f"{base} Darker colors mean more Democratic plus Republican votes. Hover or click for exact vote totals."
    if layer_column == "turnout_rate":
        return f"{base} Darker green means higher turnout. Hover or click for the turnout percentage."
    if layer_column and "cvap" in layer_column:
        return f"{base} Darker purple means a higher {layer_label} share. Hover or click for exact CVAP percentages."
    if layer_column == "young_voter_turnout":
        return f"{base} Darker blue-green means higher young voter turnout. Hover or click for exact percentages."
    return f"{base} Hover or click a district for available statistics."


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="DistrictLens: California & Louisiana Redistricting Explorer",
        layout="wide",
    )
    inject_dark_theme_css()
    st.title("DistrictLens: California & Louisiana Redistricting Explorer")

    with st.sidebar:
        st.header("Map Controls")
        st.caption(f"App version: {APP_VERSION}")
        st.markdown("### 1. Place")
        state_name = st.selectbox("State", list(STATE_CONFIG.keys()))
        source_data_available = has_state_source_data(state_name)
        geography_options = ["Congressional Districts", "Precincts"] if source_data_available else ["Congressional Districts"]
        geography = st.selectbox("Geography", geography_options)
        if not source_data_available:
            st.caption("Deployment mode: compact congressional district data is available; precinct ZIP files are not bundled.")

        st.markdown("### 2. Data Loading")
        max_precinct_features = MAX_MAP_FEATURES
        load_precinct_elections = False
        if geography == "Congressional Districts" and source_data_available:
            load_precinct_elections = st.checkbox(
                "Recalculate district votes from precincts",
                value=state_name == "Louisiana",
                help=(
                    "Reads the large raw precinct file and aggregates Democratic/Republican votes to districts. "
                    "If a district vote cache already exists, the map may look the same."
                ),
            )
        elif geography == "Congressional Districts":
            st.caption(
                "The deployed app already includes precomputed district vote totals. "
                "Raw precinct ZIP files are too large to bundle here, so there is no precinct reload button online."
            )
        if geography == "Precincts":
            max_precinct_features = st.slider(
                "Precinct features to draw",
                min_value=500,
                max_value=20000,
                value=MAX_MAP_FEATURES,
                step=500,
                help="Drawing every California precinct can freeze the browser. Increase this only if your computer can handle it.",
            )

        st.markdown("### 3. Display")
        show_district_overlay = st.checkbox("Show congressional district boundary overlay", value=True)
        show_cities = st.checkbox("Show major city markers", value=True)
        simplify = st.checkbox("Simplify geometry for faster performance", value=True)

        st.markdown("### 4. Export")
        save_outputs_requested = st.button("Save cleaned outputs")

    with st.spinner(f"Loading {state_name} data..."):
        state_data = load_state_data(
            state_name,
            geography,
            max_precinct_features,
            load_precinct_elections,
            APP_VERSION,
        )

    for warning in state_data["warnings"]:
        if "skipped full precinct election loading" in warning or "showing the first" in warning:
            st.info(warning)
        else:
            st.warning(warning)

    active_key = "districts" if geography == "Congressional Districts" else "precincts"
    active_gdf = state_data.get(active_key)
    district_gdf = state_data.get("districts")

    if active_gdf is None or active_gdf.empty:
        st.error(f"No usable {geography.lower()} geography was loaded for {state_name}.")
        with st.expander("Debug: Loaded Data and Columns", expanded=True):
            st.json(state_data["debug"])
        return

    available_layers = available_layer_definitions(active_gdf)
    with st.sidebar:
        st.markdown("### 5. Map Theme")
        if available_layers:
            categories = layer_categories(available_layers)
            default_category_index = 0
            if "Votes" in categories:
                default_category_index = categories.index("Votes")
            layer_category = st.selectbox(
                "Layer category",
                categories,
                index=default_category_index,
                key="layer_category",
            )
            category_layers = [layer for layer in available_layers if layer["category"] == layer_category]
            selected_layer_label = st.selectbox(
                "Map layer",
                [layer["label"] for layer in category_layers],
                key="layer_label",
            )
            selected_layer = next(layer for layer in category_layers if layer["label"] == selected_layer_label)
            layer_label = selected_layer["label"]
            layer_column = selected_layer["column"]
            st.caption(selected_layer["description"])
        else:
            st.info("No calculated numeric layers are available; showing boundaries only.")
            layer_label = "Boundaries only"
            layer_column = None

        if not any(layer["column"] == "young_voter_turnout" for layer in available_layers):
            st.caption("Young turnout was not detected for this geography.")

    if save_outputs_requested:
        with st.spinner("Saving cleaned outputs..."):
            save_messages = save_outputs(state_name, state_data)
        with st.expander("Output Files", expanded=False):
            for message in save_messages:
                st.write(message)

    st.caption(map_status_text(state_name, geography, layer_label, layer_column))
    if geography == "Precincts":
        st.info(
            f"For speed, this view draws up to {max_precinct_features:,} precinct features. "
            "Congressional Districts view uses full attribute data for district summaries."
        )
    map_object = create_folium_map(
        active_gdf,
        state_name,
        geography,
        layer_column,
        show_district_overlay,
        district_gdf,
        show_cities,
        simplify,
    )
    st_folium(map_object, height=720, use_container_width=True)

    with st.expander("Debug: Loaded Data and Columns", expanded=False):
        st.json(state_data["debug"])
        st.write("Active data columns:")
        st.write(list(active_gdf.columns))

    st.subheader("Summary Metrics")
    display_summary_metrics(active_gdf)

    st.subheader("Charts")
    display_charts(active_gdf, geography)

    st.subheader("Download")
    download_current_data(active_gdf, state_name, geography)

    st.subheader("About This Map")
    st.markdown(
        """
        **D + R vote total** means Democratic votes plus Republican votes. It is a two-party election vote total,
        not total population.

        **Turnout rate** is calculated as actual votes divided by registered voters when registered voter data exists.
        If that exact calculation is not possible, the app uses an already-provided turnout field when one is detected.

        **CVAP** means Citizen Voting Age Population. CVAP layers show the share of voting-age citizens in each
        district or precinct when the app can join the CVAP data to that geography.

        California can show urban/rural, coastal/inland, and partisan vote-share patterns. Louisiana can show how
        district boundaries relate to racial representation and voting-age population patterns. The app uses flexible
        column detection, so some layers may appear for congressional districts but not precincts when the source files
        do not share a reliable join key.
        """
    )


if __name__ == "__main__":
    main()
