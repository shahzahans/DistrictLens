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
from folium.plugins import Draw
from pandas.errors import PerformanceWarning
from shapely.geometry import shape
from shapely.errors import GEOSException
from shapely.ops import unary_union
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
MIN_HYPOTHETICAL_OVERLAP_SHARE = 0.0025
APP_VERSION = "2026-05-20 editable-boundary-plan"

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
        "label": "Party winner",
        "column": "dem_share",
        "description": "Blue means Democratic winner and red means Republican winner. California uses darker color for more party votes; Louisiana uses darker color for stronger wins.",
    },
    {
        "category": "Votes",
        "label": "D + R votes",
        "column": "total_dr_votes",
        "description": "Democratic plus Republican votes. This is not total population.",
    },
    {
        "category": "Votes",
        "label": "Winner margin",
        "column": "winner_margin",
        "description": "Absolute two-party winner margin. Lower values are more competitive; higher values are safer.",
    },
    {
        "category": "Turnout",
        "label": "Overall turnout",
        "column": "turnout_rate",
        "description": "Votes divided by registered voters. Light green is lower turnout; dark green is higher turnout.",
    },
    {
        "category": "Turnout",
        "label": "Young turnout",
        "column": "young_voter_turnout",
        "description": "Estimated young voter turnout. Light purple is lower; dark purple is higher.",
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
    {
        "category": "Redistricting metrics",
        "label": "Compactness",
        "column": "compactness_polsby_popper",
        "description": "Polsby-Popper compactness score from 0 to 1. Higher values are more compact.",
    },
    {
        "category": "Redistricting metrics",
        "label": "District area",
        "column": "district_area_sq_mi",
        "description": "Estimated district area in square miles, calculated from projected geometry.",
    },
    {
        "category": "Redistricting metrics",
        "label": "District perimeter",
        "column": "district_perimeter_mi",
        "description": "Estimated district perimeter in miles, calculated from projected geometry.",
    },
]
LAYER_OPTIONS = [(layer["label"], layer["column"]) for layer in LAYER_DEFINITIONS]
LAYER_BY_COLUMN = {layer["column"]: layer for layer in LAYER_DEFINITIONS}

PERCENT_COLUMNS = {
    "dem_share",
    "rep_share",
    "winner_margin",
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


def format_percentage_points(value: Any) -> str:
    number = safe_numeric(pd.Series([value])).iloc[0]
    if pd.isna(number):
        return "Not available"
    return f"{number * 100:.1f} pp"


def format_rank(value: Any, total: int) -> str:
    number = safe_numeric(pd.Series([value])).iloc[0]
    if pd.isna(number):
        return "Not available"
    return f"#{int(number)} of {total}"


def format_competitiveness_score(value: Any) -> str:
    number = safe_numeric(pd.Series([value])).iloc[0]
    if pd.isna(number):
        return "Not available"
    return f"{number:.0f}/100"


def format_decimal_score(value: Any) -> str:
    number = safe_numeric(pd.Series([value])).iloc[0]
    if pd.isna(number):
        return "Not available"
    return f"{number:.3f}"


def format_square_miles(value: Any) -> str:
    number = safe_numeric(pd.Series([value])).iloc[0]
    if pd.isna(number):
        return "Not available"
    return f"{number:,.0f} sq mi"


def format_miles(value: Any) -> str:
    number = safe_numeric(pd.Series([value])).iloc[0]
    if pd.isna(number):
        return "Not available"
    return f"{number:,.0f} mi"


def format_district_label(value: Any) -> str:
    if pd.isna(value):
        return "Not available"
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return str(int(float(text)))
    return text


def natural_sort_key(value: Any) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


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
    elif layer_column in {"turnout_rate", "young_voter_turnout"}:
        vmin = max(0, float(numeric.quantile(0.05)))
        vmax = min(1, float(numeric.quantile(0.95)))
        if vmin == vmax:
            spread = 0.05 if vmin == 0 else max(0.02, abs(vmin) * 0.08)
            vmin = max(0, vmin - spread)
            vmax = min(1, vmax + spread)
    elif layer_column == "compactness_polsby_popper":
        vmin = max(0, float(numeric.quantile(0.05)))
        vmax = min(1, float(numeric.quantile(0.95)))
        if vmin == vmax:
            spread = max(0.02, abs(vmin) * 0.12)
            vmin = max(0, vmin - spread)
            vmax = min(1, vmax + spread)
    elif layer_column in PERCENT_COLUMNS:
        vmin, vmax = 0, 1
    else:
        vmin = float(numeric.quantile(0.02))
        vmax = float(numeric.quantile(0.98))
        if vmin == vmax:
            vmin, vmax = float(numeric.min()), float(numeric.max() or 1)
    if vmin == vmax:
        vmax = vmin + 1

    if layer_column in {"dem_share", "rep_share"}:
        tick_labels = [0, 0.5, 1]
    elif layer_column in {"turnout_rate", "young_voter_turnout"}:
        tick_labels = [0, 0.5, 1]
    elif layer_column == "compactness_polsby_popper":
        tick_labels = [round(vmin, 2), round((vmin + vmax) / 2, 2), round(vmax, 2)]
    elif layer_column in PERCENT_COLUMNS:
        tick_labels = [0, 0.5, 1]
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
            ["#f7fcf5", "#c7e9c0", "#41ab5d", "#00441b"],
            vmin=vmin,
            vmax=vmax,
            tick_labels=tick_labels,
            max_labels=3,
            text_color="#f8fafc",
        )
    elif layer_column == "winner_margin":
        color_map = cm.LinearColormap(
            ["#ffffcc", "#fdae61", "#d7191c"],
            vmin=vmin,
            vmax=vmax,
            tick_labels=tick_labels,
            max_labels=3,
            text_color="#f8fafc",
        )
    elif layer_column == "compactness_polsby_popper":
        color_map = cm.LinearColormap(
            ["#f7fcf5", "#c7e9c0", "#41ab5d", "#00441b"],
            vmin=vmin,
            vmax=vmax,
            tick_labels=tick_labels,
            max_labels=3,
            text_color="#f8fafc",
        )
    elif layer_column in {"district_area_sq_mi", "district_perimeter_mi"}:
        color_map = cm.LinearColormap(
            ["#f7fbff", "#c6dbef", "#6baed6", "#08306b"],
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
            ["#f7f4f9", "#cbc9e2", "#807dba", "#3f007d"],
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

    color_map.caption = ""
    color_map.width = 360
    color_map.height = 56
    return color_map


def clamp_unit_interval(value: float) -> float:
    """Keep a numeric value inside the 0 to 1 range."""
    return max(0.0, min(1.0, value))


def interpolate_hex_color(light_hex: str, dark_hex: str, amount: float) -> str:
    """Blend between a light and dark color using a 0 to 1 amount."""
    amount = clamp_unit_interval(amount)
    light = light_hex.lstrip("#")
    dark = dark_hex.lstrip("#")
    blended = []
    for index in range(0, 6, 2):
        start = int(light[index:index + 2], 16)
        end = int(dark[index:index + 2], 16)
        blended.append(round(start + (end - start) * amount))
    return "#" + "".join(f"{channel:02x}" for channel in blended)


def practical_value_bounds(values: pd.Series) -> tuple[float, float]:
    """Find a useful low/high range for light-to-dark shading."""
    numeric = safe_numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return 0.0, 1.0

    if len(numeric) <= 20:
        vmin = float(numeric.min())
        vmax = float(numeric.max())
    else:
        vmin = float(numeric.quantile(0.05))
        vmax = float(numeric.quantile(0.95))

    if vmin == vmax:
        spread = max(1.0, abs(vmin) * 0.08)
        vmin -= spread
        vmax += spread
    return vmin, vmax


def party_vote_count_bounds(gdf: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Find separate vote-count ranges for Democratic-won and Republican-won areas."""
    dem_votes = safe_numeric(gdf.get("dem_votes", pd.Series(dtype="float64")))
    rep_votes = safe_numeric(gdf.get("rep_votes", pd.Series(dtype="float64")))
    valid = dem_votes.notna() & rep_votes.notna()
    dem_winner_votes = dem_votes[valid & (dem_votes >= rep_votes)]
    rep_winner_votes = rep_votes[valid & (rep_votes > dem_votes)]
    return {
        "dem": practical_value_bounds(dem_winner_votes if not dem_winner_votes.empty else dem_votes),
        "rep": practical_value_bounds(rep_winner_votes if not rep_winner_votes.empty else rep_votes),
    }


def party_vote_count_color(properties: dict[str, Any], vote_bounds: dict[str, tuple[float, float]]) -> str:
    """Color by party winner, then shade by raw Democratic or Republican votes."""
    dem_votes = pd.to_numeric(properties.get("dem_votes"), errors="coerce")
    rep_votes = pd.to_numeric(properties.get("rep_votes"), errors="coerce")

    if pd.isna(dem_votes) or pd.isna(rep_votes):
        return party_winner_strength_color(properties)

    if dem_votes >= rep_votes:
        vmin, vmax = vote_bounds.get("dem", (0.0, 1.0))
        volume = 0.5 if vmax == vmin else clamp_unit_interval((float(dem_votes) - vmin) / (vmax - vmin))
        return interpolate_hex_color("#c7e9ff", "#0050a4", volume)

    vmin, vmax = vote_bounds.get("rep", (0.0, 1.0))
    volume = 0.5 if vmax == vmin else clamp_unit_interval((float(rep_votes) - vmin) / (vmax - vmin))
    return interpolate_hex_color("#f8c7c2", "#99000d", volume)


def party_winner_strength_color(properties: dict[str, Any]) -> str:
    """Color by party winner, then shade by how far the winner is above 50%."""
    dem_share = pd.to_numeric(properties.get("dem_share"), errors="coerce")
    rep_share = pd.to_numeric(properties.get("rep_share"), errors="coerce")

    if pd.isna(dem_share) or pd.isna(rep_share):
        return "#d9d9d9"

    if dem_share >= rep_share:
        strength = clamp_unit_interval((float(dem_share) - 0.5) / 0.5)
        return interpolate_hex_color("#c7e9ff", "#0050a4", strength)

    strength = clamp_unit_interval((float(rep_share) - 0.5) / 0.5)
    return interpolate_hex_color("#f8c7c2", "#99000d", strength)


def party_winner_legend_html(title: str) -> str:
    """Create a fixed red/blue legend for the party winner layer."""
    return """
    <style>
    .party-volume-legend {
        position: fixed !important;
        top: 14px !important;
        right: 14px !important;
        z-index: 999999 !important;
        background: rgba(15, 23, 42, 0.92) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(148, 163, 184, 0.45) !important;
        border-radius: 6px !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35) !important;
        padding: 9px 12px 10px 12px !important;
        width: 360px !important;
        pointer-events: none !important;
    }
    .party-volume-ticks {
        display: flex;
        justify-content: space-between;
        padding-left: 22px;
        color: #f8fafc;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 3px;
    }
    .party-volume-row {
        display: grid;
        grid-template-columns: 18px 1fr;
        align-items: center;
        gap: 6px;
        margin-top: 4px;
        color: #f8fafc;
        font-size: 12px;
        font-weight: 800;
    }
    .party-volume-bar {
        height: 14px;
        border-radius: 1px;
    }
    .party-volume-dem {
        background: linear-gradient(90deg, #c7e9ff 0%, #5aa5d8 50%, #0050a4 100%);
    }
    .party-volume-rep {
        background: linear-gradient(90deg, #f8c7c2 0%, #dc5a50 50%, #99000d 100%);
    }
    .party-volume-title {
        color: #f8fafc;
        font-size: 13px;
        font-weight: 700;
        line-height: 1.2;
        margin: 7px 0 0 24px;
        white-space: nowrap;
    }
    @media (max-width: 700px) {
        .party-volume-legend {
            width: 280px !important;
            right: 8px !important;
        }
    }
    </style>
    <div class="party-volume-legend leaflet-control">
        <div class="party-volume-ticks">
            <span>0.0</span><span>0.5</span><span>1.0</span>
        </div>
        <div class="party-volume-row">
            <span>D</span><div class="party-volume-bar party-volume-dem"></div>
        </div>
        <div class="party-volume-row">
            <span>R</span><div class="party-volume-bar party-volume-rep"></div>
        </div>
        <div class="party-volume-title">""" + title + """</div>
    </div>
    """


def fixed_decimal_legend_html(title: str, color_map: cm.LinearColormap) -> str:
    """Create a fixed 0.0, 0.5, 1.0 legend without changing map colors."""
    sample_values = np.linspace(float(color_map.vmin), float(color_map.vmax), 16)
    stops = []
    for index, value in enumerate(sample_values):
        position = round(index * 100 / (len(sample_values) - 1), 2)
        stops.append(f"{color_map.rgb_hex_str(float(value))} {position}%")
    gradient = ", ".join(stops)

    return f"""
    <style>
    .fixed-decimal-legend {{
        position: fixed !important;
        top: 14px !important;
        right: 14px !important;
        z-index: 999999 !important;
        background: rgba(15, 23, 42, 0.92) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(148, 163, 184, 0.45) !important;
        border-radius: 6px !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35) !important;
        padding: 9px 12px 10px 12px !important;
        width: 360px !important;
        pointer-events: none !important;
    }}
    .fixed-decimal-ticks {{
        display: flex;
        justify-content: space-between;
        color: #f8fafc;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 3px;
    }}
    .fixed-decimal-bar {{
        height: 30px;
        border-radius: 1px;
        background: linear-gradient(90deg, {gradient});
    }}
    .fixed-decimal-title {{
        color: #f8fafc;
        font-size: 13px;
        font-weight: 700;
        line-height: 1.2;
        margin-top: 7px;
        white-space: nowrap;
    }}
    @media (max-width: 700px) {{
        .fixed-decimal-legend {{
            width: 280px !important;
            right: 8px !important;
        }}
    }}
    </style>
    <div class="fixed-decimal-legend leaflet-control">
        <div class="fixed-decimal-ticks">
            <span>0.0</span><span>0.5</span><span>1.0</span>
        </div>
        <div class="fixed-decimal-bar"></div>
        <div class="fixed-decimal-title">{title}</div>
    </div>
    """


def legend_title(layer_column: str | None) -> str:
    """Return the short title shown below the legend scale."""
    if layer_column == "dem_share":
        return "Party winner"
    if layer_column == "rep_share":
        return "Republican share"
    if layer_column == "turnout_rate":
        return "Overall turnout"
    if layer_column == "total_dr_votes":
        return "D + R votes"
    if layer_column == "winner_margin":
        return "Winner margin"
    if layer_column == "compactness_polsby_popper":
        return "Compactness"
    if layer_column == "district_area_sq_mi":
        return "District area"
    if layer_column == "district_perimeter_mi":
        return "District perimeter"
    if layer_column:
        return LAYER_BY_COLUMN.get(layer_column, {}).get("label", layer_column)
    return "Scale"


def legend_formatter_script(layer_column: str | None) -> str:
    """Format Folium legends so tick labels are readable and captions sit below the scale."""
    if layer_column in PERCENT_COLUMNS:
        mode = "percent"
    elif layer_column == "total_dr_votes":
        mode = "votes"
    elif layer_column == "district_area_sq_mi":
        mode = "sq_mi"
    elif layer_column == "district_perimeter_mi":
        mode = "miles"
    else:
        mode = ""
    title = json.dumps(legend_title(layer_column))

    return f"""
    <style>
    .legend.leaflet-control::after {{
        content: {title};
        display: block;
        color: #f8fafc;
        font-size: 13px;
        font-weight: 700;
        line-height: 1.2;
        margin: -2px 0 2px 25px;
        max-width: 340px;
        white-space: nowrap;
    }}
    </style>
    <script>
    (function() {{
        function formatLegend() {{
            var legend = document.querySelector(".legend.leaflet-control");
            if (!legend) {{
                return;
            }}
            legend.querySelectorAll("svg").forEach(function(svg) {{
                svg.setAttribute("height", "56");
            }});
            document.querySelectorAll(".legend rect").forEach(function(rect) {{
                rect.setAttribute("height", "30");
            }});
            document.querySelectorAll(".legend .caption").forEach(function(caption) {{
                caption.textContent = "";
                caption.style.display = "none";
            }});
            document.querySelectorAll(".legend .tick text").forEach(function(label) {{
                var raw = label.textContent.replace(/,/g, "").trim();
                var value = Number(raw);
                if (!Number.isFinite(value)) {{
                    return;
                }}
                if ("{mode}" === "percent") {{
                    label.textContent = value.toFixed(1);
                }} else if ("{mode}" === "votes") {{
                    label.textContent = value >= 1000000
                        ? (value / 1000000).toFixed(1) + "M"
                        : Math.round(value / 1000) + "k";
                }} else if ("{mode}" === "sq_mi") {{
                    label.textContent = Math.round(value).toLocaleString() + " sq mi";
                }} else if ("{mode}" === "miles") {{
                    label.textContent = Math.round(value).toLocaleString() + " mi";
                }}
            }});
        }}
        if (document.body) {{
            var observer = new MutationObserver(formatLegend);
            observer.observe(document.body, {{childList: true, subtree: true}});
        }}
        setTimeout(formatLegend, 250);
        setTimeout(formatLegend, 900);
        setTimeout(formatLegend, 1600);
        setTimeout(formatLegend, 2600);
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
    if layer_column == "dem_share":
        for related_column in ("rep_share", "dem_votes", "rep_votes", "total_dr_votes"):
            if related_column in gdf.columns:
                keep_columns.append(related_column)
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
        ("Winner margin", "winner_margin", "percent"),
        ("Overall turnout", "turnout_rate", "percent"),
        ("Black CVAP", "black_cvap_pct", "percent"),
        ("Latino CVAP", "latino_cvap_pct", "percent"),
        ("Asian CVAP", "asian_cvap_pct", "percent"),
        ("White CVAP", "white_cvap_pct", "percent"),
        ("Minority CVAP", "minority_cvap_pct", "percent"),
        ("Young turnout", "young_voter_turnout", "percent"),
        ("Compactness", "compactness_polsby_popper", "score"),
        ("Area", "district_area_sq_mi", "sq_mi"),
        ("Perimeter", "district_perimeter_mi", "miles"),
        ("Competitiveness", "competitiveness_label", "text"),
        ("CVAP status", "majority_minority_status", "text"),
    ]

    for alias, source_col, kind in standard_specs:
        if source_col not in gdf.columns:
            continue
        if kind == "number":
            values = formatted_gdf[source_col].map(format_number)
        elif kind == "percent":
            values = formatted_gdf[source_col].map(format_percent)
        elif kind == "score":
            values = formatted_gdf[source_col].map(format_decimal_score)
        elif kind == "sq_mi":
            values = formatted_gdf[source_col].map(format_square_miles)
        elif kind == "miles":
            values = formatted_gdf[source_col].map(format_miles)
        elif kind == "text":
            values = formatted_gdf[source_col].map(
                lambda value: "Not available" if pd.isna(value) else str(value)
            )
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
        .legend svg {
            max-width: 390px !important;
            overflow: visible !important;
        }
        .legend .caption {
            display: none !important;
        }
        .legend-title-under {
            color: #f8fafc !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
            margin: -2px 0 2px 25px !important;
            max-width: 340px !important;
            white-space: nowrap !important;
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

    if (
        layer_column == "dem_share"
        and {"dem_share", "rep_share"}.issubset(map_gdf.columns)
    ):
        use_party_vote_counts = (
            state_name == "California"
            and {"dem_votes", "rep_votes"}.issubset(map_gdf.columns)
            and has_usable_numeric_data(map_gdf, "dem_votes")
            and has_usable_numeric_data(map_gdf, "rep_votes")
        )
        party_vote_bounds = party_vote_count_bounds(map_gdf) if use_party_vote_counts else None

        def style_function(feature: dict[str, Any]) -> dict[str, Any]:
            if use_party_vote_counts and party_vote_bounds is not None:
                fill_color = party_vote_count_color(feature["properties"], party_vote_bounds)
            else:
                fill_color = party_winner_strength_color(feature["properties"])
            return {
                "fillColor": fill_color,
                "color": "#f8fafc",
                "weight": 0.65,
                "fillOpacity": 0.9,
                "opacity": 0.85,
            }

        legend_text = "Party vote volume" if use_party_vote_counts else "Party winner strength"
        map_object.get_root().html.add_child(
            folium.Element(party_winner_legend_html(legend_text))
        )
    elif layer_column and layer_column in map_gdf.columns:
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

        if layer_column in PERCENT_COLUMNS:
            map_object.get_root().html.add_child(
                folium.Element(fixed_decimal_legend_html(legend_title(layer_column), color_map))
            )
        else:
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
                "winner_margin",
                "turnout_rate",
                "black_cvap_pct",
                "latino_cvap_pct",
                "asian_cvap_pct",
                "white_cvap_pct",
                "minority_cvap_pct",
                "young_voter_turnout",
                "compactness_polsby_popper",
                "district_area_sq_mi",
                "district_perimeter_mi",
                "competitiveness_label",
                "majority_minority_status",
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


def district_id_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Find the best district identifier for scorecard display."""
    detected = detect_columns(gdf.drop(columns="geometry", errors="ignore"))
    district_col = detected.get("district_id")
    return district_col if district_col in gdf.columns else None


def build_district_scorecard(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Create district-level profile, ranking, and label fields."""
    attrs = gdf.drop(columns="geometry", errors="ignore").copy()
    if attrs.empty:
        return pd.DataFrame()

    id_column = district_id_column(gdf)
    district_values = attrs[id_column] if id_column in attrs.columns else pd.Series(attrs.index, index=attrs.index)

    scorecard = pd.DataFrame(index=attrs.index)
    scorecard["district"] = district_values.map(format_district_label)

    numeric_columns = [
        "dem_votes",
        "rep_votes",
        "total_dr_votes",
        "dem_share",
        "rep_share",
        "turnout_rate",
        "minority_cvap_pct",
        "young_voter_turnout",
    ]
    for column in numeric_columns:
        if column in attrs.columns:
            scorecard[column] = safe_numeric(attrs[column])
        else:
            scorecard[column] = np.nan

    dem_share = scorecard["dem_share"]
    rep_share = scorecard["rep_share"]
    valid_vote_share = dem_share.notna() & rep_share.notna()
    scorecard["winner"] = "Not available"
    scorecard.loc[valid_vote_share & (dem_share >= rep_share), "winner"] = "Democratic"
    scorecard.loc[valid_vote_share & (rep_share > dem_share), "winner"] = "Republican"
    scorecard["vote_margin"] = (dem_share - rep_share).abs()
    scorecard["democratic_margin"] = dem_share - rep_share
    scorecard["republican_margin"] = rep_share - dem_share
    scorecard["competitiveness_score"] = ((1 - scorecard["vote_margin"]).clip(0, 1) * 100).round(1)

    total_districts = len(scorecard)
    scorecard["competitiveness_rank"] = scorecard["vote_margin"].rank(method="min", ascending=True)
    scorecard["minority_cvap_rank"] = scorecard["minority_cvap_pct"].rank(method="min", ascending=False)
    scorecard["young_turnout_rank"] = scorecard["young_voter_turnout"].rank(method="min", ascending=False)
    scorecard["turnout_rank"] = scorecard["turnout_rate"].rank(method="min", ascending=False)
    scorecard["winner_margin_rank"] = scorecard["vote_margin"].rank(method="min", ascending=False)
    scorecard["total_districts"] = total_districts
    return scorecard


def competitiveness_label(value: Any) -> str:
    margin = safe_numeric(pd.Series([value])).iloc[0]
    if pd.isna(margin):
        return "Competitiveness unknown"
    if margin <= 0.05:
        return "Toss-up district"
    if margin <= 0.10:
        return "Competitive district"
    if margin <= 0.20:
        return "Leaning district"
    return "Safe district"


def format_party_margin(row: pd.Series) -> str:
    dem_margin = safe_numeric(pd.Series([row.get("democratic_margin")])).iloc[0]
    if pd.isna(dem_margin):
        return "Not available"
    party = "D" if dem_margin >= 0 else "R"
    return f"{party} +{abs(dem_margin) * 100:.1f} pp"


def district_scorecard_tags(row: pd.Series, scorecard: pd.DataFrame) -> list[str]:
    """Create short qualitative labels for the selected district."""
    tags = [competitiveness_label(row.get("vote_margin"))]

    minority_cvap = safe_numeric(pd.Series([row.get("minority_cvap_pct")])).iloc[0]
    if pd.notna(minority_cvap):
        if minority_cvap >= 0.50:
            tags.append("Majority-minority CVAP")
        elif minority_cvap >= 0.40:
            tags.append("Near majority-minority CVAP")

    turnout = safe_numeric(pd.Series([row.get("turnout_rate")])).iloc[0]
    turnout_median = safe_numeric(scorecard.get("turnout_rate", pd.Series(dtype="float64"))).median()
    if pd.notna(turnout) and pd.notna(turnout_median):
        if turnout >= turnout_median + 0.05:
            tags.append("High overall turnout")
        elif turnout <= turnout_median - 0.05:
            tags.append("Low overall turnout")

    young_turnout = safe_numeric(pd.Series([row.get("young_voter_turnout")])).iloc[0]
    young_median = safe_numeric(scorecard.get("young_voter_turnout", pd.Series(dtype="float64"))).median()
    if pd.notna(young_turnout) and pd.notna(young_median):
        if young_turnout >= young_median + 0.05:
            tags.append("High young turnout")
        elif young_turnout <= young_median - 0.05:
            tags.append("Low young turnout")

    return tags


def district_ranking_display(scorecard: pd.DataFrame, sort_column: str, ascending: bool) -> pd.DataFrame:
    """Return a formatted ranking table for Streamlit display."""
    ranked = scorecard.copy()
    ranked["_rank_value"] = safe_numeric(ranked[sort_column])
    ranked = ranked[ranked["_rank_value"].notna()].sort_values(
        ["_rank_value", "district"],
        ascending=[ascending, True],
        key=lambda series: series.map(natural_sort_key) if series.name == "district" else series,
    )
    ranked = ranked.reset_index(drop=True)
    ranked["Rank"] = np.arange(1, len(ranked) + 1)

    return pd.DataFrame(
        {
            "Rank": ranked["Rank"],
            "District": ranked["district"],
            "Winner": ranked["winner"],
            "D/R margin": ranked.apply(format_party_margin, axis=1),
            "Competitiveness": ranked["competitiveness_score"].map(format_competitiveness_score),
            "Minority CVAP": ranked["minority_cvap_pct"].map(format_percent),
            "Overall turnout": ranked["turnout_rate"].map(format_percent),
            "Young turnout": ranked["young_voter_turnout"].map(format_percent),
            "D + R votes": ranked["total_dr_votes"].map(format_number),
        }
    )


def display_district_scorecard(gdf: gpd.GeoDataFrame, state_name: str) -> None:
    """Display a district selector, profile metrics, and ranking model."""
    scorecard = build_district_scorecard(gdf)
    if scorecard.empty:
        st.info("District scorecard could not be created for this data.")
        return

    st.subheader("District Scorecard")
    district_options = sorted(scorecard["district"].dropna().unique(), key=natural_sort_key)
    selected_district = st.selectbox(
        "Select district",
        district_options,
        key=f"{state_name}_district_scorecard_select",
    )
    selected_row = scorecard.loc[scorecard["district"] == selected_district].iloc[0]

    profile_columns = st.columns(5)
    profile_columns[0].metric("Vote Winner", selected_row.get("winner", "Not available"))
    profile_columns[1].metric("D/R Margin", format_party_margin(selected_row))
    profile_columns[2].metric(
        "Competitive Rank",
        format_rank(selected_row.get("competitiveness_rank"), len(scorecard)),
    )
    profile_columns[3].metric("Minority CVAP", format_percent(selected_row.get("minority_cvap_pct")))
    profile_columns[4].metric("Young Turnout", format_percent(selected_row.get("young_voter_turnout")))

    st.caption("Scorecard tags: " + " | ".join(district_scorecard_tags(selected_row, scorecard)))

    rank_options = {
        "Competitiveness": ("vote_margin", True),
        "Minority CVAP share": ("minority_cvap_pct", False),
        "Young turnout": ("young_voter_turnout", False),
        "Overall turnout": ("turnout_rate", False),
        "Democratic margin": ("democratic_margin", False),
        "Republican margin": ("republican_margin", False),
        "Winner margin": ("vote_margin", False),
        "D + R votes": ("total_dr_votes", False),
    }
    ranking_choice = st.selectbox(
        "Rank districts by",
        list(rank_options.keys()),
        key=f"{state_name}_district_rank_dimension",
    )
    sort_column, ascending = rank_options[ranking_choice]
    ranking_table = district_ranking_display(scorecard, sort_column, ascending)
    if ranking_table.empty:
        st.info(f"{ranking_choice} could not be ranked because the needed data is unavailable.")
    else:
        table_height = min(560, 38 * (len(ranking_table) + 1))
        st.dataframe(ranking_table, hide_index=True, use_container_width=True, height=table_height)


def geojson_feature_label(feature: dict[str, Any], index: int) -> str:
    """Choose a useful display label for a proposed district GeoJSON feature."""
    properties = feature.get("properties") or {}
    for key in ["proposal_label", "district", "DISTRICT", "DISTRICT_I", "CD119", "BASENAME", "name", "NAME", "id", "ID"]:
        value = properties.get(key)
        if value not in [None, ""]:
            return format_district_label(value)
    feature_id = feature.get("id")
    if feature_id not in [None, ""]:
        return format_district_label(feature_id)
    return f"Proposed {index}"


def proposed_geometries_from_geojson(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return labeled proposed district geometries from GeoJSON."""
    proposed: list[dict[str, Any]] = []
    try:
        if data.get("type") == "FeatureCollection":
            for index, feature in enumerate(data.get("features", []), start=1):
                if not feature.get("geometry"):
                    continue
                geometry = shape(feature["geometry"])
                if geometry.is_empty:
                    continue
                proposed.append(
                    {
                        "label": geojson_feature_label(feature, index),
                        "geometry": geometry,
                        "properties": feature.get("properties") or {},
                    }
                )
            return proposed
        if data.get("type") == "Feature":
            if data.get("geometry"):
                geometry = shape(data["geometry"])
                if not geometry.is_empty:
                    proposed.append(
                        {
                            "label": geojson_feature_label(data, 1),
                            "geometry": geometry,
                            "properties": data.get("properties") or {},
                        }
                    )
            return proposed
        if data.get("type"):
            geometry = shape(data)
            if not geometry.is_empty:
                proposed.append({"label": "Proposed 1", "geometry": geometry, "properties": {}})
    except (KeyError, TypeError, ValueError):
        return []
    return proposed


def geometry_from_geojson(data: dict[str, Any]) -> Any | None:
    """Return a single shapely geometry from a GeoJSON object."""
    geometries = [item["geometry"] for item in proposed_geometries_from_geojson(data)]
    if not geometries:
        return None
    return unary_union(geometries)


def latest_drawn_geometry(draw_output: dict[str, Any] | None, session_key: str) -> Any | None:
    """Get the latest drawn geometry from streamlit-folium, preserving it across reruns."""
    if not draw_output:
        return geometry_from_geojson(st.session_state.get(session_key, {}))

    drawings = draw_output.get("all_drawings") or []
    if drawings:
        st.session_state[session_key] = drawings[-1]
        return geometry_from_geojson(drawings[-1])

    last_drawing = draw_output.get("last_active_drawing")
    if last_drawing:
        st.session_state[session_key] = last_drawing
        return geometry_from_geojson(last_drawing)

    return geometry_from_geojson(st.session_state.get(session_key, {}))


def drawn_plan_geometries(draw_output: dict[str, Any] | None, session_key: str) -> list[dict[str, Any]]:
    """Get all drawn proposed districts from streamlit-folium."""
    if draw_output and draw_output.get("all_drawings"):
        st.session_state[session_key] = draw_output["all_drawings"]

    drawings = st.session_state.get(session_key, [])
    proposed = []
    for index, drawing in enumerate(drawings, start=1):
        for item in proposed_geometries_from_geojson(drawing):
            item["label"] = f"Drawn {index}"
            proposed.append(item)
    return proposed


def uploaded_geojson_geometry(uploaded_file: Any) -> Any | None:
    """Read a Streamlit uploaded GeoJSON file into one shapely geometry."""
    if uploaded_file is None:
        return None
    try:
        data = json.loads(uploaded_file.getvalue().decode("utf-8-sig"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return geometry_from_geojson(data)


def uploaded_geojson_plan(uploaded_file: Any) -> list[dict[str, Any]]:
    """Read a Streamlit uploaded GeoJSON file into labeled proposed districts."""
    if uploaded_file is None:
        return []
    try:
        data = json.loads(uploaded_file.getvalue().decode("utf-8-sig"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return proposed_geometries_from_geojson(data)


def projected_area_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project data to a continental equal-area CRS for overlap calculations."""
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    return gdf.to_crs(5070)


def majority_minority_status(value: Any) -> str:
    minority_share = safe_numeric(pd.Series([value])).iloc[0]
    if pd.isna(minority_share):
        return "CVAP status unknown"
    if minority_share >= 0.50:
        return "Majority-minority CVAP"
    if minority_share >= 0.40:
        return "Near majority-minority CVAP"
    return "Majority white CVAP"


def add_redistricting_metrics(gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
    """Add geometry, competitiveness, and representation metrics to districts."""
    if gdf is None or gdf.empty or "geometry" not in gdf.columns:
        return gdf

    enriched = gdf.copy()
    try:
        projected = projected_area_gdf(enriched)
        area_sq_meters = projected.geometry.area.replace(0, np.nan)
        perimeter_meters = projected.geometry.length.replace(0, np.nan)
    except Exception:
        return enriched

    enriched["district_area_sq_mi"] = area_sq_meters / 2_589_988.110336
    enriched["district_perimeter_mi"] = perimeter_meters / 1_609.344
    enriched["compactness_polsby_popper"] = (
        4 * np.pi * area_sq_meters / (perimeter_meters ** 2)
    ).clip(lower=0, upper=1)

    if {"dem_share", "rep_share"}.issubset(enriched.columns):
        dem_share = safe_numeric(enriched["dem_share"])
        rep_share = safe_numeric(enriched["rep_share"])
        enriched["winner_margin"] = (dem_share - rep_share).abs()
        enriched["competitiveness_label"] = enriched["winner_margin"].map(competitiveness_label)

    if "minority_cvap_pct" in enriched.columns:
        minority_share = safe_numeric(enriched["minority_cvap_pct"])
        enriched["majority_minority_flag"] = (minority_share >= 0.50).astype(float)
        enriched["majority_minority_status"] = enriched["minority_cvap_pct"].map(majority_minority_status)

    return enriched


def estimate_hypothetical_district(
    district_gdf: gpd.GeoDataFrame,
    proposed_geometry: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate district metrics from area-weighted overlap with current districts."""
    if district_gdf is None or district_gdf.empty or proposed_geometry is None or proposed_geometry.is_empty:
        return {}, pd.DataFrame()

    proposed_gdf = gpd.GeoDataFrame({"geometry": [proposed_geometry]}, crs=4326)
    districts_equal_area = projected_area_gdf(district_gdf)
    proposed_equal_area = projected_area_gdf(proposed_gdf).geometry.iloc[0]

    if proposed_equal_area.is_empty:
        return {}, pd.DataFrame()
    if not proposed_equal_area.is_valid:
        proposed_equal_area = proposed_equal_area.buffer(0)
    proposed_area = proposed_equal_area.area
    if proposed_area <= 0:
        return {}, pd.DataFrame()

    district_col = district_id_column(district_gdf)
    overlap_rows = []
    for index, district_row in districts_equal_area.iterrows():
        geometry = district_row.geometry
        if geometry is None or geometry.is_empty or not geometry.intersects(proposed_equal_area):
            continue
        overlap = geometry.intersection(proposed_equal_area)
        if overlap.is_empty:
            continue
        overlap_area = overlap.area
        if overlap_area <= 0:
            continue
        share_of_proposed = overlap_area / proposed_area
        if share_of_proposed < MIN_HYPOTHETICAL_OVERLAP_SHARE:
            continue
        current_area = geometry.area
        original_row = district_gdf.loc[index]
        district_value = original_row[district_col] if district_col in district_gdf.columns else index
        overlap_rows.append(
            {
                "district": format_district_label(district_value),
                "overlap_area": overlap_area,
                "share_of_proposed": share_of_proposed,
                "share_of_current_district": overlap_area / current_area if current_area else np.nan,
                "source_index": index,
            }
        )

    overlaps = pd.DataFrame(overlap_rows)
    if overlaps.empty:
        return {"coverage_pct": 0, "intersecting_districts": 0}, overlaps

    metrics: dict[str, Any] = {
        "coverage_pct": min(float(overlaps["share_of_proposed"].sum()), 1.0),
        "intersecting_districts": len(overlaps),
    }

    count_columns = ["dem_votes", "rep_votes", "total_votes", "registered_voters"]
    for column in count_columns:
        if column not in district_gdf.columns:
            continue
        estimated = 0.0
        has_values = False
        for _, overlap_row in overlaps.iterrows():
            value = safe_numeric(pd.Series([district_gdf.loc[overlap_row["source_index"], column]])).iloc[0]
            if pd.notna(value):
                estimated += float(value) * float(overlap_row["share_of_current_district"])
                has_values = True
        if has_values:
            metrics[column] = estimated

    if "dem_votes" in metrics and "rep_votes" in metrics:
        metrics["total_dr_votes"] = metrics["dem_votes"] + metrics["rep_votes"]
        if metrics["total_dr_votes"]:
            metrics["dem_share"] = metrics["dem_votes"] / metrics["total_dr_votes"]
            metrics["rep_share"] = metrics["rep_votes"] / metrics["total_dr_votes"]

    if "total_votes" in metrics and "registered_voters" in metrics and metrics["registered_voters"]:
        metrics["turnout_rate"] = metrics["total_votes"] / metrics["registered_voters"]

    weighted_columns = [
        "black_cvap_pct",
        "latino_cvap_pct",
        "asian_cvap_pct",
        "white_cvap_pct",
        "minority_cvap_pct",
        "turnout_rate",
        "young_voter_turnout",
    ]
    total_overlap_area = overlaps["overlap_area"].sum()
    for column in weighted_columns:
        if column not in district_gdf.columns or column in metrics:
            continue
        weighted_total = 0.0
        weight_sum = 0.0
        for _, overlap_row in overlaps.iterrows():
            value = safe_numeric(pd.Series([district_gdf.loc[overlap_row["source_index"], column]])).iloc[0]
            if pd.notna(value):
                weight = float(overlap_row["overlap_area"]) / total_overlap_area
                weighted_total += float(value) * weight
                weight_sum += weight
        if weight_sum:
            metrics[column] = weighted_total / weight_sum

    if "dem_share" in metrics and "rep_share" in metrics:
        winner = "Democratic" if metrics["dem_share"] >= metrics["rep_share"] else "Republican"
        metrics["winner"] = winner
        metrics["vote_margin"] = abs(metrics["dem_share"] - metrics["rep_share"])

    display_columns = [
        "dem_share",
        "minority_cvap_pct",
        "turnout_rate",
        "young_voter_turnout",
    ]
    for column in display_columns:
        if column in district_gdf.columns:
            overlaps[column] = overlaps["source_index"].map(lambda idx: district_gdf.loc[idx, column])

    overlaps = overlaps.drop(columns=["source_index"])
    return metrics, overlaps


def create_hypothetical_draw_map(
    district_gdf: gpd.GeoDataFrame,
    state_name: str,
    simplify: bool,
) -> folium.Map:
    """Build a map with current districts and drawing tools."""
    config = STATE_CONFIG[state_name]
    map_object = folium.Map(
        location=config["center"],
        zoom_start=config["zoom"],
        tiles="cartodbdark_matter",
        control_scale=True,
    )
    overlay = prepare_map_gdf(district_gdf, None, simplify, [])
    if not overlay.empty:
        folium.GeoJson(
            data=geojson_dict(overlay),
            name="Current Congressional Districts",
            interactive=False,
            style_function=lambda _: {
                "fillOpacity": 0.08,
                "fillColor": "#38bdf8",
                "color": "#f8fafc",
                "weight": 1.4,
                "opacity": 0.85,
            },
        ).add_to(map_object)

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": {
                "allowIntersection": False,
                "showArea": True,
                "shapeOptions": {"color": "#facc15", "fillColor": "#facc15", "fillOpacity": 0.22},
            },
            "rectangle": {
                "showArea": True,
                "shapeOptions": {"color": "#facc15", "fillColor": "#facc15", "fillOpacity": 0.22},
            },
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(map_object)
    folium.LayerControl(collapsed=True).add_to(map_object)
    return map_object


def proposed_geometry_metrics(geometry: Any) -> dict[str, Any]:
    """Calculate area, perimeter, and compactness for one proposed district geometry."""
    if geometry is None or geometry.is_empty:
        return {}
    proposed_gdf = gpd.GeoDataFrame({"geometry": [geometry]}, crs=4326)
    try:
        projected = projected_area_gdf(proposed_gdf)
        projected_geometry = projected.geometry.iloc[0]
        if not projected_geometry.is_valid:
            projected_geometry = projected_geometry.buffer(0)
        area_sq_meters = projected_geometry.area
        perimeter_meters = projected_geometry.length
    except Exception:
        return {}
    if area_sq_meters <= 0 or perimeter_meters <= 0:
        return {}
    return {
        "district_area_sq_mi": area_sq_meters / 2_589_988.110336,
        "district_perimeter_mi": perimeter_meters / 1_609.344,
        "compactness_polsby_popper": max(
            0.0,
            min(1.0, 4 * np.pi * area_sq_meters / (perimeter_meters ** 2)),
        ),
    }


def proposed_plan_geojson(proposed_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Serialize proposed plan geometries for Folium display."""
    if not proposed_items:
        return {"type": "FeatureCollection", "features": []}
    proposed_gdf = gpd.GeoDataFrame(
        {
            "proposal_label": [item["label"] for item in proposed_items],
            "geometry": [item["geometry"] for item in proposed_items],
        },
        crs=4326,
    )
    return geojson_dict(proposed_gdf)


def add_proposed_plan_overlay(map_object: folium.Map, proposed_items: list[dict[str, Any]]) -> None:
    """Draw proposed districts on top of a plan map."""
    if not proposed_items:
        return
    folium.GeoJson(
        data=proposed_plan_geojson(proposed_items),
        name="Proposed Plan",
        tooltip=folium.GeoJsonTooltip(fields=["proposal_label"], aliases=["Proposed district"]),
        style_function=lambda _: {
            "fillColor": "#facc15",
            "color": "#facc15",
            "weight": 2.4,
            "fillOpacity": 0.22,
            "opacity": 0.95,
        },
        highlight_function=lambda _: {"weight": 3.4, "color": "#fef08a", "fillOpacity": 0.32},
    ).add_to(map_object)


def create_proposed_plan_map(
    district_gdf: gpd.GeoDataFrame,
    state_name: str,
    simplify: bool,
    proposed_items: list[dict[str, Any]] | None = None,
) -> folium.Map:
    """Build a map for drawing or viewing a proposed district plan."""
    map_object = create_hypothetical_draw_map(district_gdf, state_name, simplify)
    add_proposed_plan_overlay(map_object, proposed_items or [])
    return map_object


def district_label_map(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """Map display district labels to GeoDataFrame indexes."""
    id_column = district_id_column(gdf)
    labels: dict[str, Any] = {}
    attrs = gdf.drop(columns="geometry", errors="ignore")
    for index, row in attrs.iterrows():
        value = row[id_column] if id_column in row.index else index
        label = format_district_label(value)
        labels[label] = index
    return dict(sorted(labels.items(), key=lambda item: natural_sort_key(item[0])))


def current_plan_gdf(gdf: gpd.GeoDataFrame, simplify: bool = False) -> gpd.GeoDataFrame:
    """Convert current district geometries into a labeled GeoDataFrame for proposed-plan tools."""
    labels = district_label_map(gdf)
    rows = []
    for label, index in labels.items():
        geometry = gdf.loc[index].geometry
        if geometry is None or geometry.is_empty:
            continue
        rows.append({"proposal_label": label, "geometry": geometry})

    plan_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs if gdf.crs is not None else 4326)
    return prepare_map_gdf(plan_gdf, None, simplify, ["proposal_label"])


def current_plan_items(gdf: gpd.GeoDataFrame, simplify: bool = False) -> list[dict[str, Any]]:
    """Convert current district geometries into proposed-plan item objects."""
    plan_gdf = current_plan_gdf(gdf, simplify)
    items = []
    for _, row in plan_gdf.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        label = format_district_label(row.get("proposal_label", len(items) + 1))
        items.append({"label": label, "geometry": geometry, "properties": {"source": "current"}})
    return items


def safe_geometry(geometry: Any) -> Any:
    """Repair a geometry when possible without failing the app."""
    if geometry is None or geometry.is_empty:
        return geometry
    try:
        if not geometry.is_valid:
            return geometry.buffer(0)
    except Exception:
        return geometry
    return geometry


def geometry_parts(geometry: Any) -> list[Any]:
    """Return polygon-like parts from a Shapely geometry."""
    geometry = safe_geometry(geometry)
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return list(getattr(geometry, "geoms", [geometry]))
    return [
        part
        for part in getattr(geometry, "geoms", [])
        if part.geom_type in {"Polygon", "MultiPolygon"} and not part.is_empty
    ]


def projected_geometry_area(geometry: Any) -> float:
    """Measure one geometry in square meters using the app's equal-area projection."""
    if geometry is None or geometry.is_empty:
        return 0.0
    try:
        projected = projected_area_gdf(gpd.GeoDataFrame({"geometry": [safe_geometry(geometry)]}, crs=4326))
        projected_geometry = safe_geometry(projected.geometry.iloc[0])
        return float(projected_geometry.area) if projected_geometry is not None else 0.0
    except Exception:
        return 0.0


def geometry_change_share(reference_geometry: Any, edited_geometry: Any) -> float:
    """Return how much a geometry changed as a share of its original area."""
    reference_geometry = safe_geometry(reference_geometry)
    edited_geometry = safe_geometry(edited_geometry)
    if reference_geometry is None or edited_geometry is None:
        return 0.0
    reference_area = projected_geometry_area(reference_geometry)
    if reference_area <= 0:
        return 0.0
    try:
        changed_area = projected_geometry_area(reference_geometry.symmetric_difference(edited_geometry))
    except Exception:
        return 0.0
    return changed_area / reference_area


def align_edited_items_to_current_labels(
    edited_items: list[dict[str, Any]],
    baseline_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep edited browser features attached to the district labels loaded into the edit map."""
    baseline_labels = [item["label"] for item in baseline_items]
    baseline_label_set = set(baseline_labels)
    aligned = []
    used_labels: set[str] = set()
    for index, item in enumerate(edited_items):
        label = str(item.get("label") or "")
        if label not in baseline_label_set and index < len(baseline_labels):
            label = baseline_labels[index]
        if label in used_labels:
            continue
        used_labels.add(label)
        aligned.append(
            {
                "label": label,
                "geometry": safe_geometry(item.get("geometry")),
                "properties": item.get("properties") or {},
            }
        )
    return aligned


def best_recipient_for_lost_piece(
    piece: Any,
    source_label: str,
    result_by_label: dict[str, Any],
    current_by_label: dict[str, Any],
) -> str | None:
    """Choose the neighboring district that should receive an uncovered piece."""
    candidates = []
    for label, geometry in result_by_label.items():
        if label == source_label or geometry is None or geometry.is_empty:
            continue
        try:
            current_geometry = current_by_label.get(label)
            shared_length = piece.boundary.intersection(geometry.boundary).length
            if current_geometry is not None and not current_geometry.is_empty:
                shared_length = max(shared_length, piece.boundary.intersection(current_geometry.boundary).length)
            distance = piece.distance(geometry)
        except Exception:
            shared_length = 0.0
            distance = float("inf")
        candidates.append((shared_length, -distance, label))

    if not candidates:
        return None
    return max(candidates, key=lambda value: (value[0], value[1], natural_sort_key(value[2])))[2]


def apply_topology_preserving_edits(
    current_items: list[dict[str, Any]],
    baseline_items: list[dict[str, Any]],
    edited_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Apply browser-edited boundary changes to the current plan while keeping one clean plan.

    The browser editor moves one polygon at a time. This repair step infers gained and
    lost area from the edited district and gives/takes that area from neighboring districts.
    """
    current_by_label = {item["label"]: safe_geometry(item["geometry"]) for item in current_items}
    baseline_by_label = {item["label"]: safe_geometry(item["geometry"]) for item in baseline_items}
    edited_by_label = {item["label"]: safe_geometry(item["geometry"]) for item in edited_items}

    try:
        current_union = safe_geometry(unary_union([geometry for geometry in current_by_label.values() if geometry is not None]))
    except Exception:
        current_union = None

    result_by_label = dict(current_by_label)
    changed_labels = []
    gained_by_label: dict[str, Any] = {}
    lost_by_label: dict[str, Any] = {}

    for label, edited_geometry in edited_by_label.items():
        baseline_geometry = baseline_by_label.get(label)
        if label not in current_by_label or baseline_geometry is None or edited_geometry is None:
            continue
        if geometry_change_share(baseline_geometry, edited_geometry) < 0.00001:
            continue

        changed_labels.append(label)
        try:
            gained = safe_geometry(edited_geometry.difference(baseline_geometry))
            lost = safe_geometry(baseline_geometry.difference(edited_geometry))
            if current_union is not None and gained is not None and not gained.is_empty:
                gained = safe_geometry(gained.intersection(current_union))
        except Exception:
            continue
        gained_by_label[label] = gained
        lost_by_label[label] = lost

    if not changed_labels:
        return current_items, []

    for label in changed_labels:
        updated_geometry = result_by_label[label]
        lost = lost_by_label.get(label)
        gained = gained_by_label.get(label)
        try:
            if lost is not None and not lost.is_empty:
                updated_geometry = safe_geometry(updated_geometry.difference(lost))
            if gained is not None and not gained.is_empty:
                updated_geometry = safe_geometry(updated_geometry.union(gained))
        except Exception:
            continue
        result_by_label[label] = updated_geometry

    for label, gained in gained_by_label.items():
        if gained is None or gained.is_empty:
            continue
        for other_label, geometry in list(result_by_label.items()):
            if other_label == label or geometry is None or geometry.is_empty:
                continue
            try:
                if geometry.intersects(gained):
                    result_by_label[other_label] = safe_geometry(geometry.difference(gained))
            except Exception:
                continue

    for source_label, lost in lost_by_label.items():
        if lost is None or lost.is_empty:
            continue
        try:
            covered = safe_geometry(unary_union([geometry for geometry in result_by_label.values() if geometry is not None]))
            uncovered = safe_geometry(lost.difference(covered)) if covered is not None else lost
        except Exception:
            uncovered = lost
        for piece in geometry_parts(uncovered):
            recipient_label = best_recipient_for_lost_piece(piece, source_label, result_by_label, current_by_label)
            if recipient_label is None:
                recipient_label = source_label
            try:
                result_by_label[recipient_label] = safe_geometry(result_by_label[recipient_label].union(piece))
            except Exception:
                continue

    items = []
    for item in current_items:
        geometry = result_by_label.get(item["label"])
        if current_union is not None and geometry is not None and not geometry.is_empty:
            try:
                geometry = safe_geometry(geometry.intersection(current_union))
            except Exception:
                pass
        if geometry is None or geometry.is_empty:
            continue
        items.append({"label": item["label"], "geometry": geometry, "properties": {"source": "boundary_edit"}})
    return items, sorted(changed_labels, key=natural_sort_key)


def edited_current_plan_items(
    draw_output: dict[str, Any] | None,
    session_key: str,
    current_items: list[dict[str, Any]],
    baseline_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read edited current-plan polygons from streamlit-folium and repair them into one plan."""
    if draw_output and draw_output.get("all_drawings"):
        st.session_state[session_key] = draw_output["all_drawings"]

    drawings = st.session_state.get(session_key, [])
    if not drawings:
        return current_items, []

    edited_items = proposed_geometries_from_geojson({"type": "FeatureCollection", "features": drawings})
    aligned_items = align_edited_items_to_current_labels(edited_items, baseline_items)
    if not aligned_items:
        return current_items, []
    return apply_topology_preserving_edits(current_items, baseline_items, aligned_items)


def create_editable_current_plan_map(
    district_gdf: gpd.GeoDataFrame,
    state_name: str,
    simplify: bool,
) -> folium.Map:
    """Build a map where current district polygons are editable instead of manually transferred."""
    config = STATE_CONFIG[state_name]
    map_object = folium.Map(
        location=config["center"],
        zoom_start=config["zoom"],
        tiles="cartodbdark_matter",
        control_scale=True,
    )
    editable_gdf = current_plan_gdf(district_gdf, simplify)
    editable_layer = folium.GeoJson(
        data=geojson_dict(editable_gdf),
        name="Editable Current Districts",
        tooltip=folium.GeoJsonTooltip(fields=["proposal_label"], aliases=["District"]),
        style_function=lambda _: {
            "fillOpacity": 0.14,
            "fillColor": "#38bdf8",
            "color": "#f8fafc",
            "weight": 1.7,
            "opacity": 0.95,
        },
        highlight_function=lambda _: {"weight": 2.8, "color": "#facc15", "fillOpacity": 0.24},
    )
    editable_layer.add_to(map_object)
    Draw(
        export=False,
        feature_group=editable_layer,
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": False,
            "rectangle": False,
        },
        edit_options={"edit": True, "remove": False},
    ).add_to(map_object)
    folium.LayerControl(collapsed=True).add_to(map_object)
    return map_object


def create_proposed_plan_preview_map(
    district_gdf: gpd.GeoDataFrame,
    state_name: str,
    simplify: bool,
    proposed_items: list[dict[str, Any]],
) -> folium.Map:
    """Show a static preview of the repaired proposed plan."""
    config = STATE_CONFIG[state_name]
    map_object = folium.Map(
        location=config["center"],
        zoom_start=config["zoom"],
        tiles="cartodbdark_matter",
        control_scale=True,
    )
    current_overlay = prepare_map_gdf(district_gdf, None, simplify, [])
    if not current_overlay.empty:
        folium.GeoJson(
            data=geojson_dict(current_overlay),
            name="Current District Boundaries",
            interactive=False,
            style_function=lambda _: {
                "fillOpacity": 0.02,
                "fillColor": "#94a3b8",
                "color": "#94a3b8",
                "weight": 1.0,
                "opacity": 0.65,
            },
        ).add_to(map_object)
    add_proposed_plan_overlay(map_object, proposed_items)
    folium.LayerControl(collapsed=True).add_to(map_object)
    return map_object


def format_overlap_table(overlaps: pd.DataFrame) -> pd.DataFrame:
    """Format current-district overlap rows for display."""
    if overlaps.empty:
        return pd.DataFrame()
    table = overlaps.copy()
    return pd.DataFrame(
        {
            "Current district": table["district"],
            "Share of proposed": table["share_of_proposed"].map(format_percent),
            "Share of current district": table["share_of_current_district"].map(format_percent),
            "Current D share": table.get("dem_share", pd.Series(np.nan, index=table.index)).map(format_percent),
            "Current minority CVAP": table.get("minority_cvap_pct", pd.Series(np.nan, index=table.index)).map(format_percent),
            "Current turnout": table.get("turnout_rate", pd.Series(np.nan, index=table.index)).map(format_percent),
            "Current young turnout": table.get("young_voter_turnout", pd.Series(np.nan, index=table.index)).map(format_percent),
        }
    )


def display_hypothetical_metrics(metrics: dict[str, Any], overlaps: pd.DataFrame) -> None:
    """Show estimated metrics for a proposed district."""
    winner = metrics.get("winner", "Not available")
    profile_columns = st.columns(5)
    profile_columns[0].metric("Estimated Winner", winner)
    profile_columns[1].metric("D/R Margin", format_percentage_points(metrics.get("vote_margin")))
    profile_columns[2].metric("Democratic Share", format_percent(metrics.get("dem_share")))
    profile_columns[3].metric("Minority CVAP", format_percent(metrics.get("minority_cvap_pct")))
    profile_columns[4].metric("Coverage", format_percent(metrics.get("coverage_pct")))

    detail_columns = st.columns(5)
    detail_columns[0].metric("D + R Votes", format_number(metrics.get("total_dr_votes")))
    detail_columns[1].metric("Overall Turnout", format_percent(metrics.get("turnout_rate")))
    detail_columns[2].metric("Young Turnout", format_percent(metrics.get("young_voter_turnout")))
    detail_columns[3].metric("Current Districts", format_number(metrics.get("intersecting_districts")))
    detail_columns[4].metric("Registered Voters", format_number(metrics.get("registered_voters")))

    if safe_numeric(pd.Series([metrics.get("coverage_pct")])).iloc[0] < 0.98:
        st.warning("The proposed shape extends outside the current district layer, so the estimate uses only covered area.")

    overlap_table = format_overlap_table(overlaps)
    if not overlap_table.empty:
        st.dataframe(overlap_table, hide_index=True, use_container_width=True)


def display_hypothetical_district_tool(
    district_gdf: gpd.GeoDataFrame,
    state_name: str,
    simplify: bool,
) -> None:
    """Let users draw or upload a proposed district and estimate its metrics."""
    if district_gdf is None or district_gdf.empty:
        return

    st.subheader("Hypothetical District Tool")
    st.caption(
        "Draw a proposed district or upload GeoJSON. Estimates are approximate and area-weighted from current districts."
    )

    session_key = f"{state_name}_hypothetical_drawn_feature"
    if st.button("Clear proposed district", key=f"{state_name}_clear_hypothetical"):
        st.session_state.pop(session_key, None)

    draw_map = create_hypothetical_draw_map(district_gdf, state_name, simplify)
    draw_output = st_folium(
        draw_map,
        height=520,
        use_container_width=True,
        returned_objects=["last_active_drawing", "all_drawings"],
        key=f"{state_name}_hypothetical_draw_map",
    )
    drawn_geometry = latest_drawn_geometry(draw_output, session_key)

    uploaded_file = st.file_uploader(
        "Upload proposed district GeoJSON",
        type=["geojson", "json"],
        key=f"{state_name}_hypothetical_geojson_upload",
    )
    uploaded_geometry = uploaded_geojson_geometry(uploaded_file)
    if uploaded_file is not None and uploaded_geometry is None:
        st.warning("The uploaded file could not be read as GeoJSON geometry.")

    proposed_geometry = uploaded_geometry if uploaded_geometry is not None else drawn_geometry
    if proposed_geometry is None:
        st.info("Draw a polygon or rectangle on the map to estimate a new district.")
        return

    metrics, overlaps = estimate_hypothetical_district(district_gdf, proposed_geometry)
    if not metrics or overlaps.empty:
        st.warning("The proposed district does not overlap the current district layer.")
        return
    display_hypothetical_metrics(metrics, overlaps)


def estimate_proposed_plan(
    district_gdf: gpd.GeoDataFrame,
    proposed_items: list[dict[str, Any]],
) -> pd.DataFrame:
    """Estimate metrics for each proposed district in a proposed plan."""
    rows = []
    for index, item in enumerate(proposed_items, start=1):
        metrics, _overlaps = estimate_hypothetical_district(district_gdf, item.get("geometry"))
        if not metrics:
            continue
        geometry_metrics = proposed_geometry_metrics(item.get("geometry"))
        label = item.get("label") or f"Proposed {index}"
        dem_share = safe_numeric(pd.Series([metrics.get("dem_share")])).iloc[0]
        rep_share = safe_numeric(pd.Series([metrics.get("rep_share")])).iloc[0]
        if pd.isna(dem_share) or pd.isna(rep_share):
            winner = "Not available"
        else:
            winner = "Democratic" if dem_share >= rep_share else "Republican"
        rows.append(
            {
                "Proposed district": label,
                "Estimated winner": winner,
                "Winner margin": safe_numeric(pd.Series([metrics.get("vote_margin")])).iloc[0],
                "Democratic share": dem_share,
                "Republican share": rep_share,
                "D + R votes": safe_numeric(pd.Series([metrics.get("total_dr_votes")])).iloc[0],
                "Turnout": safe_numeric(pd.Series([metrics.get("turnout_rate")])).iloc[0],
                "Young turnout": safe_numeric(pd.Series([metrics.get("young_voter_turnout")])).iloc[0],
                "Minority CVAP": safe_numeric(pd.Series([metrics.get("minority_cvap_pct")])).iloc[0],
                "Black CVAP": safe_numeric(pd.Series([metrics.get("black_cvap_pct")])).iloc[0],
                "Latino CVAP": safe_numeric(pd.Series([metrics.get("latino_cvap_pct")])).iloc[0],
                "Coverage": safe_numeric(pd.Series([metrics.get("coverage_pct")])).iloc[0],
                "Current districts touched": safe_numeric(pd.Series([metrics.get("intersecting_districts")])).iloc[0],
                "Area": safe_numeric(pd.Series([geometry_metrics.get("district_area_sq_mi")])).iloc[0],
                "Perimeter": safe_numeric(pd.Series([geometry_metrics.get("district_perimeter_mi")])).iloc[0],
                "Compactness": safe_numeric(pd.Series([geometry_metrics.get("compactness_polsby_popper")])).iloc[0],
            }
        )
    return pd.DataFrame(rows)


def current_plan_outcome_metrics(district_gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """Create raw outcome metrics for the current district plan."""
    dem_share = safe_numeric(district_gdf.get("dem_share", pd.Series(dtype="float64")))
    rep_share = safe_numeric(district_gdf.get("rep_share", pd.Series(dtype="float64")))
    winner_margin = safe_numeric(district_gdf.get("winner_margin", (dem_share - rep_share).abs()))
    minority_cvap = safe_numeric(district_gdf.get("minority_cvap_pct", pd.Series(dtype="float64")))
    compactness = safe_numeric(district_gdf.get("compactness_polsby_popper", pd.Series(dtype="float64")))
    turnout = safe_numeric(district_gdf.get("turnout_rate", pd.Series(dtype="float64")))
    young_turnout = safe_numeric(district_gdf.get("young_voter_turnout", pd.Series(dtype="float64")))
    dem_votes = safe_numeric(district_gdf.get("dem_votes", pd.Series(dtype="float64"))).sum()
    rep_votes = safe_numeric(district_gdf.get("rep_votes", pd.Series(dtype="float64"))).sum()
    total_votes = dem_votes + rep_votes
    return {
        "district_count": len(district_gdf),
        "democratic_seats": ((dem_share >= rep_share) & dem_share.notna() & rep_share.notna()).sum(),
        "republican_seats": ((rep_share > dem_share) & dem_share.notna() & rep_share.notna()).sum(),
        "competitive_districts": (winner_margin <= 0.10).sum(),
        "majority_minority_districts": (minority_cvap >= 0.50).sum(),
        "average_compactness": compactness.mean(),
        "average_minority_cvap": minority_cvap.mean(),
        "average_turnout": turnout.mean(),
        "average_young_turnout": young_turnout.mean(),
        "democratic_share": dem_votes / total_votes if total_votes else np.nan,
    }


def proposed_plan_outcome_metrics(estimates: pd.DataFrame) -> dict[str, Any]:
    """Create raw outcome metrics for a proposed district plan."""
    if estimates.empty:
        return {}
    dem_votes = safe_numeric(estimates.get("Democratic share", pd.Series(dtype="float64"))) * safe_numeric(
        estimates.get("D + R votes", pd.Series(dtype="float64"))
    )
    rep_votes = safe_numeric(estimates.get("Republican share", pd.Series(dtype="float64"))) * safe_numeric(
        estimates.get("D + R votes", pd.Series(dtype="float64"))
    )
    total_votes = dem_votes.sum() + rep_votes.sum()
    return {
        "district_count": len(estimates),
        "democratic_seats": (estimates["Estimated winner"] == "Democratic").sum(),
        "republican_seats": (estimates["Estimated winner"] == "Republican").sum(),
        "competitive_districts": (safe_numeric(estimates["Winner margin"]) <= 0.10).sum(),
        "majority_minority_districts": (safe_numeric(estimates["Minority CVAP"]) >= 0.50).sum(),
        "average_compactness": safe_numeric(estimates["Compactness"]).mean(),
        "average_minority_cvap": safe_numeric(estimates["Minority CVAP"]).mean(),
        "average_turnout": safe_numeric(estimates["Turnout"]).mean(),
        "average_young_turnout": safe_numeric(estimates["Young turnout"]).mean(),
        "democratic_share": dem_votes.sum() / total_votes if total_votes else np.nan,
    }


def format_signed_number_delta(delta: Any) -> str:
    number = safe_numeric(pd.Series([delta])).iloc[0]
    if pd.isna(number):
        return "Not available"
    return f"{number:+,.0f}"


def format_signed_percent_delta(delta: Any) -> str:
    number = safe_numeric(pd.Series([delta])).iloc[0]
    if pd.isna(number):
        return "Not available"
    return f"{number * 100:+.1f} pp"


def format_signed_score_delta(delta: Any) -> str:
    number = safe_numeric(pd.Series([delta])).iloc[0]
    if pd.isna(number):
        return "Not available"
    return f"{number:+.3f}"


def proposed_plan_summary_table(district_gdf: gpd.GeoDataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    """Format a current-vs-proposed plan comparison table."""
    current = current_plan_outcome_metrics(district_gdf)
    proposed = proposed_plan_outcome_metrics(estimates)
    specs = [
        ("District count", "district_count", format_number, format_signed_number_delta),
        ("Estimated Democratic seats", "democratic_seats", format_number, format_signed_number_delta),
        ("Estimated Republican seats", "republican_seats", format_number, format_signed_number_delta),
        ("Competitive districts", "competitive_districts", format_number, format_signed_number_delta),
        ("Majority-minority CVAP districts", "majority_minority_districts", format_number, format_signed_number_delta),
        ("Average compactness", "average_compactness", format_decimal_score, format_signed_score_delta),
        ("Average minority CVAP", "average_minority_cvap", format_percent, format_signed_percent_delta),
        ("Average turnout", "average_turnout", format_percent, format_signed_percent_delta),
        ("Average young turnout", "average_young_turnout", format_percent, format_signed_percent_delta),
        ("Estimated Democratic vote share", "democratic_share", format_percent, format_signed_percent_delta),
    ]
    rows = []
    for label, key, formatter, delta_formatter in specs:
        current_value = current.get(key)
        proposed_value = proposed.get(key)
        rows.append(
            {
                "Metric": label,
                "Current plan": formatter(current_value),
                "Proposed plan": formatter(proposed_value),
                "Difference": delta_formatter(safe_numeric(pd.Series([proposed_value])).iloc[0] - safe_numeric(pd.Series([current_value])).iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def proposed_plan_display_table(estimates: pd.DataFrame) -> pd.DataFrame:
    """Format proposed district estimates for display."""
    if estimates.empty:
        return pd.DataFrame()
    table = estimates.copy()
    table = table.sort_values("Proposed district", key=lambda series: series.map(natural_sort_key))
    return pd.DataFrame(
        {
            "Proposed district": table["Proposed district"],
            "Estimated winner": table["Estimated winner"],
            "Winner margin": table["Winner margin"].map(format_percent),
            "D share": table["Democratic share"].map(format_percent),
            "Turnout": table["Turnout"].map(format_percent),
            "Young turnout": table["Young turnout"].map(format_percent),
            "Minority CVAP": table["Minority CVAP"].map(format_percent),
            "Black CVAP": table["Black CVAP"].map(format_percent),
            "Latino CVAP": table["Latino CVAP"].map(format_percent),
            "Compactness": table["Compactness"].map(format_decimal_score),
            "Area": table["Area"].map(format_square_miles),
            "Coverage": table["Coverage"].map(format_percent),
            "Current districts touched": table["Current districts touched"].map(format_number),
            "D + R votes": table["D + R votes"].map(format_number),
        }
    )


def plan_topology_diagnostics(
    district_gdf: gpd.GeoDataFrame,
    proposed_items: list[dict[str, Any]],
) -> dict[str, float]:
    """Measure coverage, gaps, and overlaps for a proposed plan."""
    if district_gdf is None or district_gdf.empty or not proposed_items:
        return {}
    try:
        current_projected = projected_area_gdf(district_gdf[["geometry"]].copy())
        proposed_gdf = gpd.GeoDataFrame(
            {"geometry": [safe_geometry(item["geometry"]) for item in proposed_items]},
            crs=4326,
        )
        proposed_projected = projected_area_gdf(proposed_gdf)
        current_union = safe_geometry(unary_union(current_projected.geometry))
        proposed_geometries = [
            safe_geometry(geometry)
            for geometry in proposed_projected.geometry
            if geometry is not None and not geometry.is_empty
        ]
        proposed_union = safe_geometry(unary_union(proposed_geometries))
    except Exception:
        return {}

    if current_union is None or current_union.is_empty or proposed_union is None or proposed_union.is_empty:
        return {}

    current_area = float(current_union.area)
    if current_area <= 0:
        return {}

    try:
        gap_area = float(current_union.difference(proposed_union).area)
        outside_area = float(proposed_union.difference(current_union).area)
    except Exception:
        gap_area = np.nan
        outside_area = np.nan

    overlap_area = 0.0
    assigned_geometry = None
    for geometry in proposed_geometries:
        try:
            if assigned_geometry is not None:
                overlap_area += float(geometry.intersection(assigned_geometry).area)
                assigned_geometry = safe_geometry(assigned_geometry.union(geometry))
            else:
                assigned_geometry = geometry
        except Exception:
            continue

    return {
        "coverage_share": max(0.0, min(1.0, 1.0 - (gap_area / current_area))) if pd.notna(gap_area) else np.nan,
        "gap_share": gap_area / current_area if pd.notna(gap_area) else np.nan,
        "overlap_share": overlap_area / current_area if pd.notna(overlap_area) else np.nan,
        "outside_share": outside_area / current_area if pd.notna(outside_area) else np.nan,
    }


def display_plan_topology_status(district_gdf: gpd.GeoDataFrame, proposed_items: list[dict[str, Any]]) -> None:
    """Show whether a proposed plan covers the current district layer cleanly."""
    diagnostics = plan_topology_diagnostics(district_gdf, proposed_items)
    if not diagnostics:
        return

    topology_columns = st.columns(4)
    topology_columns[0].metric("Plan Coverage", format_percent(diagnostics.get("coverage_share")))
    topology_columns[1].metric("Internal Gaps", format_percent(diagnostics.get("gap_share")))
    topology_columns[2].metric("Internal Overlap", format_percent(diagnostics.get("overlap_share")))
    topology_columns[3].metric("Outside Current Map", format_percent(diagnostics.get("outside_share")))

    gap_share = safe_numeric(pd.Series([diagnostics.get("gap_share")])).iloc[0]
    overlap_share = safe_numeric(pd.Series([diagnostics.get("overlap_share")])).iloc[0]
    outside_share = safe_numeric(pd.Series([diagnostics.get("outside_share")])).iloc[0]
    if max(gap_share, overlap_share, outside_share) <= 0.001:
        st.success("Topology check: the proposed plan has no meaningful gaps or overlaps against the current district map.")
    else:
        st.warning(
            "Topology check: this plan has visible gaps, overlaps, or area outside the current map. "
            "Treat the outcome numbers as exploratory."
        )


def display_proposed_plan_mode(
    district_gdf: gpd.GeoDataFrame,
    state_name: str,
    simplify: bool,
) -> None:
    """Compare a multi-district proposed plan against the current district plan."""
    if district_gdf is None or district_gdf.empty:
        return

    st.subheader("Proposed Plan Mode")
    st.caption(
        "Edit current boundaries, draw multiple districts, or upload a proposed-plan GeoJSON. Estimates update from current district overlaps."
    )

    plan_mode = st.radio(
        "Proposed plan method",
        ["Edit current boundaries", "Draw or upload custom plan"],
        horizontal=True,
        key=f"{state_name}_proposed_plan_method",
    )

    proposed_items: list[dict[str, Any]] = []

    if plan_mode == "Edit current boundaries":
        edit_session_key = f"{state_name}_editable_current_plan_features"
        current_items = current_plan_items(district_gdf, simplify=False)
        baseline_items = current_plan_items(district_gdf, simplify=simplify)

        if st.button("Reset boundary edits", key=f"{state_name}_reset_editable_boundaries"):
            st.session_state.pop(edit_session_key, None)

        st.caption(
            "Use the map edit tool, drag district boundary vertices, then save the edit in the map toolbar. "
            "The app repairs the changed area into neighboring districts and updates the plan summary below."
        )
        edit_map = create_editable_current_plan_map(district_gdf, state_name, simplify)
        draw_output = st_folium(
            edit_map,
            height=560,
            use_container_width=True,
            returned_objects=["last_active_drawing", "all_drawings"],
            key=f"{state_name}_editable_current_plan_map",
        )
        proposed_items, changed_labels = edited_current_plan_items(
            draw_output,
            edit_session_key,
            current_items,
            baseline_items,
        )
        if changed_labels:
            label_text = ", ".join(changed_labels[:12])
            if len(changed_labels) > 12:
                label_text += f", +{len(changed_labels) - 12} more"
            st.caption(f"Edited district boundary detected for: {label_text}.")
        else:
            st.info("Current boundaries are loaded as the baseline. After you drag and save an edit, this becomes the proposed plan.")

        preview_map = create_proposed_plan_preview_map(district_gdf, state_name, simplify, proposed_items)
        st_folium(
            preview_map,
            height=420,
            use_container_width=True,
            returned_objects=[],
            key=f"{state_name}_editable_plan_preview_map",
        )
    else:
        upload_key = f"{state_name}_proposed_plan_geojson_upload"
        uploaded_file = st.file_uploader(
            "Upload proposed plan GeoJSON",
            type=["geojson", "json"],
            key=upload_key,
        )
        uploaded_items = uploaded_geojson_plan(uploaded_file)
        if uploaded_file is not None and not uploaded_items:
            st.warning("The uploaded file could not be read as proposed district GeoJSON.")

        session_key = f"{state_name}_proposed_plan_drawings"
        if st.button("Clear proposed plan drawings", key=f"{state_name}_clear_proposed_plan"):
            st.session_state.pop(session_key, None)

        draw_map = create_proposed_plan_map(district_gdf, state_name, simplify, uploaded_items)
        draw_output = st_folium(
            draw_map,
            height=560,
            use_container_width=True,
            returned_objects=["last_active_drawing", "all_drawings"],
            key=f"{state_name}_proposed_plan_draw_map",
        )
        drawn_items = drawn_plan_geometries(draw_output, session_key)
        proposed_items = uploaded_items if uploaded_items else drawn_items

        if not proposed_items:
            st.info("Draw multiple polygons/rectangles or upload a GeoJSON FeatureCollection to compare a proposed plan.")
            return

    estimates = estimate_proposed_plan(district_gdf, proposed_items)
    if estimates.empty:
        st.warning("The proposed plan does not overlap the current district layer.")
        return

    if len(estimates) != len(district_gdf):
        st.warning(
            f"This proposed plan has {len(estimates):,} estimated district(s), while the current plan has "
            f"{len(district_gdf):,}. The comparison is useful, but it is not a full same-size replacement plan."
        )

    summary_tab, district_tab = st.tabs(["Plan Outcome Summary", "Proposed Districts"])
    with summary_tab:
        display_plan_topology_status(district_gdf, proposed_items)
        st.dataframe(
            proposed_plan_summary_table(district_gdf, estimates),
            hide_index=True,
            use_container_width=True,
        )
    with district_tab:
        display_table = proposed_plan_display_table(estimates)
        table_height = min(560, 38 * (len(display_table) + 1))
        st.dataframe(display_table, hide_index=True, use_container_width=True, height=table_height)


def redistricting_metrics_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return a formatted district-level redistricting metrics table."""
    if gdf is None or gdf.empty:
        return pd.DataFrame()

    attrs = gdf.drop(columns="geometry", errors="ignore").copy()
    id_column = district_id_column(gdf)
    district_values = attrs[id_column] if id_column in attrs.columns else pd.Series(attrs.index, index=attrs.index)
    table = pd.DataFrame(
        {
            "District": district_values.map(format_district_label),
            "Competitiveness": attrs.get("competitiveness_label", pd.Series("Not available", index=attrs.index)),
            "Winner margin": attrs.get("winner_margin", pd.Series(np.nan, index=attrs.index)).map(format_percent),
            "CVAP status": attrs.get("majority_minority_status", pd.Series("Not available", index=attrs.index)),
            "Minority CVAP": attrs.get("minority_cvap_pct", pd.Series(np.nan, index=attrs.index)).map(format_percent),
            "Compactness": attrs.get("compactness_polsby_popper", pd.Series(np.nan, index=attrs.index)).map(format_decimal_score),
            "Area": attrs.get("district_area_sq_mi", pd.Series(np.nan, index=attrs.index)).map(format_square_miles),
            "Perimeter": attrs.get("district_perimeter_mi", pd.Series(np.nan, index=attrs.index)).map(format_miles),
        }
    )

    sort_values = safe_numeric(attrs.get("compactness_polsby_popper", pd.Series(np.nan, index=attrs.index)))
    table["_compactness_sort"] = sort_values
    table = table.sort_values(
        ["_compactness_sort", "District"],
        ascending=[True, True],
        key=lambda series: series.map(natural_sort_key) if series.name == "District" else series,
        na_position="last",
    )
    return table.drop(columns="_compactness_sort")


def district_name_from_row(row: pd.Series, gdf: gpd.GeoDataFrame) -> str:
    id_column = district_id_column(gdf)
    if id_column in row.index:
        return format_district_label(row[id_column])
    return format_district_label(row.name)


def display_redistricting_metrics(gdf: gpd.GeoDataFrame) -> None:
    """Show compactness, representation, and competitiveness metrics."""
    st.subheader("Redistricting Metrics")

    compactness = safe_numeric(gdf.get("compactness_polsby_popper", pd.Series(dtype="float64")))
    winner_margin = safe_numeric(gdf.get("winner_margin", pd.Series(dtype="float64")))
    minority_cvap = safe_numeric(gdf.get("minority_cvap_pct", pd.Series(dtype="float64")))

    valid_compactness = compactness.dropna()
    most_compact = "Not available"
    least_compact = "Not available"
    if not valid_compactness.empty:
        most_row = gdf.loc[valid_compactness.idxmax()]
        least_row = gdf.loc[valid_compactness.idxmin()]
        most_compact = district_name_from_row(most_row, gdf)
        least_compact = district_name_from_row(least_row, gdf)

    metrics_columns = st.columns(5)
    metrics_columns[0].metric("Average Compactness", format_decimal_score(valid_compactness.mean()))
    metrics_columns[1].metric("Most Compact", most_compact)
    metrics_columns[2].metric("Least Compact", least_compact)
    metrics_columns[3].metric("Majority-Minority Districts", format_number((minority_cvap >= 0.50).sum()))
    metrics_columns[4].metric("Competitive Districts", format_number((winner_margin <= 0.10).sum()))

    st.caption(
        "Compactness uses the Polsby-Popper score. Competitive districts are those with a two-party winner margin of 10 percentage points or less."
    )
    table = redistricting_metrics_table(gdf)
    if table.empty:
        st.info("Redistricting metrics could not be calculated for this district layer.")
    else:
        table_height = min(560, 38 * (len(table) + 1))
        st.dataframe(table, hide_index=True, use_container_width=True, height=table_height)


@st.cache_data(show_spinner=False)
def load_comparison_districts(app_version: str = APP_VERSION) -> tuple[dict[str, gpd.GeoDataFrame], list[str]]:
    """Load compact district files for the CA vs LA comparison panel."""
    del app_version
    comparison_data: dict[str, gpd.GeoDataFrame] = {}
    messages: list[str] = []

    for state_name in STATE_CONFIG:
        path = deploy_district_path(state_name)
        if not path.exists():
            messages.append(f"{state_name}: comparison file was not found at {path.name}.")
            continue
        try:
            gdf = gpd.read_file(path, engine="pyogrio")
            gdf = ensure_wgs84(gdf, f"{state_name} comparison districts", messages)
            gdf = add_redistricting_metrics(gdf)
        except Exception as exc:
            messages.append(f"{state_name}: comparison data could not be loaded: {exc}")
            continue
        comparison_data[state_name] = gdf

    return comparison_data, messages


def comparison_state_summary_rows(comparison_data: dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    """Create one formatted summary row per state."""
    rows = []
    for state_name, gdf in comparison_data.items():
        summary = summarize_metrics(gdf)
        winner_margin = safe_numeric(gdf.get("winner_margin", pd.Series(dtype="float64")))
        minority_cvap = safe_numeric(gdf.get("minority_cvap_pct", pd.Series(dtype="float64")))
        compactness = safe_numeric(gdf.get("compactness_polsby_popper", pd.Series(dtype="float64")))
        young_turnout = safe_numeric(gdf.get("young_voter_turnout", pd.Series(dtype="float64")))
        rows.append(
            {
                "State": state_name,
                "Districts": format_number(summary.get("rows")),
                "Statewide D share": format_percent(summary.get("statewide_democratic_share")),
                "Statewide R share": format_percent(summary.get("statewide_republican_share")),
                "Average turnout": format_percent(summary.get("average_turnout_rate")),
                "Average young turnout": format_percent(young_turnout.mean()),
                "Average minority CVAP": format_percent(minority_cvap.mean()),
                "Majority-minority districts": format_number((minority_cvap >= 0.50).sum()),
                "Competitive districts": format_number((winner_margin <= 0.10).sum()),
                "Average compactness": format_decimal_score(compactness.mean()),
            }
        )
    return pd.DataFrame(rows)


def comparison_district_frame(comparison_data: dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    """Create a combined district table with raw numeric columns for comparison."""
    rows = []
    for state_name, gdf in comparison_data.items():
        id_column = district_id_column(gdf)
        for index, row in gdf.drop(columns="geometry", errors="ignore").iterrows():
            district_value = row[id_column] if id_column in row.index else index
            dem_share = safe_numeric(pd.Series([row.get("dem_share")])).iloc[0]
            rep_share = safe_numeric(pd.Series([row.get("rep_share")])).iloc[0]
            if pd.isna(dem_share) or pd.isna(rep_share):
                vote_winner = "Not available"
            else:
                vote_winner = "Democratic" if dem_share >= rep_share else "Republican"
            rows.append(
                {
                    "State": state_name,
                    "District": format_district_label(district_value),
                    "Competitiveness": row.get("competitiveness_label", "Not available"),
                    "Vote winner": vote_winner,
                    "Winner margin": safe_numeric(pd.Series([row.get("winner_margin")])).iloc[0],
                    "Democratic share": dem_share,
                    "Overall turnout": safe_numeric(pd.Series([row.get("turnout_rate")])).iloc[0],
                    "Young turnout": safe_numeric(pd.Series([row.get("young_voter_turnout")])).iloc[0],
                    "Minority CVAP": safe_numeric(pd.Series([row.get("minority_cvap_pct")])).iloc[0],
                    "Black CVAP": safe_numeric(pd.Series([row.get("black_cvap_pct")])).iloc[0],
                    "Latino CVAP": safe_numeric(pd.Series([row.get("latino_cvap_pct")])).iloc[0],
                    "Compactness": safe_numeric(pd.Series([row.get("compactness_polsby_popper")])).iloc[0],
                    "Area": safe_numeric(pd.Series([row.get("district_area_sq_mi")])).iloc[0],
                    "D + R votes": safe_numeric(pd.Series([row.get("total_dr_votes")])).iloc[0],
                }
            )
    return pd.DataFrame(rows)


def comparison_display_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Format the combined comparison table for Streamlit display."""
    if frame.empty:
        return pd.DataFrame()
    table = frame.copy()
    table = table.sort_values(
        ["State", "District"],
        ascending=[True, True],
        key=lambda series: series.map(natural_sort_key) if series.name == "District" else series,
    )
    return pd.DataFrame(
        {
            "State": table["State"],
            "District": table["District"],
            "Vote winner": table["Vote winner"],
            "Competitiveness": table["Competitiveness"],
            "Winner margin": table["Winner margin"].map(format_percent),
            "D share": table["Democratic share"].map(format_percent),
            "Turnout": table["Overall turnout"].map(format_percent),
            "Young turnout": table["Young turnout"].map(format_percent),
            "Minority CVAP": table["Minority CVAP"].map(format_percent),
            "Black CVAP": table["Black CVAP"].map(format_percent),
            "Latino CVAP": table["Latino CVAP"].map(format_percent),
            "Compactness": table["Compactness"].map(format_decimal_score),
            "Area": table["Area"].map(format_square_miles),
            "D + R votes": table["D + R votes"].map(format_number),
        }
    )


def comparison_leader_cell(frame: pd.DataFrame, state_name: str, column: str, largest: bool, formatter: Any) -> str:
    """Return one formatted state leader cell."""
    state_frame = frame[frame["State"] == state_name].copy()
    values = safe_numeric(state_frame[column])
    state_frame = state_frame[values.notna()].copy()
    if state_frame.empty:
        return "Not available"
    values = safe_numeric(state_frame[column])
    leader_index = values.idxmax() if largest else values.idxmin()
    leader = state_frame.loc[leader_index]
    return f"District {leader['District']} ({formatter(leader[column])})"


def comparison_leaders_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a side-by-side table of state leaders."""
    if frame.empty:
        return pd.DataFrame()

    leader_specs = [
        ("Most competitive", "Winner margin", False, format_percent),
        ("Highest minority CVAP", "Minority CVAP", True, format_percent),
        ("Highest Black CVAP", "Black CVAP", True, format_percent),
        ("Highest Latino CVAP", "Latino CVAP", True, format_percent),
        ("Highest overall turnout", "Overall turnout", True, format_percent),
        ("Highest young turnout", "Young turnout", True, format_percent),
        ("Most compact", "Compactness", True, format_decimal_score),
        ("Largest district", "Area", True, format_square_miles),
        ("Most D + R votes", "D + R votes", True, format_number),
    ]
    rows = []
    for label, column, largest, formatter in leader_specs:
        row = {"Category": label}
        for state_name in comparison_state_order(frame):
            row[state_name] = comparison_leader_cell(frame, state_name, column, largest, formatter)
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_state_order(frame: pd.DataFrame) -> list[str]:
    """Preserve configured state order for comparison display."""
    present_states = set(frame.get("State", pd.Series(dtype="object")).dropna())
    return [state_name for state_name in STATE_CONFIG if state_name in present_states]


def display_state_comparison() -> None:
    """Show side-by-side California and Louisiana comparison tabs."""
    st.subheader("CA vs LA Comparison")
    comparison_data, messages = load_comparison_districts(APP_VERSION)
    for message in messages:
        st.warning(message)
    if len(comparison_data) < 2:
        st.info("Comparison needs both compact district files.")
        return

    summary_table = comparison_state_summary_rows(comparison_data)
    district_frame = comparison_district_frame(comparison_data)
    leaders_table = comparison_leaders_table(district_frame)

    summary_tab, leaders_tab, districts_tab = st.tabs(["State Summary", "District Leaders", "All Districts"])
    with summary_tab:
        st.dataframe(summary_table, hide_index=True, use_container_width=True)
    with leaders_tab:
        st.dataframe(leaders_table, hide_index=True, use_container_width=True)
    with districts_tab:
        display_table = comparison_display_table(district_frame)
        table_height = min(560, 38 * (len(display_table) + 1))
        st.dataframe(display_table, hide_index=True, use_container_width=True, height=table_height)


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
        if state_name == "California":
            return (
                f"{base} Blue districts had more Democratic votes and red districts had more "
                "Republican votes. Darker blue means more Democratic votes; darker red means "
                "more Republican votes. Hover or click for exact votes and percentages."
            )
        return (
            f"{base} Blue districts voted more Democratic and red districts voted more Republican. "
            "Darker color means a stronger win above 50%. Hover or click for exact votes and percentages."
        )
    if layer_column == "rep_share":
        return f"{base} Darker red means a higher Republican vote share. Hover or click for exact percentages."
    if layer_column == "total_dr_votes":
        return f"{base} Darker colors mean more Democratic plus Republican votes. Hover or click for exact vote totals."
    if layer_column == "winner_margin":
        return f"{base} Darker colors mean a larger two-party winner margin. Lighter areas are more competitive."
    if layer_column == "turnout_rate":
        return f"{base} Darker green means higher turnout within this state. Hover or click for the turnout percentage."
    if layer_column and "cvap" in layer_column:
        return f"{base} Darker purple means a higher {layer_label} share. Hover or click for exact CVAP percentages."
    if layer_column == "young_voter_turnout":
        return f"{base} Darker purple means higher young voter turnout within this state. Hover or click for exact percentages."
    if layer_column == "compactness_polsby_popper":
        return f"{base} Darker green means a higher Polsby-Popper compactness score within this state."
    if layer_column == "district_area_sq_mi":
        return f"{base} Darker blue means a larger estimated district area."
    if layer_column == "district_perimeter_mi":
        return f"{base} Darker blue means a longer estimated district perimeter."
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

    if geography == "Congressional Districts":
        active_gdf = add_redistricting_metrics(active_gdf)
        district_gdf = active_gdf
        state_data["districts"] = active_gdf

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

    if geography == "Congressional Districts":
        display_district_scorecard(active_gdf, state_name)
        display_redistricting_metrics(active_gdf)
        display_state_comparison()
        display_proposed_plan_mode(active_gdf, state_name, simplify)

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
