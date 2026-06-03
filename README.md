# DistrictLens: California & Louisiana Redistricting Explorer

This project is an interactive Python and Streamlit dashboard for comparing California and Louisiana congressional redistricting patterns. It combines congressional district boundaries, 2024 precinct election results, L2 turnout statistics, and 2023 CVAP demographic data.

The app is designed for a class project: it inspects the data files, detects likely column names, calculates useful map layers when possible, and keeps running with clear warnings when a layer cannot be created.

## Folder Setup

Keep the project folder organized like this:

```text
CA-LA Model/
├── app.py
├── data/
│   ├── CA/
│   │   ├── ca_2024_gen_prec.zip
│   │   ├── ca_cong_adopted_2025.zip
│   │   ├── ca_cvap_2023_b.zip
│   │   └── CA_l2_2024_gen_stats_2020block.zip
│   └── LA/
│       ├── la_2024_gen_prec.zip
│       ├── la_cong_adopted_2024.zip
│       ├── la_cvap_2023_b.zip
│       └── LA_l2_2024_gen_stats_2020block.zip
├── outputs/
├── deploy_data/
├── README.md
└── requirements.txt
```

The app automatically extracts each ZIP into a generated folder next to the ZIP file. You do not need to unzip the data manually.

`deploy_data/` contains compact congressional district GeoJSON files for Streamlit Community Cloud. These files let the public app run without uploading the very large raw ZIP files.

## Install

Create and activate a virtual environment, then install the packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The app reads geospatial files with `pyogrio`, which avoids many local GDAL setup issues. Geospatial Python packages can still be sensitive to Python versions. If installation is difficult, try Python 3.10, 3.11, or 3.12.

## Run

```bash
streamlit run app.py
```

The dashboard opens in your browser. Use the sidebar to choose a state, geography, map layer, district overlay, city markers, and simplified geometry.

## District Scorecard

Congressional district view includes a district scorecard for California and Louisiana. Select a district to see the vote winner, D/R margin, competitive rank, minority CVAP share, and young turnout. The ranking table can sort districts by competitiveness, minority CVAP share, young turnout, overall turnout, Democratic margin, Republican margin, winner margin, or D + R votes.

## Proposed Plan Mode

Congressional district view also includes proposed plan mode. Draw a small boundary-adjustment area and the app automatically detects which district loses area and which neighboring district gains it; direct editable-boundary and draw/upload modes are also available. The app repairs edited boundary changes into a same-size proposed plan, checks for gaps/overlaps, and estimates vote share, turnout, CVAP demographics, compactness, and overlap with current districts. The plan summary compares estimated outcomes against the current plan, including seats, competitiveness, majority-minority CVAP districts, compactness, and turnout. Estimates are approximate because they are area-weighted from the current congressional district layer.

The final embedded proposed-plan HTML maps are `my_calimap.html` and `Louisian_Proposed_plan.html`. Congressional district view shows these maps inside the app with a short title and description, then displays the fairness metric table directly below each map. In the California section, `proposed_california_plan.html` appears first for comparison with its own fairness metrics, followed by the final map titled `Final California Proposed Plan` with a separate final-plan fairness metric table. In the Louisiana section, the original `LA proposed reock map.html` appears first for comparison with its own fairness metrics, followed by the final map titled `Final Proposed Louisiana Plan` with a separate final-plan fairness metric table.

## Redistricting Metrics

Congressional district view includes redistricting metrics for compactness, district area, district perimeter, majority-minority CVAP status, and competitiveness. Compactness uses the Polsby-Popper score, where higher values are more compact. Competitive districts are those with a two-party winner margin of 10 percentage points or less.

## CA vs LA Comparison

Congressional district view includes a comparison section that loads the compact California and Louisiana district files side by side. The comparison has tabs for statewide summaries, district leaders, and a combined district table covering vote share, turnout, CVAP, compactness, area, and vote totals.

## Deploy

This repo is ready for Streamlit Community Cloud:

```text
Repository: shahzahans/DistrictLens
Branch: main
Main file path: app.py
```

The public deployed app uses `deploy_data/` by default because the full `data/` folder is intentionally not committed to GitHub. The deployed version supports congressional district maps and summary layers. Full precinct views require the local ZIP files listed above.

## Performance Tips

California's precinct GeoJSON is very large. For a fast first load, keep **Load full precinct election data** turned off when viewing California congressional districts. In that mode, the app loads district boundaries and block-level CVAP/L2 layers, then uses a compact cache in `outputs/cache/` on future runs.

Turn **Load full precinct election data** on only when you need Democratic or Republican vote-share layers aggregated from precinct results. This can take longer the first time, but the processed district output is cached afterward.

For precinct maps, the app draws a limited number of precinct features by default so the browser does not freeze. Use the **Precinct features to draw** slider if you want to show more features.

The **Save cleaned outputs** button is manual. Avoid saving full California precinct GeoJSON unless you really need it, because that file can be close to a gigabyte.

## Datasets

### California

- `ca_cong_adopted_2025.zip`: 2025 adopted congressional district boundary plan.
- `ca_2024_gen_prec.zip`: 2024 general election precinct-level results and precinct boundaries.
- `CA_l2_2024_gen_stats_2020block.zip`: L2 voter file turnout statistics aggregated to 2020 Census blocks.
- `ca_cvap_2023_b.zip`: 2023 Citizen Voting Age Population data disaggregated to 2020 Census blocks.

### Louisiana

- `la_cong_adopted_2024.zip`: 2024 adopted congressional district boundary plan.
- `la_2024_gen_prec.zip`: 2024 general election precinct-level results and precinct boundaries.
- `LA_l2_2024_gen_stats_2020block.zip`: L2 voter file turnout statistics aggregated to 2020 Census blocks.
- `la_cvap_2023_b.zip`: 2023 Citizen Voting Age Population data disaggregated to 2020 Census blocks.

## Map Layers

The sidebar groups available map layers into clearer categories. A category only appears when at least one layer in that group can be calculated.

### Votes

- Party winner: blue districts had more Democratic votes and red districts had more Republican votes. In California, darker blue means more Democratic votes and darker red means more Republican votes. Louisiana keeps the stronger-winner shading used in the deployed map.
- D + R votes: Democratic plus Republican votes.

### Turnout

- Overall turnout: votes divided by registered voters when those fields are available.
- Young turnout: estimated turnout for young voter age groups, usually 18-24, when L2 age fields are detected.

### CVAP demographics

- Black CVAP
- Latino CVAP
- Asian CVAP
- White CVAP
- Minority CVAP

For election results, the app creates:

- `total_dr_votes = dem_votes + rep_votes`
- `dem_share = dem_votes / total_dr_votes`
- `rep_share = rep_votes / total_dr_votes`
- `turnout_rate = total_votes / registered_voters`, when both columns exist

For CVAP, the app creates:

- `black_cvap_pct = black_cvap / total_cvap`
- `latino_cvap_pct = latino_cvap / total_cvap`
- `asian_cvap_pct = asian_cvap / total_cvap`
- `white_cvap_pct = white_cvap / total_cvap`
- `minority_cvap_pct = 1 - white_cvap_pct`

## Important Notes

**D + R vote total** means Democratic votes plus Republican votes. It is not total population, and it may not equal all ballots cast because third-party, write-in, blank, and other contest-specific votes are not included.

**CVAP** means Citizen Voting Age Population. It estimates the voting-age citizen population, which is different from total population and registered voters.

The L2 and CVAP files are block-level CSVs. The app tries to aggregate them to congressional districts using the district block assignment CSVs inside the congressional district ZIP files. Precinct-level CVAP or L2 layers only appear if the source data has a reliable shared join key with the precinct geography.

## Outputs

When enabled in the sidebar, the app writes cleaned files to `outputs/`:

- Cleaned GeoJSON for loaded congressional districts
- Cleaned GeoJSON for loaded precincts
- Summary CSV files for each geography
- A basic state summary CSV

There is also a download button for the currently selected geography data.

## Limitations

- Column names differ across sources, so the app uses flexible detection instead of hardcoded names.
- Some layers may be unavailable if the needed columns are missing or if block-level data cannot be joined to the selected geography.
- Large precinct files, especially California, can take time to load and render. The simplified geometry option improves performance.
- Turnout rate depends on registered voter fields being present. If registered voter data is missing, the app can only use an existing turnout field if one is detected.
- The dashboard is exploratory and educational. It should not be treated as a legal redistricting analysis by itself.
