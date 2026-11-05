# Data

| Folder | Content | Versioned |
|---|---|---|
| `raw/` | Downloads exactly as received (OWID CSV) | no, reproducible via loaders |
| `interim/` | Cached descriptions, intermediate tables | no |
| `processed/` | Model-ready tables | no |
| `external/` | Hand-curated inputs: Wikipedia title overrides, manual descriptions | **yes** |

## Curated inputs

- `external/wikipedia_titles.csv` maps dataset names to Wikipedia article titles. The OWID
  entity names are matched best-effort; check the spike notebook output for technologies
  with missing descriptions and extend this file.
- `external/descriptions_manual.csv` (optional, columns `technology,description,source`)
  overrides any fetched text. Use it for leakage-free descriptions written from the
  perspective of the launch period, e.g. from early patents or contemporary press.

## Sources and citation

- Our World in Data, *Technology adoption by households in the United States*, compiled
  from Comin & Hobijn (2004) and others. Licensed CC BY by OWID; cite the original sources.
- Comin, D. & Hobijn, B. (2009). *The CHAT Dataset*. NBER Working Paper 15319 (planned for v1).
- Wikipedia summaries via the REST API (CC BY-SA).
