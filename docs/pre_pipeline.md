# Pre-Pipeline

The pre-pipeline prepares raw KCC CSV data for the main pipeline by:
1. Filtering by state, optional district, and optional query domains
2. Normalizing raw crop names to canonical forms

It is orchestrated by `run_pre_pipeline.py` and called by `pipeline_server.py` inside `_run_full_sync`.

---

## Input

**`cleaned_data.csv`** — the raw Kisan Call Centre dataset stored in the Zoho WorkDrive root. Contains one row per farmer call, with columns including:

| Column | Description |
|--------|-------------|
| `StateName` | State the call came from |
| `DistrictName` | District |
| `QueryType` / `domain` | Agricultural domain (e.g. "Plant Protection", "Agronomy") |
| `KCCQueryText` | The farmer's question text |
| `Crop` | Raw crop name as typed by the call centre |
| `QueryDate` | Timestamp of the call |

---

## Stage 1 — State / District / Domain Filter

**Script:** `pre_pipeline/get_state_crop_rows.py`  
**Orchestrator call:**
```python
run_state_filter(input_path, state, intermediate, domains, district)
```

Filters the raw CSV to rows matching:
- `StateName == state` (case-insensitive substring match)
- `DistrictName == district` if provided
- `domain` (or `QueryType`) in `domains` list if provided

Writes a filtered intermediate CSV. Typical output is 10k–500k rows depending on state/domain selection.

---

## Stage 2 — Crop Name Normalization

**Script:** `pre_pipeline/crop_normalizer.py`  
**Orchestrator call:**
```python
run_crop_normalizer(intermediate, output_path, crops=None)
```

Maps each row's `Crop` value through `mapping.crop_mapping` to its canonical name. Rows whose raw crop name is not in the mapping (or whose canonical form is not in the `crops` filter list) are dropped. If `crops=None`, all mappable crops are kept.

Writes the final normalized CSV to `output_path`. This becomes the input to the main pipeline.

---

## Crop Mapping

**File:** `pre_pipeline/mapping.py`

Contains `crop_mapping: dict[str, str]` — over 400 entries mapping raw variant spellings to canonical crop names. Examples:

```python
"paddy": "Rice",
"dhan": "Rice",
"RICE": "Rice",
"wheat": "Wheat",
"gehun": "Wheat",
"tomato": "Tomato",
"tamatar": "Tomato",
...
```

The mapping is intentionally case-normalized at load time. All comparisons are done after `.lower().strip()`.

### Helper functions

```python
get_filtered_mapping(primary_crops: list[str]) -> dict[str, str]
```
Returns only entries whose canonical value is in `primary_crops`. Used when the caller specifies an explicit crop list.

```python
get_reverse_mapping(primary_crops: list[str]) -> dict[str, list[str]]
```
Returns `{canonical → [all raw variants]}`. Used for cross-crop filter keyword generation in Stage 6.

---

## Output

**`<district>_<N>.csv`** — normalized pre-pipeline output saved as `APP_DATA/<state_slug>/<district>_<N>/<district>_<N>.csv`.

The `<N>` suffix is a version number. When the same state/district is processed with different domains, `_next_versioned_path` increments `N` so previous outputs are not overwritten.

Each run also writes **`meta.json`** alongside the CSV:
```json
{
  "state": "<state>",
  "district": "<district>",
  "domains": ["Plant Protection", ...],
  "crops": ["Rice", "Wheat", ...]
}
```

`meta.json` is used by subsequent `/run/full` calls to decide whether to reuse the cached pre-pipeline output (same state/district/domains) or create a new version.

---

## Caching and Reuse Logic

The server (`_run_full_sync`) applies the following logic before running pre-pipeline:

1. **No pre-output path specified:** Run pre-pipeline into a temp file, delete after use.
2. **Pre-output path specified, file exists, same state/district/domains:**
   - If no `crops` filter: reuse as-is, discover crops from the file.
   - If `crops` filter: run only for the missing crops and append rows to the existing file.
3. **Pre-output path specified, file exists, different state/district/domains:** Version the output (`district_1/`, `district_2/`, ...) and run fresh.
4. **Pre-output path specified, file doesn't exist:** Run fresh, write to path.

Both the normalized CSV and `meta.json` are uploaded to Zoho immediately after writing.

---

## Subprocess Streaming

Both `run_state_filter` and `run_crop_normalizer` are called as **subprocesses** via `_stream()`:

```python
def _stream(cmd: list[str]):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, ...)
    for line in proc.stdout:
        print(line, end="")
        _ctl.check_cancel()
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
```

This:
- Streams output line-by-line to the job's stdout buffer (via `_JobStdout`)
- Checks the cancel event after each line so a user stop propagates within ~1 line of output
