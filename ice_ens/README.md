# ice_ens

Compares sea ice in a CAMulator + POP2 + CICE5 checkpoint against a ground-truth run.

Three steps: **load** the two datasets, **regrid the reference** onto the checkpoint's
native gx1v7 grid, **look** at them side by side. Seven modules, three notebooks.

## Which way the regridding goes

The **checkpoint stays native**. The **reference is what moves.**

- The checkpoint is the field under evaluation, so interpolating it would introduce
  regridding artifacts into the thing being judged.
- Every checkpoint is already on gx1v7, so checkpoint-vs-checkpoint and
  checkpoint-vs-reference are both a plain cell-for-cell subtract.
- POP's own `TAREA` stays the area weight for every number that comes out.

## What it reads

| | |
|---|---|
| checkpoint | `/glade/derecho/scratch/liyuanpu/g.e21.CAMULATOR_POP_<label>/run/*.pop.h.YYYY-MM.nc` |
| | `IFRAC` — "Ice Fraction from Coupler", gx1v7 curvilinear 384×320, monthly |
| ground truth | `.../wchapman/b_credit_runs/b.e21.CREDIT_climate_branch_1980_<year>_...zarr` |
| | `ICEFRAC` + `LANDFRAC`, regular lat/lon 192×288, 6-hourly, **one store per year, 1980–2014** |

Both are opened **in place**. Nothing is copied onto `/glade/work`.

## Files

| | |
|---|---|
| `data.py` | paths, finding the files, loading both sides |
| `grids.py` | rebuilds gx1v7 cell corners so ESMF can regrid conservatively |
| `regrid.py` | builds the regridder, carries the reference across, checks conservation |
| `metrics.py` | ice area and extent on the native grid, weighted by `TAREA` |
| `plot_poles.py` | polar maps — single-dataset, and checkpoint-vs-reference |
| `ensemble.py` | the multi-year climatology: mean, regrid, bias, area error, the figures, and the 12-month grid |
| `fluxes.py` | surface-flux time series — CAMulator against CESM, no regridding needed |

`../ice_climatology.ipynb` is the driver for the multi-year bias; `../ice_comparison.ipynb`
is the driver for one year, month by month; `../plot_fluxes.ipynb` is the driver for the
flux time series.

## Usage

```python
from ice_ens import data, regrid, metrics, plot_poles

model = data.load_checkpoint("ck33", years=[1980, 1983])   # inclusive range
truth = data.load_truth(years=[1980, 1983])                # same shorthand
ref   = regrid.to_model_grid(truth, model)                 # reference -> gx1v7

plot_poles.plot_month_compare(model, ref, 1980, 9, "north")
metrics.compare(model, ref, "north").to_dataframe()
```

`ref` is reused unchanged across every checkpoint — they all share gx1v7, so the
regrid happens once.

## Years

Will has one store per calendar year, **1980–2014**; the checkpoints run **1980–1990**,
so that overlap is the usable window. `data.truth_years()` lists what exists,
`data.truth_zarr(year)` gives one path.

`load_checkpoint` and `load_truth` take the same `years` shorthand — a bare int, a
list of years, or two values read as an inclusive range — so both sides are asked for
the same window in the same words. Give them the same one: if they differ, xarray
arithmetic in `metrics` falls back to the overlap without saying so.

`LANDFRAC` is fixed in time and identical in every store, so it is read once however
many years are requested. The weight files depend only on the two grids, so the same
pair serves every year and every checkpoint.

## Which method

| | |
|---|---|
| `method="bilinear"` | centres only, builds in a second. First-look difference maps. **Does not preserve an area integral** — never total an area off it. |
| `method="conservative"` | preserves the area integral. Needs cell corners on both grids, which `grids.py` reconstructs. **Required for any area or extent number.** |

`metrics.py` refuses a bilinearly regridded dataset rather than quietly returning a
wrong total.

Weight files land in `../weights/` — on `work`, not scratch, so a purge cannot take
them — and are reused automatically:

```
weights/conservative_ref192x288_to_gx1v7_384x320.nc
weights/bilinear_ref192x288_to_gx1v7_384x320.nc
```

## The reference is coarser than gx1v7

Equivalent cell width (√area), median by band:

| band | reference | gx1v7 | ratio |
|---|---|---|---|
| 60–70° | 78 km | 52 km | 1.50× |
| 70–80° | 62 km | 41 km | 1.51× |
| 80–90° | 36 km | 34 km | 1.07× |
| global | 101 km | 62 km | 1.65× |

This does not make the direction wrong. A conservative map coarse→fine preserves the
area integral exactly and invents no ice — the round trip reproduces the reference's
own native NH ice area to about 0.1%. But the reference panel looks blocky, and the
*effective* resolution of a difference map is the reference's, not gx1v7's. Fine
structure in a difference is the model's own; there is no reference detail below
~60–80 km to compare it against, whichever grid you compute on.

## Fluxes — `ice_ens/fluxes.py`

A different comparison from everything above: the **atmosphere** side, CAMulator's own
output against the CESM run, as a monthly-mean time series over a latitude band rather
than a map. **Nothing here is regridded** — CAMulator writes on the same regular
192×288 lat/lon grid CESM uses, so the two sides are directly comparable and none of
`grids.py` / `regrid.py` is involved.

| | |
|---|---|
| CAMulator | `.../g.e21.CAMULATOR_POP_<label>/run/camulator_out/<year>/camulator.h1d.<YYYY-MM-DD>.nc` — **one file per day**, 365 a year |
| CESM | the same per-year zarrs the ice comparison uses, via `data.truth_zarr` — 6-hourly |

| function | |
|---|---|
| `build_flux_cache(case, years)` | **the one-time step.** Reads every daily file once, taking all six fluxes out of it at a time, and writes both models' monthly means to scratch. ~1 min per year; skips years already built. |
| `load_flux_cache(case, source, years)` | that cache back, for any window. `source` is `"camulator"` or `"cesm"`. |
| `cache_path(case, source, year)` | where one year of one side lives. |
| `camulator_files(case, year)` | the daily files for one case-year, sorted. The `camulator_out` counterpart of `data.pop_files`. |
| `load_camulator(case, var, years)` | those files stacked into one `(time, lat, lon)` field, with a time coord parsed from the filenames. The uncached path — for a variable outside `FLUX_VARS`. |
| `load_cesm(var, years)` | the same variable out of the reference zarrs, concatenated across years. |
| `regional_mean(field, lat_range)` | `cos(lat)`-weighted mean over a band → 1-D in time. Default `(60, 90)`; `None` for the globe. |
| `monthly_mean(series, climatology)` | `climatology=True` (default) → the 12-month seasonal cycle over the whole window; `False` → the continuous monthly series. |
| `flux_series(case, camulator_var, cesm_var, ...)` | **the one call.** Both sides read off the cache, banded, reduced to months. Returns two 1-D DataArrays. `months=` narrows to certain calendar months; `cache=False` reads the raw output instead. |
| `plot_series(cam, cesm, case=...)` | the two curves on one figure. Same variable → one shared y axis; different variables → CAMulator left, CESM right. Saves to `output/<case>/fluxes/<CAM_VAR>/`. |
| `flux_map(case, camulator_var, cesm_var, ...)` | the map counterpart of `flux_series` — the band is *not* collapsed, so each side comes back as a full `(lat, lon)` field. `months=` picks a season: `7`, `(6,7,8)`, `(12,1,2)`. |
| `plot_maps(cam, cesm, case=..., region=...)` | those two as maps plus their difference. `region` is `"north"`/`"south"`/`"global"`; the polar panels reuse `plot_poles`' projection and clip, so a flux map lines up with an ice map. |
| `flux_map_months(case, cam_var, cesm_var, ...)` | twelve **separate** climatologies — each month averaged over the window on its own. Returns two `(month, lat, lon)` fields. |
| `plot_map_grid(cam, cesm, case=..., region=...)` | one panel per month on one shared colorbar. `field=` is `"difference"` (default), `"camulator"` or `"cesm"`. The flux counterpart of `ensemble.plot_month_grid`. |
| `flux_balance(case, down_var, up_var, ...)` | **a radiative pair and its net.** Returns two Datasets holding `down`, `up` and `net = down - up`. Takes the same window/band/climatology arguments as `flux_series`, which it calls twice. |
| `flux_budget(case, ...)` | the whole surface budget: `net_sw`, `net_lw`, `turbulent` and their sum. The one combination that closes to near zero — globally. |
| `plot_balance(cam, cesm, case=...)` | components on top, their sum below with a zero line. Draws whatever `flux_balance` or `flux_budget` returned; colour is the component, line style is the model. |
| `flux_daily(case, cam_var, cesm_var, when, lat_range)` | **day by day through one month.** 31 points a side, both as daily means. Reads the raw output, not the cache. `when` is `YYYY-MM` or `YYYY-MM-DD`. |
| `flux_day(case, cam_var, cesm_var, when)` | **initial conditions.** One single day from each side, no monthly mean anywhere. Returns two `(lat, lon)` fields ready for `plot_maps`. `when` is any `YYYY-MM-DD`; defaults to `1980-01-01`. |
| `map_dir(case, var, kind, region)` | where a map lands — `fluxes/<var>/spatial/<kind>/<Arctic\|Antarctic\|Global>/`. |

### Where maps land

Season above hemisphere, the same shape as the ice output's
`ice_plot/12-Month/Arctic`, but nested under the variable since fluxes are one
directory per variable:

```
output/<case>/fluxes/<VAR>/
├── <VAR>_vs_<VAR>_60-90N_1980-1984_clim.png     time series stay at the root
└── spatial/
    ├── 12-Month/{Arctic,Antarctic,Global}/      one panel per month
    ├── JJA/{Arctic,Antarctic,Global}/           one season, three panels
    ├── DJF/…   Jul/…   Annual/…                 whatever `months=` was
    └── 1980-01-01/{Arctic,Antarctic,Global}/    one day, from `flux_day`
```

The season directory is named from `months=`, so JJA and DJF never overwrite each
other, and `plot_map_grid` always writes to `12-Month/` regardless of the subset it
was given.

**One shared colorbar across all twelve panels**, scaled from every month together —
the same choice `ensemble.plot_bias_grid` makes. Per-panel scales would hide exactly
the seasonal signal the figure exists to show. Layout follows the ice grids too: three
across up to six panels, four across beyond, panels shrinking past six so a whole year
fits a page.

`years` is the same shorthand `load_checkpoint` and `load_truth` take — a bare int, a
list, or two values read as an inclusive range.

```python
from ice_ens import build_flux_cache, flux_series, plot_series

build_flux_cache("ck33", (1980, 1984))        # slow, once per checkpoint
cam, cesm = flux_series("ck33", "FLUS", "FLUS", years=(1980, 1984), lat_range=(60, 90))
plot_series(cam, cesm, case="ck33")           # -> .../fluxes/FLUS/..._1980-1984_clim.png
```

### One month, year by year

`flux_series(..., climatology=False, months=1)` gives **one point per year** — every
January in the window on a real time axis. The default `climatology=True` collapses
the window to a twelve-point seasonal cycle, and `climatology=False` with no `months`
gives the full month-by-month series; neither shows whether the two models pull apart
in one particular month, year after year.

Order matters inside the function, and it is the reason `months` cannot simply be
applied first: the monthly reduction happens **before** the month filter. On the
uncached path the raw samples are daily or 6-hourly, and resampling a January-only
series onto a regular monthly axis would fill the eleven gaps between Januarys with
NaN instead of closing them.

A single month with `climatology=True` collapses to a single point, which is rarely
what you want. The month is written into the figure title and filename, so a Januarys
figure never overwrites the all-months one beside it.

### Net radiation, and what actually closes

`FLDS_J`/`FSDS_J` are downwelling and carry *positive = energy into the surface*;
`FLUS`/`FSUS` are upwelling and carry *positive = upward*. So the net the surface keeps
is a plain `down - up`, which is what `flux_balance` returns alongside the two
components.

**Neither net is zero, and neither should be.** Over 60-90N, 1980-1984:

| | net LW | net SW | net rad | SH+LH | budget |
|---|---|---|---|---|---|
| CAMulator | -40.2 | +57.4 | +17.2 | -30.8 | **-13.6** |
| CESM | -31.5 | +46.7 | +15.2 | -26.3 | **-11.2** |

Net longwave is negative because the surface radiates away more than the atmosphere
returns. Net shortwave is near zero in the polar night and peaks past 120 W/m² at
midsummer.

`flux_budget` sums all four. The turbulent pair **adds** rather than subtracts, because
`SHFLX`/`LHFLX` are already on the positive-into-surface convention `_to_watts` stamps
on them.

**Where it closes.** Globally the budget lands at about +0.9 W/m² for CESM and +6.5 for
the emulator — the closure is real, and that 5.6 W/m² gap is an energy-budget bias in
the emulator rather than a plotting artifact. Over a latitude band it does not close:
the Arctic deficit above is heat transported into the region, which none of these six
fields carries. **A residual over a band is physics; a large residual over the globe
would be a bug**, which makes `lat_range=None` the check worth running after any change
to the sign conventions.

### Day by day through one month

`flux_series` collapses a month to one point, so a January comparison is two dots and
the weather inside the month is gone. `flux_daily` gives 31 points a side.

**Both sides come back as daily means.** CAMulator writes daily already, so CESM's
four 6-hourly steps are averaged per day to match. Drawing CESM at its native cadence
against a CAMulator line that is *already* a daily mean would make the emulator look
the steadier of the two, which is a fact about how its output was written and not
about the model. Averaged this way the two land on identical timestamps and read off
one axis.

This is the one view the cache cannot serve, since the cache is monthly means, so it
reads the raw output and takes a few seconds rather than being instant. The mean of
the 31 points reconciles exactly with the single January point from
`flux_series(..., climatology=False, months=1)`, which is the check that the daily
reduction is right.

```python
cam, cesm = flux_daily("ck70", "SHFLX", "SHFLX", when="1980-01")
fig, ax = plot_series(cam, cesm, case="ck70")
```

### Initial conditions — one day, no averaging

`flux_day` exists because everything else here goes through monthly means, which
is the wrong instrument for asking how far apart the two models *start*. By the
end of January a month of drift is already folded in.

Both sides come back as that day's **mean**. A CAMulator daily file is already the
mean of its four 6-hourly steps, so CESM's four steps for the same date are
averaged to match — comparing the emulator's day against one CESM instant would be
comparing a daily mean with a snapshot.

It does not touch the cache, so it works for any date in the run, not just the
years that have been cached. Output lands under `spatial/<date>/`.

```python
cam, cesm = flux_day("ck70", "SHFLX", "SHFLX", when="1980-01-01")
fig, axes = plot_maps(cam, cesm, case="ck70", region="north")
```

For ck70 the two are already apart on day one — a few W/m² in the global mean and
several hundred at individual cells — so the run does not start from a state
identical to CESM.

### The flux cache

The emulator writes one file per day, so a year is 365 file opens — that read is the
only slow thing in this module, and re-doing it per variable is what the cache exists
to avoid. `build_flux_cache` does it once, pulling all six of `FLUX_VARS` out of each
file in the same pass, and stores **monthly means on the full 192×288 grid** — not a
regional average, so `lat_range` stays a knob afterwards.

    /glade/derecho/scratch/tvoeller/honors_thesis/flux_cache/<case>/
        camulator_fluxes_monthly_<year>.nc      ~11 MB
        cesm_fluxes_monthly_<year>.nc           ~22 MB

**One file per year, not one per window.** A 1-year, 5-year and 10-year climatology are
three selections of the same files, and widening 1980–1984 to 1980–1989 builds five
years rather than ten. `flux_series` calls `build_flux_cache` itself, so a missing year
is filled in rather than erroring — the first plot of a checkpoint is slow and every
one after it is immediate.

ck33 is built for **1980–1989**, the ten complete years it has (1990 stops at day 161).
Roughly a minute per year for CAMulator plus ten seconds for CESM: about 10 minutes and
335 MB for the decade.

It lives on scratch because it is the one bulky thing this package writes, and scratch
is purged on a 90-day cycle. That is fine — it is derived data, and rebuilding it is one
call. Nothing else in the repo depends on it surviving.

`force=True` rebuilds a window that is already there; needed only if the underlying run
was rewritten. A year cached without one of the variables you ask for is rebuilt on its
own.

CAMulator writes 19 2-D fields — `PS TREFHT TS PRECT QFLX U10 TAUX TAUY FSDS_J FLDS_J
FSUS FLUS SHFLX LHFLX FSUTOA FLUT CLDHGH CLDLOW CLDMED` — plus `U V T Qtot` on 32
levels. CESM has all of those and ~30 more, so the two variable names are independent
arguments and need not match.

`FLUX_VARS` — `FLDS_J FSDS_J FSUS FLUS SHFLX LHFLX` — is what gets cached and what
`output/<case>/fluxes/` has directories for. Any of the other 13 2-D fields is still
reachable with `flux_series(..., cache=False)`, which reads the raw output directly;
`plot_series` creates its output directory on demand.

### Things that will bite you — fluxes

**The CAMulator files have no `time` dimension.** Each one is a single
`(latitude, longitude)` field with the date only in its filename, so
`groupby("time.month")` on a freshly opened file fails. `load_camulator` parses the date
out of the name and attaches it as `cftime.DatetimeNoLeap`, matching CESM's calendar.

**`latitude` is `float32` in CAMulator and `float64` in CESM.** The values agree to
3.8e-6 — but only **2 of the 192** are bit-identical, so xarray's exact-index alignment
silently inner-joins a `cam - cesm` down to two rows and returns nonsense.

**Casting the dtype does not fix this** — float32 → float64 changes the type and keeps
the wrong value; the two indexes are still unequal. The coordinate *values* have to be
replaced. `_match_grid` takes CESM's as canonical and stamps them on the emulator side
at read time, so both halves of the cache carry bit-identical coordinates. It checks the
gap against `COORD_TOL` (1e-4°) first and raises if the two are genuinely different
grids, rather than quietly relabelling one as the other. `longitude` already matches
exactly.

This one only shows up when something actually subtracts the two — a series plot draws
each side independently and looks perfectly fine while the difference is broken.

**All six fluxes are stored per 6-hour step, and are divided out on read.** Both models
write them as `J m-2` accumulated over one step rather than as a rate, so the raw
numbers run to millions and mean nothing on a figure axis. `STEP_SECONDS` is 21600 —
6 × 3600, **one** step, not a day — and `_to_watts` applies it in `load_flux_cache`,
`load_camulator` and `load_cesm`, the three doors into the data. Everything the notebook
sees is `W m-2`; anything outside `FLUX_VARS` is untouched, so `TREFHT` is still kelvin.

The division happens **on read, not on cache build**. The files under `flux_cache/` hold
the per-step energy exactly as the two models wrote it, so they stay a faithful copy of
the source and a change of units here never costs a rebuild. Open one directly and the
values are order 1e6; that is expected.

CESM declares the scaling itself for the four radiative fluxes — `units: J m-2 per 6hr
step`, `scaling: raw_W_m2 * 21600`. `SHFLX` and `LHFLX` carry no attributes on either
side, so they were settled by magnitude instead: the area-weighted global means over
January 1980 are -4.39e5 and -1.83e6, which divided by 21600 give -20.3 and -84.8 W/m²
against CESM's canonical ~20 and ~85. The other four land on 189, 338, 25 and 392 W/m².
The emulator side agrees on all six to within a few percent.

The sign is negative on both models, which is the positive-into-surface convention the
radiative fluxes declare for themselves. `_to_watts` stamps that on `SHFLX`/`LHFLX`
along with the unit, which is also what stops those two plotting with a bare axis label.

**The CAMulator files carry no attributes at all** — no units, no long names. They are
unlabelled versions of the same fields CESM writes, checked value by value on
1980-01-15. Units for the axis labels are therefore taken from the CESM side.

**A daily CAMulator file is the mean of the four 6-hourly steps, not a snapshot.** The
header says nothing either way — no `cell_methods`, no `time_bnds`, no global attrs — so
it was settled numerically. On 1980-06-21 at the equator, each individual CESM 6-hourly
step has 83 of 288 longitudes in darkness and peaks near 1.8e7; the CAMulator daily
`FSDS_J` has **no** dark longitudes and peaks at 5.9e6, which is what CESM's own mean of
four looks like (0 dark, 6.5e6). Globally that day: CAMulator 3.858e6 against CESM's
mean-of-four 3.867e6, with the individual steps spanning 3.74–3.96e6. So the units stay
*per 6hr step* — a mean, not a daily total — and averaging daily means over whole months
gives the same monthly mean as averaging the 6-hourly steps. **This is why the divisor
is 21600 and not 86400.** A daily file that were a sum of four steps would need the
day's length; a mean of four does not.

**The band mean is `cos(lat)`-weighted, and it has to be.** The default band is 60–90N,
which is nothing but pole; an unweighted mean over a regular lat/lon grid over-counts
the top rows badly.

**Loading is ~1 minute per year per variable** — 365 separate file opens, against one
zarr store on the CESM side. Get the figure right on 1980 before widening `years`.

## Things that will bite you

**There are no `cpl.hi` files in these cases.** The CESM2/MCT coupler history stream
was not enabled. `IFRAC` in the POP monthly stream is the same coupler field, received
by the ocean, and is what `Check_dataset.ipynb` was already reading.

**POP stamps a monthly file at the end of its averaging interval** — January's file
carries a timestamp of 1 February. Trusting it shifts the whole seasonal cycle by a
month. The month is taken from the filename instead, and matching between the two
datasets is done on `(year, month)` integer coords.

**Sea fraction is `1 - LANDFRAC`, never `OCNFRAC`.** CAM's three fractions satisfy
`LANDFRAC + OCNFRAC + ICEFRAC = 1`, so `OCNFRAC` is the *open* ocean and collapses to
~0 under the pack. Dividing by it to undo land dilution deletes the solid ice instead
and turns the Arctic seasonal cycle into something that peaks in June. This was caught
by checking the regridded NH area against the reference's own native NH area — worth
repeating after any change here.

**The two ice fractions mean different things until you fix them.** POP `IFRAC` is ice
fraction of an *ocean* cell, and gx1v7's land mask is binary — a gx1v7 ocean cell is
wholly ocean. CAM `ICEFRAC` is ice area over *total* cell area, so its coastal cells are
diluted by their land part. `to_model_grid` divides that dilution out with the
regridded sea fraction and returns both conventions: `IFRAC` (POP's, for differencing)
and `ICEFRAC` (CAM's, area-conserving, for the conservation check).

**The two land masks do not agree.** A few percent of the reference's ice lands on cells
POP calls land and is dropped — 2% in summer, 5% in winter. `check_conservation`
reports it as `land_fraction_of_total`. It is a real mask mismatch, not a regridding
error, and it is why `IFRAC` totals sit slightly below the reference's native total.

**gx1v7's cell corners are not in the file** and have to be rebuilt from `ULAT`/`ULONG`
(POP stores those as each T cell's *northeast* corner). Two traps, both handled in
`grids.py`: the periodic seam falls mid-row so longitudes need unwrapping, and in the
tripole rows a row does not advance a net 360°, so the wrap-around column is snapped
onto its neighbour's branch rather than shifted by a fixed −360. The reconstruction
matches POP's own `TAREA` to 4e-4 everywhere except the southernmost row, which is
extrapolated and is entirely land under Antarctica.

`regrid.check_conservation` is how you confirm all of that held: ice area in ≈ ice area
out, currently −4.2e-4.
