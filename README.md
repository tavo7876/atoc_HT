# ice_ens — climatological sea-ice bias of the CAMulator checkpoints

Multi-year mean `IFRAC` bias of every CAMulator + POP2 + CICE5 checkpoint against one
CESM reference run, as a small-multiples map plus a hemispheric ice-area error plot.

Run it on the **NPL kernel** — `/glade/u/apps/opt/conda/envs/npl` — which is where
`xesmf` lives. The default JupyterHub kernel does not have it.

## The seven functions — `ice_ens/ensemble.py`

| | |
|---|---|
| `climatology(path, months, years, var)` | load (zarr or netCDF), keep the chosen months+years, time-mean → one 2-D field. **The only place month/year selection happens** — edit here to change what gets averaged. |
| `regrid_to_gx1v7(ref_field, target_field, weights_path)` | conservative regrid of the reference onto the checkpoints' native gx1v7. Built once, reused. |
| `ice_area(field, cell_area, lat, hemisphere)` | `sum(field * cell_area)` over one hemisphere → scalar in 10⁶ km². |
| `bias_ensemble(checkpoints, reference_path, months, years, hemisphere, ref_var, weights_path)` | drives all of the above over `{label: path}`. Returns `(biases, area_errors)`. |
| `plot_bias_grid(biases, hemisphere, fig_dir, months, years, title, name)` | one polar map per **dict key**, one shared RdBu_r colorbar centred on zero. |
| `plot_area_error(area_errors, hemisphere, fig_dir, months, years, title, name)` | one dot per dict key, zero line = reference. |
| `plot_month_grid(checkpoint_path, reference_path, years, hemisphere, label, months, ...)` | one checkpoint's whole seasonal cycle: a bias panel per month, laid out 4×3, plus the matching area-error figure. Twelve *separate* climatologies, not an annual mean. |

Everything is a plain argument. No config file, no globals, no paths baked in.

Drawing reuses `ice_ens/plot_poles.py` (polar axis, pcolormesh-vs-contour, gx1v7
coordinate cleanup) and `ice_ens/grids.py` (gx1v7 cell corners). Neither is duplicated.

## Data flow

checkpoint → `climatology` → 2-D field on gx1v7
reference → `climatology` → 2-D field on lat/lon → `regrid_to_gx1v7` → gx1v7
**bias = checkpoint − reference**, per checkpoint → `plot_bias_grid`
`ice_area` on each **native** grid → error per checkpoint → `plot_area_error`

The reference is climatologied once and regridded once; every checkpoint reuses it.

## Example

```python
from ice_ens import data, bias_ensemble, plot_bias_grid, plot_area_error

CASE    = "ck33"
FIG_DIR = data.out_dir(CASE, "ensemble")        # -> output/ck33/ensemble/

biases, areas = bias_ensemble(
    {CASE: f"/glade/derecho/scratch/liyuanpu/g.e21.CAMULATOR_POP_{CASE}/run"},
    reference_path="/glade/derecho/scratch/wchapman/b_credit_runs/"
                   "b.e21.CREDIT_climate_branch_1980_*_zmdata_ERA5scaled_zmdata_Qtot.zarr",
    months=[8], years=("1980", "1984"), hemisphere="north", ref_var="ICEFRAC",
    weights_path=f"{data.WEIGHT_DIR}/conservative_ref192x288_to_gx1v7_384x320.nc",
)
plot_bias_grid(biases, "north", fig_dir=FIG_DIR)
plot_area_error(areas, "north", fig_dir=FIG_DIR, months=[8], years=("1980", "1984"))
```

For the whole seasonal cycle instead of one window — twelve panels, one a month, on one
shared colorbar — that entire block collapses to:

```python
from ice_ens import plot_month_grid

fig, biases, areas = plot_month_grid(
    f"/glade/derecho/scratch/liyuanpu/g.e21.CAMULATOR_POP_{CASE}/run",
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "b.e21.CREDIT_climate_branch_1980_*_zmdata_ERA5scaled_zmdata_Qtot.zarr",
    ("1980", "1989"), "north", label=CASE,
    weights_path=f"{data.WEIGHT_DIR}/conservative_ref192x288_to_gx1v7_384x320.nc",
    fig_dir=FIG_DIR,
)
```

About two minutes for a year. One checkpoint per call — two would be 24 panels.

Output goes to `output/<case>/ensemble/`, alongside that case's `figures/` from the
single-year comparison — `data.out_dir(case, kind)` builds and creates the directory.

`months=[6,7,8]` gives a JJA *mean* — one field, one panel. For the three months
**side by side** instead, call `bias_ensemble` once per month and key the dict by month
name: both plot functions draw one panel per key and don't care what the keys mean, so
the three share one colorbar. Pass `title=` and `name=` when you do, or the figure is
captioned as a seasonal mean and overwrites the file that really is one. Add checkpoints
by adding dict entries; loop over both for every checkpoint x month in one figure.

## Conventions

- **bias = checkpoint − reference. Positive = TOO MUCH ice.** Everywhere, both figures.
- **Checkpoints are never regridded.** They are the field under evaluation and they all
  already share gx1v7, so the reference is the thing that moves.
- **Ice area is computed on native grids**, one side at a time — checkpoint with POP's
  `TAREA` (cm², so ×1e-4), reference with its own cell area — so neither total
  inherits a regridding error.

## Things that will bite you

- **The reference is one zarr per calendar year, 1980–2014.** Pass the `*` glob above,
  not a single store, or a "5-year mean" is really one year.
- **POP stamps a monthly file at the end of its interval** — `pop.h.1980-01.nc` (the
  January mean) carries a time of 1980-02-01. `climatology` uses the `time_bound`
  midpoint instead; selecting on the raw stamp shifts the seasonal cycle by a month.
- **A checkpoint that doesn't cover the whole window is skipped, loudly.** Checkpoint
  spans: ck70 → 1990-10, ck33 → 1990-05, ck36 → 1987-12, ck69 → 1987-05, ck66 → 1985-12,
  **ck64 → 1981-12 only**. All start 1980-01. A 20-year window is not possible.
- **`ICEFRAC` and `IFRAC` are not the same denominator**, so every coastline carries a
  spurious *positive* bias. Currently unhandled, by decision — see below.

## Things to mention while building

### 1. The coastal cells are exaggerated — CAM `ICEFRAC` vs POP `IFRAC`

The reference's only sea-ice variable is CAM's `ICEFRAC` = ice area / **total** cell
area. The checkpoints report POP's `IFRAC` = ice area / the cell's **ocean** part. On
any cell that is partly land, `ICEFRAC` ≤ `IFRAC` for the same physical ice, so the
reference reads too *low* there and `bias = checkpoint − reference` is pushed *up*.
Every coastline therefore carries a spurious **positive** — too-much-ice — bias. It is
an artifact of the two conventions, not a model error. Interior ocean is unaffected.

Measured on ck33, August, 1980–1984, Arctic: mean bias binned by the reference's
regridded sea fraction (`1 − LANDFRAC`), where 1.0 = a cell with no land in it.

| regridded sea fraction | cells | mean bias |
|---|---|---|
| 0.00–0.50 | 211 | **+0.139** |
| 0.50–0.90 | 2906 | +0.040 |
| 0.90–0.99 | 1683 | +0.029 |
| 0.99–1.00 | 536 | +0.034 |
| ≈ 1.00 (open ocean) | 33620 | **+0.016** |

Dividing the regridded `ICEFRAC` by the regridded sea fraction drops the coastal mean
from +0.040 to +0.012 and leaves the open-ocean mean at +0.016 **unchanged**. That
asymmetry is the tell — a real model error would move both.

**Question: is the coastal artifact acceptable in a climatological bias map, or should
the un-dilution be applied?** If applied, the denominator is `1 - LANDFRAC` and **not**
`OCNFRAC` — in CAM `LANDFRAC + OCNFRAC + ICEFRAC = 1`, so `OCNFRAC` is the *open* ocean
and goes to zero under the pack, which would delete the solid ice instead of
un-diluting it.

`ice_climatology.ipynb` reproduces this table for whichever
checkpoint is loaded, so it can be re-checked at any time.

### 2. The hemispheric area errors are NOT affected by that

Worth saying plainly, because it is the obvious next thing to doubt. `ice_area` totals
each side on its **own native grid**:

- checkpoint, `sum(IFRAC × TAREA)` — gx1v7's land mask is binary, so an ocean cell is
  *wholly* ocean and this is real ice area;
- reference, `sum(ICEFRAC × cell_area)` — ice/total × total is also real ice area.

Neither side is regridded, so no dilution and no interpolation error can enter. The
check, Arctic August 1980–1984, in 10⁶ km²:

| | |
|---|---|
| reference, own native grid | 5.825 |
| reference regridded, raw | 5.617 |
| reference regridded, un-diluted | 5.820 |
| ck33, own native grid | 6.388 |

The regrid loses 0.21 to dilution and the land-mask mismatch — which is exactly why the
area numbers are not computed that way. **ck33's +0.563 Arctic error is real.**

Same window, all checkpoints that cover it: ck33 **+0.56**, ck36 **+0.87**, ck66
**+0.48**, ck69 **+0.48**, ck70 **+1.37**. Every one has too much Arctic ice in August;
ck70 is worst by a wide margin.

### 3. A 20-year window is not reachable

The longest checkpoint is ck70 at 10 years 10 months. 1980–1989 is the ceiling, and
only ck33 and ck70 cover it. The reference runs to 2014, so it is the checkpoints that
constrain this, not the ground truth.

### 4. The reference is averaged over the same window as the checkpoint

Both sides get the same `months`/`years` — Aug 1980–84 checkpoint against Aug 1980–84
reference. The alternative, a longer reference baseline (e.g. 1980–2014), answers a
different question: bias against the long-term climate rather than bias over this
window. Worth deciding deliberately rather than by default.
