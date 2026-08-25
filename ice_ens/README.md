# ice_ens

Compares sea ice in a CAMulator + POP2 + CICE5 checkpoint against a ground-truth run.

Three steps: **load** the two datasets, **regrid the reference** onto the checkpoint's
native gx1v7 grid, **look** at them side by side. Six modules, two notebooks.

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

`../ice_climatology.ipynb` is the driver for the multi-year bias; `../ice_comparison.ipynb`
is the driver for one year, month by month.

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
