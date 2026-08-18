# ice_ens

Compares sea ice in a CAMulator + POP2 + CICE5 checkpoint against a ground-truth run.

Three steps: **load** the two datasets, **regrid** the checkpoint onto the truth's
grid, **look** at them side by side. Four modules, one notebook.

## What it reads

| | |
|---|---|
| checkpoint | `/glade/derecho/scratch/liyuanpu/g.e21.CAMULATOR_POP_<label>/run/*.pop.h.YYYY-MM.nc` |
| | `IFRAC` — "Ice Fraction from Coupler", gx1v7 curvilinear 384×320, monthly |
| ground truth | `.../wchapman/b_credit_runs/b.e21.CREDIT_climate_branch_1980_1980_...zarr` |
| | `ICEFRAC`, regular lat/lon 192×288, 6-hourly, calendar year 1980 |

Both are opened **in place**. Nothing is copied onto `/glade/work`.

## Files

| | |
|---|---|
| `data.py` | paths, finding the files, loading both sides |
| `grids.py` | rebuilds gx1v7 cell corners so ESMF can regrid conservatively |
| `regrid.py` | builds the regridder, moves the checkpoint across, checks conservation |
| `plot_poles.py` | polar maps — single-dataset, and checkpoint-vs-truth |

`../ice_comparison.ipynb` is the driver. Start there.

## Usage

```python
from ice_ens import data, regrid, plot_poles

model = data.load_checkpoint("ck33", years=[1980])
truth = data.load_truth()
model = regrid.to_truth_grid(model, truth)

plot_poles.plot_month_compare(model, truth, 1980, 9, "north")   # 3 panels, one month
plot_poles.plot_year_compare(model, truth, 1980, "north")        # 12 months, 2 rows
plot_poles.save_figures(model, truth, "ck33", 1980)              # write both, both poles
```

Output lands in `output/<label>/figures/`. Weight files go in `weights/` and are
reused automatically.

Pass a short label (`"ck33"`) or a full case name — `data.case_name` expands the
former into `g.e21.CAMULATOR_POP_ck33`. Anything not fitting that pattern must be
given in full.

## Knobs

Top of `plot_poles.py`:

- `ICE_LEVELS` — contour bands, currently steps of 0.1. Drawing cost scales roughly
  linearly with the number of levels, so 0.05 looks smoother and takes twice as long.
- `DIFF_LEVELS`, `ICE_CMAP`, `DIFF_CMAP` — scales and colormaps.
- `COASTLINE` — `"50m"`. `"110m"` is coarser and only ~4 s faster per 12-panel figure.

Top of `data.py`: input paths, `OUTPUT_DIR`, `WEIGHT_DIR`.

## Things that will bite you

**There are no `cpl.hi` files in these cases.** The CESM2/MCT coupler history stream
was not enabled. `IFRAC` in the POP monthly stream is the same coupler field, received
by the ocean, and is what `Check_dataset.ipynb` was already reading.

**The ground truth covers 1980 only**, so that is the comparison window however far a
checkpoint runs.

**POP stamps a monthly file at the end of its averaging interval** — January's file
carries a timestamp of 1 February. Trusting it shifts the whole seasonal cycle by a
month. The month is taken from the filename instead, and matching between the two
datasets is done on `(year, month)` integer coords.

**The two ice fractions mean different things until you fix them.** POP `IFRAC` is ice
fraction of an *ocean* cell, and gx1v7's land mask is binary. CAM `ICEFRAC` is ice area
over *total* cell area, so coastal cells are diluted by their land part. Filling POP
land with zero before a conservative regrid reproduces CAM's convention, which is what
puts the two on the same footing.

**gx1v7's cell corners are not in the file** and have to be rebuilt from `ULAT`/`ULONG`
(POP stores those as each T cell's *northeast* corner). Two traps, both handled in
`grids.py`: the periodic seam falls mid-row so longitudes need unwrapping, and in the
tripole rows a row does not advance a net 360°, so the wrap-around column is snapped
onto its neighbour's branch rather than shifted by a fixed −360. The reconstruction
matches POP's own `TAREA` to 4e-4 everywhere except the southernmost row, which is
extrapolated and is entirely land under Antarctica.

`regrid.check_conservation` is how you confirm all of that held: ice area in ≈ ice area
out, currently 4e-4.
