"""Plot distributions, divergence and score for every dataset in --data.

    python scripts/skate/plot_divergence.py --data data/ --out output/skate/divergence/

Besides the per-scenario figures, ``overview_AB`` and ``overview_C`` (PNG and PDF) put each
scenario's demonstrations, divergence and score side by side, a row per scenario;
``overview_demos`` puts every scenario's demonstrations in one row; ``overview_fields`` puts
the divergence of every scenario across its top row and the score across its bottom; and
``{scenario}_phase_evolution`` (PNG and PDF) puts its x, y and theta marginals over phase side
by side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate import SCENARIOS, TrajectoryDataset  # noqa: E402
from fm_bc.skate import plots  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("output/skate/divergence"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = []
    for name in SCENARIOS:
        path = args.data / f"{name}.npz"
        if not path.exists():
            print(f"skip {name}: {path} missing")
            continue
        ds = TrajectoryDataset.load(path)
        datasets.append(ds)
        figures = {
            "distribution": lambda s: plots.plot_distribution(ds, save=s),
            "phase_density": lambda s: plots.plot_phase_density(ds, save=s),
        }
        for c, comp in enumerate(("x", "y", "theta")):
            figures[f"phase_evolution_{comp}"] = (
                lambda s, c=c: plots.plot_phase_density_evolution(ds, component=c, save=s))
        if ds.divergence is not None:
            figures |= {
                "divergence_paths": lambda s: plots.plot_divergence_paths(ds, save=s),
                "divergence_paths_xy": lambda s: plots.plot_divergence_paths(
                    ds, key="divergence_xy", save=s),
                "divergence_phase": lambda s: plots.plot_divergence_vs_phase(ds, save=s),
                "score": lambda s: plots.plot_score_snapshots(ds, save=s),
                "score_paths": lambda s: plots.plot_score_paths(ds, save=s),
            }
        for kind, draw in figures.items():
            save = args.out / f"{name}_{kind}.png"
            draw(save)
            plt.close("all")
            print(f"wrote {save}")
        for ext in ("png", "pdf"):
            save = args.out / f"{name}_phase_evolution.{ext}"
            plots.plot_phase_evolution_overview(ds, save=save)
            plt.close("all")
            print(f"wrote {save}")

    if datasets and all(d.divergence is not None for d in datasets):
        save = args.out / "summary.png"
        plots.plot_scenario_summary(datasets, save=save)
        print(f"wrote {save}")

    with_fields = [d for d in datasets if d.divergence is not None]
    if with_fields:
        for kind, draw in (("demos", plots.plot_demo_overview),
                           ("fields", plots.plot_field_overview)):
            for ext in ("png", "pdf"):
                save = args.out / f"overview_{kind}.{ext}"
                draw(with_fields, save=save)
                plt.close("all")
                print(f"wrote {save}")

    by_name = {d.scenario: d for d in datasets if d.divergence is not None}
    for tag, names in (("AB", ["single_path", "dual_goal"]), ("C", ["dual_path"])):
        if all(n in by_name for n in names):
            for ext in ("png", "pdf"):
                save = args.out / f"overview_{tag}.{ext}"
                plots.plot_scenario_overview([by_name[n] for n in names], save=save)
                plt.close("all")
                print(f"wrote {save}")


if __name__ == "__main__":
    main()
