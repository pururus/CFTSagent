"""Графики метрик и таблицы LaTeX для статьи."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from experiments.evaluation import CUTOFFS, METRICS
from experiments.model_suite import OBJECTIVES


METRIC_LABELS = {"Recall": "Recall", "NDCG": "NDCG", "FullCoverage": "FC"}


def _save_figure(figure, path):
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_metrics(curves, path, *, context=""):
    """Построить три K-кривые и общую легенду с полными именами методов."""
    models = list(dict.fromkeys(curves.model))
    colors = plt.get_cmap("tab20")
    styles = ("-", "--", ":", "-.")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, metric in zip(axes, METRICS):
        for index, model in enumerate(models):
            frame = curves[(curves.model == model) & (curves.metric == metric)].sort_values("k")
            axis.plot(frame.k, frame.value, color=colors(index % 20),
                      linestyle=styles[index // 20 % len(styles)],
                      marker=("o", "s", "^", "D")[index % 4],
                      markevery=4, markersize=3, linewidth=1.7, label=model)
        axis.set(xlabel="K", ylabel=METRIC_LABELS[metric], ylim=(0, 1),
                 xticks=[1, 3, 5, 10, 15, 20])
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.01),
                  ncol=2, frameon=False, fontsize=8)
    figure.suptitle(context, fontsize=14)
    figure.tight_layout()
    _save_figure(figure, path)


def plot_family_curves(curves, directory, *, context="", selected_models=()):
    """Сохранить Recall@K, NDCG@K и FC@K всех вариантов Family CF."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    configurations = {
        model: model.split(" | ")[-2]
        for model in curves["model"].unique()
    }
    markers = dict(zip(selected_models, ("o", "s", "^")))
    colors = {"MSE": "#0072B2", "BPR/random": "#D55E00", "BPR/semantic": "#009E73"}
    styles = {"MSE": "-", "BPR/random": "--", "BPR/semantic": ":"}
    filenames = {"Recall": "recall", "NDCG": "ndcg", "FullCoverage": "fc"}
    losses = set(configurations.values())

    for metric in METRICS:
        figure, axis = plt.subplots(figsize=(9, 5.5), layout="constrained")
        metric_curves = curves[curves["metric"] == metric].groupby("model", sort=False)
        for model, points in metric_curves:
            points = points.sort_values("k")
            loss = configurations[model]
            highlighted = model in markers
            axis.plot(points.k, points.value, color=colors[loss], linestyle=styles[loss],
                      alpha=1 if highlighted else .16, linewidth=2 if highlighted else .9,
                      zorder=3 if highlighted else 2, marker=markers.get(model),
                      markevery=3, markersize=4,
                      label=model.removeprefix("Family CF | ") if highlighted else None)
        handles = [Line2D([0], [0], color=colors[loss], linestyle=styles[loss], label=loss)
                   for loss in OBJECTIVES if loss in losses]
        selected_handles, _ = axis.get_legend_handles_labels()
        axis.legend(handles=handles + selected_handles, loc="upper left", fontsize=8)
        label = METRIC_LABELS[metric]
        axis.set(title=" · ".join(filter(None, (context, "Family CF", label + "@K"))),
                 xlabel="K", ylabel=label + "@K", ylim=(0, 1),
                 xticks=[1, 3, 5, 10, 15, 20])
        axis.grid(alpha=.2)
        _save_figure(figure, directory / f"family_{filenames[metric]}_by_k.png")


def _best_model(curves, method):
    """Выбрать конфигурацию по FC@10, Recall@10, NDCG@10 и имени."""
    frame = curves[(curves.k == 10) & curves.model.str.startswith(method + " | ")]
    scores = frame.pivot(index="model", columns="metric", values="value").reset_index()
    if scores.empty:
        raise ValueError(f"В таблице выбора отсутствует {method}")
    return scores.sort_values(["FullCoverage", "Recall", "NDCG", "model"],
                              ascending=[False, False, False, True]).iloc[0]["model"]


def plot_run_results(run, groups=("G1", "G2", "G3")):
    """Построить графики Family и четыре сравнения из сохранённых CSV запуска.

    Для старых запусков без общей validation-таблицы Query/Projected выбираются
    описательно по ToolLens test. На ToolBench применяется тот же выбор.
    Источник выбора записывается в model_groups_selected.csv.
    """
    run = Path(run)
    validation = run / "ToolLens" / "validation"
    ranking_path = validation / "validation_ranking.csv"
    selected_family = pd.read_csv(ranking_path, nrows=3)["model"].tolist() if ranking_path.is_file() else []
    test_outputs = {
        "ToolLens test": run / "ToolLens" / "test",
        **{f"ToolBench {group} test": run / "ToolBench" / group for group in groups},
    }
    family_outputs = {
        "ToolLens validation": validation,
        **{context: directory / "family" for context, directory in test_outputs.items()},
    }
    for context, directory in family_outputs.items():
        curves_path = directory / "curves.csv"
        if curves_path.is_file():
            plot_family_curves(pd.read_csv(curves_path), directory, context=context,
                               selected_models=selected_family)
        else:
            print(f"{context}: таблица кривых Family ещё не сохранена")

    selection_path = validation / "comparison" / "curves.csv"
    selection_split = "ToolLens val"
    if not selection_path.is_file():
        selection_path = run / "ToolLens" / "test" / "comparison" / "curves.csv"
        selection_split = "ToolLens test (описательное сравнение)"
    records = []
    if selection_path.is_file():
        selection_curves = pd.read_csv(selection_path)
        records.extend(dict(method=method, model=_best_model(selection_curves, method),
                            selection_split=selection_split,
                            criterion="FullCoverage@10, Recall@10, NDCG@10, model")
                       for method in ("Query-KNN CF", "Projected CF"))
    if selected_family:
        records.append(dict(method="Family CF", model=selected_family[0],
                            selection_split="ToolLens val",
                            criterion="FullCoverage@10, Recall@10, NDCG@10, model"))
    selected = pd.DataFrame(records, columns=["method", "model", "selection_split", "criterion"])
    selected.to_csv(run / "model_groups_selected.csv", index=False)
    baselines = ("Popularity", "BM25", "BGE")
    titles = {"query": "Бейзлайны и Query", "projected": "Бейзлайны и Projected",
              "family": "Бейзлайны и Family", "best": "Бейзлайны и лучший от каждого метода"}
    for context, directory in test_outputs.items():
        curves_path = directory / "comparison" / "curves.csv"
        if not curves_path.is_file():
            print(f"{context}: таблица общего сравнения ещё не сохранена")
            continue
        curves = pd.read_csv(curves_path)
        method = curves.model.str.split(" | ", regex=False).str[0]
        masks = {"query": curves.model.isin(baselines) | method.eq("Query-KNN CF"),
                 "projected": curves.model.isin(baselines) | method.eq("Projected CF"),
                 "family": curves.model.isin(baselines) | method.eq("Family CF")}
        missing = set(selected.model) - set(curves.model)
        if len(selected) == 3 and not missing:
            masks["best"] = curves.model.isin([*baselines, *selected.model])
        else:
            print(f"{context}: график лучших пропущен, отсутствуют {sorted(missing) or 'результаты выбора'}")
        output = directory / "comparison" / "model_groups"
        output.mkdir(parents=True, exist_ok=True)
        for group, mask in masks.items():
            title = f"{context} · {titles[group]}"
            if group == "best":
                title += f"\nQuery/Projected — {selection_split}; Family — ToolLens val"
            plot_metrics(curves[mask], output / f"baselines_{group}.png", context=title)
    return selected


def export_results(curves, directory, *, family_grid=False, context=""):
    """Сохранить CSV, таблицы booktabs и графики, вернуть широкую таблицу.

    Для статьи создаётся отдельная таблица для каждой метрики. Короткие
    обозначения M01… и отдельная расшифровка позволяют сохранить ширину.
    Большие таблицы используют longtable. В документе нужны пакеты
    booktabs и longtable.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    section = "family" if family_grid else "comparison"
    context_label = re.sub(r"[^a-zA-Z0-9-]+", "-", context).strip("-")
    label_prefix = "tab:" + "-".join(filter(None, [context_label, directory.name, section]))
    curves = curves.groupby(["model", "metric", "k"], as_index=False, sort=False).value.mean()
    models = list(dict.fromkeys(curves.model))
    columns = pd.MultiIndex.from_product([METRICS, CUTOFFS], names=["metric", "k"])
    table = curves[curves.k.isin(CUTOFFS)].pivot(
        index="model", columns=["metric", "k"], values="value"
    ).reindex(index=models, columns=columns)
    table.to_csv(directory / "metrics.csv", float_format="%.6f")
    curves.to_csv(directory / "curves.csv", index=False, float_format="%.6f")
    codes = [f"M{index:02d}" for index in range(1, len(models) + 1)]
    legend = pd.DataFrame({"Method": codes, "Configuration": models})
    legend.to_csv(directory / "methods.csv", index=False)
    legend.to_latex(directory / "methods.tex", index=False, escape=True,
                    longtable=len(models) > 25, column_format=r"lp{0.78\linewidth}",
                    caption=f"{context} · Method configurations",
                    label=label_prefix + "-methods")
    for metric in METRICS:
        panel = table[metric].copy()
        panel.index = pd.Index(codes, name="Method")
        panel.columns = [f"{METRIC_LABELS[metric]}@{cutoff}" for cutoff in CUTOFFS]
        panel.to_latex(directory / f"{metric.lower()}.tex", escape=True,
                       longtable=family_grid or len(models) > 25,
                       float_format="%.4f", na_rep="—", column_format="lrrrr",
                       caption=f"{context} · {METRIC_LABELS[metric]} by method",
                       label=label_prefix + "-" + metric.lower())
    if not family_grid:
        plot_metrics(curves, directory / "metrics.png", context=context)
    return table
