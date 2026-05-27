# -*- coding: utf-8 -*-
"""전략 비교 실험 유틸리티 테스트."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from experiments import (
    format_summary_table,
    plot_experiment_grid,
    plot_experiment_movement,
    run_experiments,
    summarize_results,
)


def test_run_experiments_records_history_and_lr_decay():
    np.random.seed(123)
    x_train = np.random.rand(128, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, 128)
    x_test = np.random.rand(32, 784).astype(np.float32)
    y_test = np.random.randint(0, 10, 32)

    configs = [
        {
            "name": "constant",
            "model_kwargs": {
                "use_batchnorm": False,
                "use_dropout": False,
                "init_method": "normal",
            },
            "optimizer_kwargs": {"lr": 0.001},
        },
        {
            "name": "decay",
            "model_kwargs": {
                "use_batchnorm": False,
                "use_dropout": False,
                "init_method": "normal",
            },
            "optimizer_kwargs": {"lr": 0.01},
            "lr_schedule": lambda epoch: 0.01 * (0.5 ** epoch),
        },
    ]

    results = run_experiments(
        configs,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=2,
        batch_size=64,
        seed=7,
        train_eval_size=32,
        verbose=False,
    )

    assert [result["name"] for result in results] == ["constant", "decay"]
    assert len(results[0]["history"]) == 2
    assert results[1]["history"][0]["lr"] == 0.01
    assert results[1]["history"][1]["lr"] == 0.005

    for result in results:
        for row in result["history"]:
            assert row["train_loss"] >= 0
            assert row["val_loss"] >= 0
            assert 0 <= row["train_acc_pct"] <= 100
            assert 0 <= row["val_acc_pct"] <= 100


def test_summary_and_plot_helpers():
    results = [
        {
            "name": "sample",
            "params": 123,
            "history": [
                {
                    "epoch": 1,
                    "lr": 0.001,
                    "train_loss": 1.0,
                    "train_eval_loss": 0.9,
                    "train_acc_pct": 75.0,
                    "val_loss": 1.1,
                    "val_acc_pct": 70.0,
                },
                {
                    "epoch": 2,
                    "lr": 0.001,
                    "train_loss": 0.5,
                    "train_eval_loss": 0.4,
                    "train_acc_pct": 90.0,
                    "val_loss": 0.6,
                    "val_acc_pct": 88.0,
                },
            ],
        }
    ]

    summary = summarize_results(results)
    assert summary[0]["best_epoch"] == 2
    assert "sample" in format_summary_table(summary)

    fig = plot_experiment_grid(results, zoom=5)
    assert len(fig.axes) == 2

    auto_fig = plot_experiment_grid(results, auto_scale=True)
    assert len(auto_fig.axes) == 2

    movement_fig = plot_experiment_movement(results)
    assert len(movement_fig.axes) == 2
