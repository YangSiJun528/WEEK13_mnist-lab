# -*- coding: utf-8 -*-
"""MNIST 전략 비교 실험 유틸리티."""

import math

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss
from network import NeuralNetwork
from optimizers import Adam


def default_experiment_configs():
    """보고서용 6개 비교 전략을 반환합니다."""
    return [
        {
            "name": "baseline",
            "model_kwargs": {
                "use_batchnorm": True,
                "use_dropout": True,
                "dropout_ratio": 0.5,
                "init_method": "he",
            },
            "optimizer_kwargs": {"lr": 0.001},
        },
        {
            "name": "high_lr",
            "model_kwargs": {
                "use_batchnorm": True,
                "use_dropout": True,
                "dropout_ratio": 0.5,
                "init_method": "he",
            },
            "optimizer_kwargs": {"lr": 0.01},
        },
        {
            "name": "lr_decay",
            "model_kwargs": {
                "use_batchnorm": True,
                "use_dropout": True,
                "dropout_ratio": 0.5,
                "init_method": "he",
            },
            "optimizer_kwargs": {"lr": 0.01},
            "lr_schedule": lambda epoch: 0.01 * math.pow(0.6, epoch),
        },
        {
            "name": "no_dropout",
            "model_kwargs": {
                "use_batchnorm": True,
                "use_dropout": False,
                "dropout_ratio": 0.0,
                "init_method": "he",
            },
            "optimizer_kwargs": {"lr": 0.001},
        },
        {
            "name": "no_batchnorm",
            "model_kwargs": {
                "use_batchnorm": False,
                "use_dropout": True,
                "dropout_ratio": 0.5,
                "init_method": "he",
            },
            "optimizer_kwargs": {"lr": 0.001},
        },
        {
            "name": "xavier_init",
            "model_kwargs": {
                "use_batchnorm": True,
                "use_dropout": True,
                "dropout_ratio": 0.5,
                "init_method": "xavier",
            },
            "optimizer_kwargs": {"lr": 0.001},
        },
    ]


def _accuracy(y_pred, y_true):
    return float(np.mean(np.argmax(y_pred, axis=1) == y_true) * 100)


def _evaluate_metrics(model, x, y):
    y_pred = model.predict(x)
    return {
        "loss": float(cross_entropy_loss(y_pred, y)),
        "acc_pct": _accuracy(y_pred, y),
    }


def _train_one_epoch(model, optimizer, x_train, y_train, batch_size):
    train_size = x_train.shape[0]
    indices = np.random.permutation(train_size)
    epoch_loss = 0.0
    batch_count = 0

    for start in range(0, train_size, batch_size):
        batch_indices = indices[start:start + batch_size]
        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]

        y_pred = model.forward(x_batch, train=True)
        loss = cross_entropy_loss(y_pred, y_batch)

        dout = y_pred.copy()
        dout[np.arange(x_batch.shape[0]), y_batch] -= 1
        dout /= x_batch.shape[0]

        model.backward(dout)
        optimizer.update(model.params, model.grads)

        epoch_loss += loss
        batch_count += 1

    return float(epoch_loss / batch_count)


def run_experiments(
    configs,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=20,
    batch_size=128,
    seed=42,
    train_eval_size=10000,
    verbose=True,
):
    """
    여러 학습 전략을 순서대로 실행하고 epoch별 metric을 기록합니다.

    configs 항목 형식:
        {
            "name": "baseline",
            "model_kwargs": {...},
            "optimizer_kwargs": {"lr": 0.001},
            "lr_schedule": optional callable(epoch) -> lr,
        }
    """
    rng = np.random.default_rng(seed)
    train_eval_size = min(train_eval_size, x_train.shape[0])
    train_eval_indices = rng.choice(x_train.shape[0], size=train_eval_size, replace=False)
    x_train_eval = x_train[train_eval_indices]
    y_train_eval = y_train[train_eval_indices]

    results = []

    for config_index, config in enumerate(configs):
        name = config["name"]
        model_kwargs = config.get("model_kwargs", {})
        optimizer_kwargs = config.get("optimizer_kwargs", {})
        lr_schedule = config.get("lr_schedule")

        np.random.seed(seed + config_index)
        model = NeuralNetwork(**model_kwargs)
        optimizer = Adam(**optimizer_kwargs)
        history = []

        if verbose:
            print(f"\n[{name}] start")

        for epoch in range(epochs):
            if lr_schedule is not None:
                optimizer.lr = float(lr_schedule(epoch))

            train_loss = _train_one_epoch(model, optimizer, x_train, y_train, batch_size)
            train_metrics = _evaluate_metrics(model, x_train_eval, y_train_eval)
            val_metrics = _evaluate_metrics(model, x_test, y_test)

            epoch_record = {
                "epoch": epoch + 1,
                "lr": float(optimizer.lr),
                "train_loss": train_loss,
                "train_eval_loss": train_metrics["loss"],
                "train_acc_pct": train_metrics["acc_pct"],
                "val_loss": val_metrics["loss"],
                "val_acc_pct": val_metrics["acc_pct"],
            }
            history.append(epoch_record)

            if verbose:
                print(
                    f"epoch {epoch + 1:02d}/{epochs} "
                    f"lr={optimizer.lr:.6f} "
                    f"train_loss={train_loss:.4f} "
                    f"train_acc={train_metrics['acc_pct']:.2f}% "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['acc_pct']:.2f}%"
                )

        results.append(
            {
                "name": name,
                "config": config,
                "history": history,
                "model": model,
                "params": int(sum(p.size for p in model.params.values())),
            }
        )

    return results


def summarize_results(results):
    """실험 결과를 보고서에 옮기기 쉬운 list[dict] 형태로 요약합니다."""
    summary = []
    for result in results:
        history = result["history"]
        final = history[-1]
        best = max(history, key=lambda item: item["val_acc_pct"])
        summary.append(
            {
                "name": result["name"],
                "final_val_acc_pct": final["val_acc_pct"],
                "best_val_acc_pct": best["val_acc_pct"],
                "best_epoch": best["epoch"],
                "final_train_acc_pct": final["train_acc_pct"],
                "final_val_loss": final["val_loss"],
                "final_lr": final["lr"],
                "params": result["params"],
            }
        )
    return summary


def format_summary_table(summary):
    """summary list를 Markdown 표 문자열로 변환합니다."""
    lines = [
        "| strategy | final val acc | best val acc | best epoch | train acc | val loss | final lr | params |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {name} | {final_val_acc_pct:.2f}% | {best_val_acc_pct:.2f}% | "
            "{best_epoch} | {final_train_acc_pct:.2f}% | {final_val_loss:.4f} | "
            "{final_lr:.6f} | {params:,} |".format(**row)
        )
    return "\n".join(lines)


def plot_experiment_grid(results, zoom=1):
    """
    전략별 train/test loss와 accuracy를 그립니다.

    zoom=5이면 accuracy는 80~100%, zoom=10이면 90~100% 영역을 확대합니다.
    loss도 0 근처를 볼 수 있도록 전체 최대 loss를 zoom 배율로 줄여 표시합니다.
    """
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows), squeeze=False)

    all_losses = []
    for result in results:
        all_losses.extend([row["train_loss"] for row in result["history"]])
        all_losses.extend([row["val_loss"] for row in result["history"]])
    max_loss = max(all_losses) if all_losses else 1.0
    loss_top = max_loss if zoom <= 1 else max(0.05, max_loss / zoom)
    acc_bottom = 0.0 if zoom <= 1 else max(0.0, 100.0 - 100.0 / zoom)

    for row_index, result in enumerate(results):
        history = result["history"]
        epochs = [row["epoch"] for row in history]

        ax_loss = axes[row_index][0]
        ax_acc = axes[row_index][1]

        ax_loss.plot(epochs, [row["train_loss"] for row in history], label="train loss")
        ax_loss.plot(epochs, [row["val_loss"] for row in history], label="test loss")
        ax_loss.set_title(f"{result['name']} loss (zoom x{zoom})")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.set_ylim(0, loss_top)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend()

        ax_acc.plot(epochs, [row["train_acc_pct"] for row in history], label="train acc")
        ax_acc.plot(epochs, [row["val_acc_pct"] for row in history], label="test acc")
        ax_acc.set_title(f"{result['name']} accuracy (zoom x{zoom})")
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("accuracy (%)")
        ax_acc.set_ylim(acc_bottom, 100.0)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend()

    fig.tight_layout()
    return fig
