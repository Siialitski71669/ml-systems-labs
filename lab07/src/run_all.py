import argparse, csv, json, os, sys, time
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from models import make_teacher, make_student, count_parameters
from engine import (set_seed, make_loaders, train_epochs, evaluate, save_checkpoint,
                    load_checkpoint, count_nonzero_parameters, dense_size_bytes,
                    sparse_format_size_bytes, measure_latency)
from pruning import apply_magnitude_pruning, remove_pruning_reparam, structured_compact_copy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    os.chdir(ROOT.parent)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = Path(cfg["output"]["results_dir"]); ckpts = Path(cfg["output"]["checkpoints_dir"])
    results.mkdir(parents=True, exist_ok=True); ckpts.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = make_loaders(cfg)
    tr = cfg["training"]; bench = cfg["benchmark"]
    print(f"Device: {device}")
    teacher = make_teacher().to(device)
    teacher_params = count_parameters(teacher)
    print(f"Teacher parameters: {teacher_params:,}")
    teacher_path = ckpts / "teacher_baseline.pth"
    if teacher_path.exists():
        print("Loading existing teacher checkpoint.")
        load_checkpoint(teacher, teacher_path, device)
    else:
        print("Training oversized teacher...")
        train_epochs(teacher, train_loader, val_loader, device, tr["teacher_epochs"],
                     tr["learning_rate"], tr["weight_decay"])
        save_checkpoint(teacher, teacher_path)
    test_loss, teacher_acc = evaluate(teacher, test_loader, device)
    teacher_latency, _ = measure_latency(teacher, test_loader, device, bench["warmup_batches"],
                                          bench["repeats"], bench["cpu_threads"])
    teacher_dense = dense_size_bytes(teacher, ckpts / "teacher_dense_measure.pth")
    teacher_sparse = sparse_format_size_bytes(teacher, ckpts / "teacher_sparse_measure")
    teacher_nz, teacher_total = count_nonzero_parameters(teacher)
    baseline = {"model":"Teacher baseline","family":"baseline","sparsity":0.0,
                "accuracy":teacher_acc,"parameters":teacher_params,"nonzero_parameters":teacher_nz,
                "latency_ms_per_image":teacher_latency,"dense_size_bytes":teacher_dense,
                "sparse_format_bytes":teacher_sparse}
    all_models = [baseline]
    pruning_rows = []

    # Global and layer-wise unstructured pruning, each independently starts from baseline teacher.
    for mode in ("global", "layerwise"):
        for amount in cfg["pruning"]["sparsities"]:
            print(f"Pruning: {mode}, target={amount:.0%}")
            model = make_teacher().to(device); model.load_state_dict(teacher.state_dict())
            masks = apply_magnitude_pruning(model, float(amount), mode)
            if amount > 0:
                train_epochs(model, train_loader, val_loader, device, tr["finetune_epochs"],
                             tr["learning_rate"] * 0.2, tr["weight_decay"], masks=masks)
            remove_pruning_reparam(model)
            _, acc = evaluate(model, test_loader, device)
            nz, total = count_nonzero_parameters(model)
            actual = 1.0 - nz / max(total,1)
            latency, _ = measure_latency(model, test_loader, device, bench["warmup_batches"],
                                          bench["repeats"], bench["cpu_threads"])
            dense = dense_size_bytes(model, ckpts / f"{mode}_{int(amount*100)}_dense.pth")
            sparse = sparse_format_size_bytes(model, ckpts / f"{mode}_{int(amount*100)}_sparse")
            row = {"method":mode,"target_sparsity":amount,"actual_sparsity":actual,
                   "accuracy":acc,"parameters":total,"nonzero_parameters":nz,
                   "latency_ms_per_image":latency,"dense_size_bytes":dense,
                   "sparse_format_bytes":sparse}
            pruning_rows.append(row)
            all_models.append({"model":f"{mode} {amount:.0%}","family":mode,**{k:v for k,v in row.items() if k not in ("method","target_sparsity","actual_sparsity")}})
            del model

    pd.DataFrame(pruning_rows).to_csv(results / "pruning_results.csv", index=False)

    structured_rows = []
    for rate in cfg["pruning"]["structured_rates"]:
        print(f"Structured filter pruning: {rate:.0%}")
        compact = structured_compact_copy(teacher, float(rate))
        train_epochs(compact, train_loader, val_loader, device, tr["finetune_epochs"],
                     tr["learning_rate"] * 0.2, tr["weight_decay"])
        _, acc = evaluate(compact, test_loader, device)
        params = count_parameters(compact)
        latency, _ = measure_latency(compact, test_loader, device, bench["warmup_batches"],
                                      bench["repeats"], bench["cpu_threads"])
        dense = dense_size_bytes(compact, ckpts / f"structured_{int(rate*100)}.pth")
        nz, total = count_nonzero_parameters(compact)
        row = {"target_filter_removal":rate,"accuracy":acc,"parameters":params,
               "latency_ms_per_image":latency,"dense_size_bytes":dense,
               "nonzero_parameters":nz}
        structured_rows.append(row)
        all_models.append({"model":f"Structured {rate:.0%}","family":"structured",**row})
    pd.DataFrame(structured_rows).to_csv(results / "structured_pruning_results.csv", index=False)

    # Student without a teacher (ordinary CE).
    print("Training student without distillation...")
    plain_student = make_student().to(device)
    train_epochs(plain_student, train_loader, val_loader, device, tr["student_epochs"],
                 tr["learning_rate"], tr["weight_decay"])
    _, plain_acc = evaluate(plain_student, test_loader, device)
    save_checkpoint(plain_student, ckpts / "student_plain.pth")
    plain_latency, _ = measure_latency(plain_student, test_loader, device, bench["warmup_batches"],
                                        bench["repeats"], bench["cpu_threads"])
    plain_size = dense_size_bytes(plain_student, ckpts / "student_plain_size.pth")
    all_models.append({"model":"Student (CE)","family":"student","accuracy":plain_acc,
                       "parameters":count_parameters(plain_student),
                       "latency_ms_per_image":plain_latency,"dense_size_bytes":plain_size})

    # Student with KD: alpha*CE + (1-alpha)*T^2*KL(student || teacher).
    print("Training student with knowledge distillation...")
    kd_student = make_student().to(device)
    train_epochs(kd_student, train_loader, val_loader, device, tr["student_epochs"],
                 tr["learning_rate"], tr["weight_decay"], teacher=teacher,
                 temperature=cfg["distillation"]["temperature"],
                 alpha=cfg["distillation"]["alpha"])
    _, kd_acc = evaluate(kd_student, test_loader, device)
    save_checkpoint(kd_student, ckpts / "student_distilled.pth")
    kd_latency, _ = measure_latency(kd_student, test_loader, device, bench["warmup_batches"],
                                     bench["repeats"], bench["cpu_threads"])
    kd_size = dense_size_bytes(kd_student, ckpts / "student_distilled_size.pth")
    all_models.append({"model":"Student (distilled)","family":"student","accuracy":kd_acc,
                       "parameters":count_parameters(kd_student),
                       "latency_ms_per_image":kd_latency,"dense_size_bytes":kd_size})

    models_df = pd.DataFrame(all_models)
    models_df.to_csv(results / "model_comparison.csv", index=False)

    # Plot unstructured pruning accuracy against achieved sparsity.
    p_df = pd.DataFrame(pruning_rows)
    fig, ax = plt.subplots(figsize=(8,5))
    for mode, group in p_df.groupby("method"):
        ax.plot(group["actual_sparsity"]*100, group["accuracy"]*100, marker="o", label=mode)
    ax.set(xlabel="Actual weight sparsity (%)", ylabel="Test accuracy (%)",
           title="Unstructured magnitude pruning: accuracy vs sparsity")
    ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(results / "pruning_accuracy.png", dpi=160); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9,5))
    subset = models_df[models_df["accuracy"].notna()].copy()
    ax1.scatter(subset["latency_ms_per_image"], subset["accuracy"]*100)
    for _, r in subset.iterrows():
        ax1.annotate(str(r["model"]), (r["latency_ms_per_image"], r["accuracy"]*100), fontsize=7)
    ax1.set(xlabel="CPU latency (ms/image)", ylabel="Test accuracy (%)",
            title="Accuracy–latency trade-off")
    ax1.grid(True, alpha=.3); fig.tight_layout()
    fig.savefig(results / "model_tradeoffs.png", dpi=160); plt.close(fig)

    # Energy proxy: transparent estimate, not a direct energy measurement.
    power = float(bench["nominal_cpu_power_watts"])
    models_df["estimated_energy_mj_per_image"] = (
        models_df["latency_ms_per_image"] * power
    )
    models_df.to_csv(results / "model_comparison.csv", index=False)
    pillars = [
        {"pillar":"Data","measurement":"Fashion-MNIST; same train/validation/test split for all runs",
         "result":"Fixed dataset and preprocessing; no data compression used"},
        {"pillar":"Model","measurement":"Parameters, nonzero weights, checkpoint size, test accuracy",
         "result":f"Teacher={teacher_params:,} params; plain student={count_parameters(plain_student):,}; distilled student={count_parameters(kd_student):,}"},
        {"pillar":"Compute","measurement":"CPU inference latency (ms/image), batch size and thread count",
         "result":f"Teacher={teacher_latency:.4f} ms/image; CE student={plain_latency:.4f}; distilled student={kd_latency:.4f}"},
        {"pillar":"Energy","measurement":"Estimated latency × nominal CPU power ({power:g} W); proxy only",
         "result":"Relative estimate; not a hardware energy-meter measurement"},
    ]
    pd.DataFrame(pillars).to_csv(results / "efficiency_pillars.csv", index=False)

    report = f"""# Lab 07 — Compression: Pruning and Knowledge Distillation

## Experimental setup
- Dataset: Fashion-MNIST, normalized grayscale 28×28 images.
- Device: `{device}`.
- Teacher parameters: **{teacher_params:,}**.
- Student parameters: **{count_parameters(plain_student):,}**.
- Distillation: temperature **{cfg['distillation']['temperature']}**, alpha **{cfg['distillation']['alpha']}**.
- Latency: CPU inference, {bench['cpu_threads']} thread(s), batch size up to 128, warm-up {bench['warmup_batches']} batches, {bench['repeats']} timed repetitions.

## Baseline
- Teacher test accuracy: **{teacher_acc*100:.2f}%**.
- Teacher CPU latency: **{teacher_latency:.4f} ms/image**.
- Dense checkpoint: **{teacher_dense/1024:.1f} KiB**.
- Sparse-format representation: **{teacher_sparse/1024:.1f} KiB**.

## Pruning
See `pruning_results.csv`, `structured_pruning_results.csv`, and `pruning_accuracy.png`.
Global pruning selects the smallest absolute weights across all prunable layers. Layer-wise pruning applies the target independently to each prunable layer. Masks are re-applied after every optimizer update during the three fine-tuning epochs. Structured pruning removes complete output filters selected by L2 norm and rebuilds a smaller dense network.

## Distillation
| Model | Test accuracy | Parameters | CPU latency (ms/image) | Dense checkpoint (KiB) |
|---|---:|---:|---:|---:|
| Student trained with CE | {plain_acc*100:.2f}% | {count_parameters(plain_student):,} | {plain_latency:.4f} | {plain_size/1024:.1f} |
| Student trained with KD | {kd_acc*100:.2f}% | {count_parameters(kd_student):,} | {kd_latency:.4f} | {kd_size/1024:.1f} |

The distillation objective is `alpha * CE(student, labels) + (1-alpha) * T^2 * KL(student/T || teacher/T)`, with T={cfg['distillation']['temperature']} and alpha={cfg['distillation']['alpha']}.

## Efficiency pillars (Chapter 9)
- **Data:** All approaches use the same Fashion-MNIST split and preprocessing, so the comparison isolates model-side changes.
- **Model:** Compare parameter count, nonzero weights, dense checkpoint size, and sparse-format size. Unstructured pruning does not reduce tensor dimensions.
- **Compute:** Compare measured CPU latency. Sparse weight count alone does not guarantee faster dense-kernel execution. Structured pruning can improve latency because it creates smaller dense layers.
- **Energy:** `estimated_energy_mj_per_image` is a proxy calculated from latency and a nominal CPU power setting ({power:g} W). It is not a direct energy measurement.

## Interpretation and conclusion
Use the CSV measurements and plots to identify the best accuracy–size–latency trade-off on this machine. In general, unstructured pruning mainly reduces the number of nonzero values and can reduce a sparse-format representation, but dense CPU kernels still process dense tensor shapes. Structured pruning physically reduces channels and can therefore reduce dense computation. Distillation transfers teacher output information to a compact student; whether it improves the student depends on the measured test accuracy.

**Important:** all values above are generated by this run. Hardware-specific latency and energy-proxy values should not be generalized to other computers.
"""
    (results / "report_generated.md").write_text(report, encoding="utf-8")
    print("\nLab 07 completed. Results:", results.resolve())

if __name__ == "__main__":
    main()
