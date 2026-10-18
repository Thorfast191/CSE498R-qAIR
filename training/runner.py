import os
import torch
from functools import partial

from torch.utils.data import DataLoader

from config import (
    CACHE_DIR,
    CKPT_DIR,
    EPOCHS,
    PATIENCE,
    PERSISTENT_STEPS,
    N_QUBITS,
    BATCH_SIZE,
    WEIGHT_DECAY,
    SEED,
    ARC_CONFIG,
    MODEL_DIM,
    PHASE_MODE,
    PHASE_SOURCE,
    MIXING,
    USE_LLM_PRIOR,
    DETERMINISTIC,
    DETERMINISTIC_WARN_ONLY,
    cache_filename,
    resolve_device,
)

from training.dataset import QAIRDataset, collate_fn
from config import EMBEDDING_DIM as DIM

from training.train import Trainer
from training.checkpoint import (
    load_best, load_or_resume, run_key, run_provenance, training_config,
)
from training.evaluate import evaluate
from training.environment import check_pins, describe, write_package_list
from training.seed import set_seed, seeded_generator

from models.full_model import QAIRvNext

device = resolve_device()


def build_loaders(cache_dir, train_samples=None, val_samples=None,
                  test_samples=None, seed=SEED, with_test=True,
                  arc_config=ARC_CONFIG):
    """
    ARC ships train / validation / test. Only train and validation were
    ever used, and validation did double duty: it selected the best
    checkpoint AND supplied the reported number, which biases every
    accuracy this project has published. The test split is loaded here so
    the final number can come from data that never took part in model
    selection.
    """

    train_ds = QAIRDataset(
        split="train", max_samples=train_samples, cache_dir=cache_dir,
        arc_config=arc_config,
    )
    val_ds = QAIRDataset(
        split="validation", max_samples=val_samples, cache_dir=cache_dir,
        arc_config=arc_config,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=seeded_generator(seed),
        collate_fn=partial(collate_fn, shuffle_options=True),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, shuffle_options=False),
    )

    test_loader = None

    if with_test:

        # Do NOT let a routine `--mode train` silently kick off an
        # hours-long LLM generation run for a split the user may not have
        # built yet. Skip with a loud, actionable message instead.
        test_cache = os.path.join(
            cache_dir, cache_filename("test", arc_config)
        )

        if not os.path.exists(test_cache):
            print(
                f"\n[NO TEST CACHE] {test_cache} does not exist.\n"
                f"  Skipping held-out test evaluation -- validation numbers\n"
                f"  only, which are optimistically biased because validation\n"
                f"  also selects the checkpoint.\n"
                f"  To build it (needs the LLM):\n"
                f"      python scripts/build_cache.py --splits test\n"
            )
        else:
            test_ds = QAIRDataset(
                split="test", max_samples=test_samples, cache_dir=cache_dir,
                arc_config=arc_config,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=partial(collate_fn, shuffle_options=False),
            )
            print(f"Test Samples  : {len(test_ds)}")

    print(f"Benchmark     : {arc_config}")
    print(f"Train Samples : {len(train_ds)}")
    print(f"Val Samples   : {len(val_ds)}")

    return train_loader, val_loader, test_loader


def run_training(
    cache_dir=CACHE_DIR,
    ckpt_dir=CKPT_DIR,
    train_samples=None,
    val_samples=None,
    epochs=EPOCHS,
    patience=PATIENCE,
    persistent_steps=PERSISTENT_STEPS,
    n_qubits=N_QUBITS,
    seed=SEED,
    name="qair",
    with_test=True,
    arc_config=ARC_CONFIG,
    phase_mode=PHASE_MODE,
    phase_source=PHASE_SOURCE,
    mixing=MIXING,
    use_llm_prior=USE_LLM_PRIOR,
    use_attack=True,
    deterministic_warn_only=DETERMINISTIC_WARN_ONLY,
    ignore_pin_mismatch=False,
):
    """
    name is the arm label; the checkpoint key is run_key(name, arc_config,
    seed), so a second seed or dataset never resumes the first.
    """

    run_name = run_key(name, arc_config, seed)

    # Before anything else: a mismatched environment must not train.
    pin_check = check_pins(override=ignore_pin_mismatch)

    print("=" * 60)
    print(f"qAIR-v45.1 TRAINING  ({arc_config})  run {run_name}")
    print("=" * 60)

    set_seed(seed, deterministic=DETERMINISTIC, warn_only=deterministic_warn_only)

    train_loader, val_loader, test_loader = build_loaders(
        cache_dir, train_samples, val_samples, seed=seed,
        with_test=with_test, arc_config=arc_config,
    )

    # Also saved as {run_name}_config.pt so inference can't silently load
    # with mismatched persistent_steps / n_qubits / phase_mode ever again.
    model_config = {
        "use_quantum": True,
        "use_validator": True,
        "persistent_steps": persistent_steps,
        "n_qubits": n_qubits,
        "use_question": True,
        "model_dim": MODEL_DIM,
        "phase_mode": phase_mode,
        "phase_source": phase_source,
        "mixing": mixing,
        "use_llm_prior": use_llm_prior,
        "use_attack": use_attack,
    }

    model = QAIRvNext(dim=DIM, verbose=True, **model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"Parameters    : {n_params:,}")

    provenance = run_provenance(
        arm=name,
        arc_config=arc_config,
        seed=seed,
        model_config=model_config,
        train_config=training_config(
            epochs, patience, WEIGHT_DECAY, train_samples, val_samples,
            deterministic_warn_only,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        pin_check=pin_check,
    )

    print(describe(provenance))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        ckpt_dir=ckpt_dir,
        name=run_name,
        provenance=provenance,
        weight_decay=WEIGHT_DECAY,
    )

    write_package_list(os.path.join(ckpt_dir, f"{run_name}_env.txt"))

    state = load_or_resume(trainer, ckpt_dir, run_name, epochs, provenance)

    history = trainer.train_from(state, epochs, patience=patience)

    history["provenance"] = provenance

    config_path = os.path.join(ckpt_dir, f"{run_name}_config.pt")
    torch.save(
        {**model_config, "arc_config": arc_config, "seed": seed}, config_path,
    )
    print(f"[CONFIG SAVED] {config_path}")

    # ----------------------------------------------------------
    # HELD-OUT TEST -- reported once, from the best checkpoint,
    # on data that took no part in model selection.
    # ----------------------------------------------------------

    if test_loader is not None:

        load_best(model, ckpt_dir, run_name, provenance, device)

        test_metrics, records = evaluate(
            model, test_loader, device, return_records=True
        )

        history["test"] = test_metrics
        history["test_records"] = records

        trainer.save_history(history)

        print("\n" + "=" * 60)
        print(f"[{run_name}] HELD-OUT TEST acc = {test_metrics['acc']:.4f}  "
              f"pairwise_cos = {test_metrics['pairwise_cos']:.4f}  "
              f"interference = {test_metrics['interference_ratio']:.4f}")
        print("=" * 60)

    print(f"[{run_name}] {describe(provenance)}")

    return history
