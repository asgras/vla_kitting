#!/usr/bin/env python3
"""
train_octo.py — Fine-tune Octo-Small on kitting demonstration data.

Runs outside ROS2. Requires: pip install octo-model jax jaxlib tensorflow tensorflow_datasets

Usage:
  python3 scripts/train_octo.py --config config/train_config.yaml
  python3 scripts/train_octo.py --config config/train_config.yaml --data_dir ./data/kitting_demos
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config: dict, data_dir: str | None = None, output_dir: str = './checkpoints'):
    """Run Octo-Small fine-tuning."""
    import jax
    import jax.numpy as jnp
    import optax
    import tensorflow_datasets as tfds
    from octo.model.octo_model import OctoModel
    from octo.data.dataset import make_single_dataset

    print(f"JAX devices: {jax.devices()}")
    print(f"Loading base model: {config['model']['base']}")

    # Load pretrained Octo-Small
    model = OctoModel.load_pretrained(config['model']['base'])

    # Update config for our action space
    model_config = model.config.copy()
    action_dim = config['model']['action_dim']
    pred_horizon = config['model']['pred_horizon']
    model_config['model']['heads']['action']['kwargs']['action_dim'] = action_dim
    model_config['model']['heads']['action']['kwargs']['pred_horizon'] = pred_horizon

    # Build dataset
    dataset_path = data_dir or config['data']['dataset_path']
    print(f"Loading dataset from: {dataset_path}")

    dataset_config = {
        'name': 'kitting_demos',
        'data_dir': dataset_path,
        'image_obs_keys': {'primary': 'image_primary'},
        'proprio_obs_key': 'proprio',
        'language_key': 'language_instruction',
        'action_proprio_normalization_type': config['data']['action_normalization'],
    }

    train_data = make_single_dataset(
        dataset_config,
        traj_transform_kwargs={
            'window_size': config['data']['window_size'],
            'action_horizon': pred_horizon,
        },
        frame_transform_kwargs={
            'resize_size': {'primary': tuple(config['data']['image_size'])},
        },
        train=True,
    )

    # Create optimizer
    lr = config['training']['learning_rate']
    warmup = config['training']['warmup_steps']
    max_steps = config['training']['max_steps']
    wd = config['training']['weight_decay']

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=max_steps,
        end_value=lr * 0.01,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=wd)

    # Initialize training
    rng = jax.random.PRNGKey(config['training'].get('seed', 42))
    batch_size = config['training']['batch_size']
    save_every = config['training']['save_every']
    eval_every = config['training']['eval_every']

    os.makedirs(output_dir, exist_ok=True)
    train_iter = train_data.batch(batch_size).as_numpy_iterator()

    # Training state
    params = model.params
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch, rng):
        def loss_fn(params):
            return model.loss(params, batch, rng=rng)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    print(f"Starting training for {max_steps} steps...")
    metrics_log = []
    start_time = time.time()

    for step in range(max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = train_data.batch(batch_size).as_numpy_iterator()
            batch = next(train_iter)

        rng, step_rng = jax.random.split(rng)
        params, opt_state, loss = train_step(params, opt_state, batch, step_rng)

        if step % 100 == 0:
            loss_val = float(loss)
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            eta_hrs = (max_steps - step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0
            print(f"Step {step:6d}/{max_steps} | "
                  f"Loss: {loss_val:.4f} | "
                  f"Steps/s: {steps_per_sec:.1f} | "
                  f"ETA: {eta_hrs:.1f}h")
            metrics_log.append({
                'step': step, 'loss': loss_val,
                'elapsed_s': elapsed})

        if step > 0 and step % save_every == 0:
            ckpt_path = os.path.join(output_dir, f'step_{step:06d}')
            model.save_pretrained(ckpt_path, params=params)
            print(f"Checkpoint saved: {ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(output_dir, 'kitting_octo_small')
    model.save_pretrained(final_path, params=params)
    print(f"Final model saved: {final_path}")

    # Save training metrics
    metrics_path = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time/3600:.1f} hours")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Octo-Small on kitting data')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override dataset directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
