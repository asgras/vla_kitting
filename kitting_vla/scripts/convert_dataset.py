#!/usr/bin/env python3
"""
convert_dataset.py — Convert raw HDF5 episodes to RLDS format (Octo native).

Reads episode_XXXXXX.hdf5 files, computes normalization statistics,
and writes TensorFlow RLDS dataset that Octo can directly consume.

Usage:
  python3 scripts/convert_dataset.py --input_dir ./data/raw --output_dir ./data/kitting_demos
  python3 scripts/convert_dataset.py --input_dir ./data/raw --output_dir ./data/kitting_demos --val_ratio 0.05
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load_episodes_hdf5(input_dir: str) -> list[dict]:
    """Load all HDF5 episodes from a directory."""
    try:
        import h5py
    except ImportError:
        return load_episodes_npz(input_dir)

    episodes = []
    for path in sorted(Path(input_dir).glob('episode_*.hdf5')):
        with h5py.File(path, 'r') as f:
            ep = {
                'image': np.array(f['image']),
                'state': np.array(f['state']),
                'action': np.array(f['action']),
                'language': str(f.attrs['language']),
                'success': bool(f.attrs['success']),
            }
            episodes.append(ep)
    return episodes


def load_episodes_npz(input_dir: str) -> list[dict]:
    """Load all NPZ episodes from a directory."""
    episodes = []
    for path in sorted(Path(input_dir).glob('episode_*.npz')):
        data = np.load(path, allow_pickle=True)
        ep = {
            'image': data['image'],
            'state': data['state'],
            'action': data['action'],
            'language': str(data['language']),
            'success': bool(data['success']),
        }
        episodes.append(ep)
    return episodes


def compute_statistics(episodes: list[dict]) -> dict:
    """Compute normalization statistics over all episodes."""
    all_actions = np.concatenate([ep['action'] for ep in episodes])
    all_states = np.concatenate([ep['state'] for ep in episodes])

    return {
        'action_mean': all_actions.mean(axis=0).tolist(),
        'action_std': all_actions.std(axis=0).tolist(),
        'state_mean': all_states.mean(axis=0).tolist(),
        'state_std': all_states.std(axis=0).tolist(),
        'num_episodes': len(episodes),
        'total_timesteps': sum(len(ep['action']) for ep in episodes),
    }


def write_rlds(episodes: list[dict], output_dir: str, split: str = 'train'):
    """Write episodes in RLDS-compatible TFRecord format."""
    import tensorflow as tf

    os.makedirs(output_dir, exist_ok=True)

    # Determine number of shards
    num_episodes = len(episodes)
    num_shards = max(1, min(8, num_episodes // 100))

    for shard_idx in range(num_shards):
        shard_episodes = episodes[shard_idx::num_shards]
        tfrecord_path = os.path.join(
            output_dir,
            f'kitting_demos-{split}.tfrecord-{shard_idx:05d}-of-{num_shards:05d}')

        writer = tf.io.TFRecordWriter(tfrecord_path)

        for ep in shard_episodes:
            n_steps = len(ep['action'])
            for t in range(n_steps):
                is_last = (t == n_steps - 1)
                step_features = {
                    'observation/image_primary': tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[ep['image'][t].tobytes()])),
                    'observation/image_primary/shape': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=list(ep['image'][t].shape))),
                    'observation/proprio': tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=ep['state'][t].tolist())),
                    'action': tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=ep['action'][t].tolist())),
                    'language_instruction': tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[ep['language'].encode('utf-8')])),
                    'is_terminal': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[1 if is_last else 0])),
                    'is_first': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[1 if t == 0 else 0])),
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=step_features))
                writer.write(example.SerializeToString())

        writer.close()
        print(f"Wrote shard {shard_idx}: {len(shard_episodes)} episodes")

    # Write dataset info
    info = {
        'name': 'kitting_demos',
        'version': '1.0.0',
        'split': split,
        'num_episodes': num_episodes,
        'features': {
            'observation/image_primary': {'shape': [256, 256, 3], 'dtype': 'uint8'},
            'observation/proprio': {'shape': [7], 'dtype': 'float32'},
            'action': {'shape': [7], 'dtype': 'float32'},
            'language_instruction': {'dtype': 'string'},
            'is_terminal': {'dtype': 'bool'},
            'is_first': {'dtype': 'bool'},
        },
    }
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Convert raw HDF5 episodes to RLDS format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory with episode_*.hdf5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for RLDS dataset')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='Fraction of episodes for validation split')
    parser.add_argument('--success_only', action='store_true',
                        help='Only include successful episodes')
    args = parser.parse_args()

    print(f"Loading episodes from {args.input_dir}...")
    episodes = load_episodes_hdf5(args.input_dir)
    print(f"Loaded {len(episodes)} episodes")

    if args.success_only:
        episodes = [ep for ep in episodes if ep['success']]
        print(f"After filtering: {len(episodes)} successful episodes")

    if len(episodes) == 0:
        print("No episodes found!")
        return

    # Compute and save statistics
    stats = compute_statistics(episodes)
    print(f"Dataset stats: {stats['num_episodes']} episodes, "
          f"{stats['total_timesteps']} timesteps")

    os.makedirs(args.output_dir, exist_ok=True)
    stats_path = os.path.join(args.output_dir, 'normalization_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {stats_path}")

    # Split into train/val
    np.random.seed(42)
    indices = np.random.permutation(len(episodes))
    val_count = max(1, int(len(episodes) * args.val_ratio))

    val_episodes = [episodes[i] for i in indices[:val_count]]
    train_episodes = [episodes[i] for i in indices[val_count:]]

    print(f"Split: {len(train_episodes)} train, {len(val_episodes)} val")

    # Write RLDS
    write_rlds(train_episodes, args.output_dir, split='train')
    write_rlds(val_episodes, args.output_dir, split='val')

    print("Conversion complete!")


if __name__ == '__main__':
    main()
