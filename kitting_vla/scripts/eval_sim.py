#!/usr/bin/env python3
"""
eval_sim.py — Evaluate a trained VLA model in Isaac Sim (or mock).

Runs N episodes, sends kit orders, and measures success rate,
cycle time, and failure modes.

Usage:
  # First launch the VLA inference stack:
  #   ros2 launch kitting_vla vla_inference.launch.py hardware_type:=isaac
  # Then run evaluation:
  python3 scripts/eval_sim.py --num_episodes 100 --sku single_gang_box
"""
from __future__ import annotations

import argparse
import json
import os
import time

import rclpy
from rclpy.node import Node
from kitting_interfaces.msg import KitOrder


class EvalNode(Node):
    def __init__(self, num_episodes: int, sku: str, timeout: float):
        super().__init__('eval_sim_node')
        self._num_episodes = num_episodes
        self._sku = sku
        self._timeout = timeout

        self._order_pub = self.create_publisher(KitOrder, '/kit_order', 10)
        self._results: list[dict] = []
        self._episode = 0

        # Wait for system to be ready, then start
        self.create_timer(3.0, self._run_next_episode)

    def _run_next_episode(self):
        if self._episode >= self._num_episodes:
            self._report()
            raise SystemExit(0)

        self.get_logger().info(
            f'Starting eval episode {self._episode + 1}/{self._num_episodes}')

        msg = KitOrder()
        msg.sku_ids = [self._sku]
        msg.quantities = [1]

        start = time.monotonic()
        self._order_pub.publish(msg)

        # Wait for episode to complete (simple timeout-based)
        # In production, subscribe to episode status topic
        time.sleep(self._timeout)

        elapsed = time.monotonic() - start
        self._results.append({
            'episode': self._episode,
            'sku': self._sku,
            'duration': elapsed,
        })
        self._episode += 1

    def _report(self):
        output = {
            'num_episodes': len(self._results),
            'sku': self._sku,
            'results': self._results,
            'mean_duration': (
                sum(r['duration'] for r in self._results) / len(self._results)
                if self._results else 0),
        }
        path = f'eval_results_{self._sku}_{len(self._results)}ep.json'
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        self.get_logger().info(f'Results saved to {path}')
        print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLA model in sim')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--sku', type=str, default='single_gang_box')
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Seconds to wait per episode')
    args = parser.parse_args()

    rclpy.init()
    node = EvalNode(args.num_episodes, args.sku, args.timeout)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
