"""
episode_manager_node.py — ROS2 wrapper for the EpisodeManager.

Standalone node that can be used to orchestrate VLA inference
by publishing kit orders and monitoring episode completion.
Useful for automated evaluation runs.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node

from kitting_interfaces.msg import KitOrder
from std_msgs.msg import String


def _make_node():
    class EpisodeManagerNode(Node):
        def __init__(self):
            super().__init__('episode_manager_node')

            self.declare_parameter('total_picks', 9)
            self.declare_parameter('sku', 'single_gang_box')
            self.declare_parameter('auto_start', False)

            self._total_picks = self.get_parameter('total_picks').value
            self._sku = self.get_parameter('sku').value
            auto_start = self.get_parameter('auto_start').value

            # Publisher for kit orders
            self._order_pub = self.create_publisher(
                KitOrder, '/kit_order', 10)

            # Status publisher
            self._status_pub = self.create_publisher(
                String, '/vla/episode_status', 10)

            if auto_start:
                # Delay start to let other nodes initialize
                self.create_timer(5.0, self._auto_start, callback_group=None)

            self.get_logger().info(
                f'Episode manager ready (sku={self._sku}, '
                f'picks={self._total_picks})')

        def _auto_start(self):
            """Send initial kit order after startup delay."""
            self._send_order(self._sku, self._total_picks)
            # Cancel the timer after first fire
            self.destroy_timer(self._timer)

        def _send_order(self, sku: str, quantity: int):
            msg = KitOrder()
            msg.sku_ids = [sku]
            msg.quantities = [quantity]
            self._order_pub.publish(msg)
            self.get_logger().info(
                f'Published kit order: {sku} x {quantity}')

    return EpisodeManagerNode()


def main(args=None):
    rclpy.init(args=args)
    node = _make_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
