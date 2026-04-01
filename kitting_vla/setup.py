import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'kitting_vla'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='VLA-based pick/place for the HC10DT kitting cell',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_inference_node = kitting_vla.vla_inference_node:main',
            'data_collector_node = kitting_vla.data_collector_node:main',
            'episode_manager_node = kitting_vla.episode_manager_node:main',
        ],
    },
)
