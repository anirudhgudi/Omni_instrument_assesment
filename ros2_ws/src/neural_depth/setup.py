from setuptools import find_packages, setup

package_name = 'neural_depth'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anirudh',
    maintainer_email='anirudh@todo.todo',
    description='Neural stereo depth estimation using RAFT-Stereo',
    license='MIT',
    entry_points={
        'console_scripts': [
            'neural_depth_node = neural_depth.neural_depth_node:main',
            'hitnet_node = neural_depth.hitnet_node:main',
        ],
    },
)
