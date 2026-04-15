from setuptools import find_packages, setup

package_name = 'stereo_depth'

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
    description='Stereo depth estimation using OpenCV StereoSGBM with WLS post-filtering',
    license='MIT',
    entry_points={
        'console_scripts': [
            'stereo_depth_node = stereo_depth.stereo_depth_node:main',
        ],
    },
)
