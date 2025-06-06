import os
from glob import glob
from setuptools import find_packages, setup


package_name = 'drivable_area'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arnav Shah',
    maintainer_email='arnshah@umich.edu',
    description='TODO: Package description',
    license='Appache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drivable_area_head = drivable_area.test_sensor:main',
            'drivable_area_test = drivable_area.drivable_area:main',
            'drivable_area_mag = drivable_area.test_sensor:main',
            'drivable_area_self = drivable_area.drivable_area_self_drive:main',
            'drivable_area_test_publisher = drivable_area.drivable_area_test_publisher:main',
            'drivable_area_test_subscriber = drivable_area.drivable_area_test_subscriber:main',
            'drivable_area_mqtt = drivable_area.driveable_area_mqtt:main',
        ],
    },
)
