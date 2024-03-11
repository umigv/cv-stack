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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arnav Shah',
    maintainer_email='kanishkkandoi52@gmail.com',
    description='TODO: Package description',
    license='Appache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drivable_area_test = drivable_area.drivable_area:main',
            'drivable_area_test_publisher = drivable_area.drivable_area_test_publisher:main',
            'drivable_area_test_subscriber = drivable_area.drivable_area_test_subscriber:main',
        ],
    },
)
