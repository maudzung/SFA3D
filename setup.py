from setuptools import setup, find_packages

package_name = 'super_fast_object_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),  # This will auto-include sfa too
    data_files=[('share/' + package_name, ['package.xml'])],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Super Fast 3D Object Detector using SFA',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'sfa_node = super_fast_object_detection.sfa_node:main',
        ],
    },
)
