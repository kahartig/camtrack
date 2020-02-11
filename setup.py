from setuptools import setup

def readme():
	with open('README.md') as f:
		return f.read()

setup(name='camtrack',
	version='0.1',
	description='Workflow for running HYSPLIT trajectories on cold air outbreaks in Arctic winter from Community Atmosphere Model, CAM4',
	long_description=readme(),
	url='https://github.com/kahartig/camtrack.git',
	author='Kara Hartig',
	author_email='kara_hartig@g.harvard.edu',
	license='GNUv3',
	packages=['camtrack'],
	scripts=['bin/concat_winter', 'bin/concat_analysis_variables', 'bin/concat_all_winters'],
	install_requires=['numpy', 'pandas', 'matplotlib', 'xarray', 'netCDF4', 'cftime', 'cartopy', 'calendar'],
	include_package_data=True,
	zip_safe=False,
	test_suite='nose.collector',
	tests_require=['nose'])
