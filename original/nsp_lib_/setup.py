"""
NetSurfP2

"""

from setuptools import setup


setup(
    name='netsurfp2_dev',
    version='2.0',
    url=None,
    # license='Proprietary',
    # author='Michael Schantz Klausen',
    # author_email='',
    #description='Something something IgG.class',
    #long_description=__doc__,
    packages=['netsurfp2_dev', ],
    #zip_safe=False,
    platforms='any',
    install_requires=[
        # 'numpy>=1.11',
    ],
    #entry_points = {'console_scripts': [
    #    'lyra_model = bcr_models.__main__:entry',
    #    'lyra_pdbdir2tpldb = bcr_models.scripts.pdbdir2tpldb:main',
    #    'lyra_bench_db = bcr_models.scripts.benchmark_db:main',
    #    'lyra_bench_dir = bcr_models.scripts.benchmark_dir:main',
    #]},
    #include_package_data=True,
    # package_data={'netsurfp2': ['model_data/*.*']},
)
