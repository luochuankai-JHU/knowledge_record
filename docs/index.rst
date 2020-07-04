.. m2g_data documentation master file, created by
   sphinx-quickstart on Tue Mar 24 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


******************
m2g
******************



Understanding the variability in brain connectivity is essential to understanding the human brain. Currently brains are commonly modeled as a connectome by modeling brain regions as nodes and connections as edges. Currently, there is a gap between the raw data, diffusion magnetic resonance imaging (dMRI) files , and actual connectomes, which can be complex and variable in development and usage.

The m2g python package, from our Johns Hopkins Open Connectome Project, is the first end to end MRI connectome estimation pipeline that can be run and has been tested in large MRI datasets. Using this package we have generated a large set of standardized connectomes () using publically available, single shell dMRI data. This large amount of connectomes is available at ___ can be directly used by researchers to utilize. 

.. toctree::
   :caption: Tutorials
   :maxdepth: 2
   
   tutorials/install
   tutorials/funcref

.. toctree::
   :caption: Pipeline Overview
   :maxdepth: 2

   pipeline/diffusion
   pipeline/functional

.. toctree::
   :maxdepth: 1
   :caption: Datasets
   
   datasets/CoRR/CoRR
   datasets/openfMRI/sssss
   datasets/KKI/KKI2009
   datasets/NKIENH/NKIENH
   
.. toctree::
   :maxdepth: 1
   :caption: License
   
   License

.. toctree::
   :maxdepth: 1
   :caption: Useful Links

   m2g @ GitHub <https://github.com/neurodata/m2g>
   m2g @ PyPI <https://pypi.org/project/m2g/>
   Issue Tracker <https://github.com/neurodata/m2g/issues>
   
