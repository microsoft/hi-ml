.. hi-ml documentation master file, created by
   sphinx-quickstart on Wed Sep  1 15:24:01 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for the Health Intelligence Machine Learning toolbox hi-ml
========================================================================

This toolbox helps to simplify and streamline work on deep learning models for healthcare and life sciences,
by providing tested components (data loaders, pre-processing), deep learning models, and cloud integration tools.

The `hi-ml` toolbox provides

- Functionality to easily run Python code in Azure Machine Learning services
- Low-level and high-level building blocks for Machine Learning / AI researchers and practitioners.

.. toctree::
   :maxdepth: 1
   :caption: Working with Azure

   first_steps.md
   azure_setup.md
   authentication.md
   datasets.md
   hyperparameter_search.md
   lowpriority.md
   commandline_tools.md
   downloading.md
   examples.md

.. toctree::
   :maxdepth: 1
   :caption: Working with Amulet

   amulet_overview.md
   amulet_example.rst

.. toctree::
   :maxdepth: 1
   :caption: Machine Learning

   logging.md
   diagnostics.md
   runner.md
   checkpoints.md

.. toctree::
   :maxdepth: 1
   :caption: Histopathology

   histopathology.md
   public_datasets.md
   panda_model.md
   tcga_model.md
   ssl_on_tile_dataset.md
   dsa.md
   montage_creation.md
   cpath_scripts

.. toctree::
   :maxdepth: 1
   :caption: Multimodal learning

   multimodal.md
   api/multimodal.rst

.. toctree::
   :maxdepth: 1
   :caption: Self supervised learning

   self_supervised_models.md

.. toctree::
   :maxdepth: 1
   :caption: Developers

   CONTRIBUTING.md
   coding_guidelines.md
   developers.md
   software_design_overview.md
   debugging.md

.. toctree::
   :maxdepth: 1
   :caption: Guidelines

   loading_images.md
   whole_slide_images.md

.. toctree::
   :maxdepth: 1
   :caption: Changelog

   CHANGELOG.md

.. toctree::
   :maxdepth: 2
   :caption: API

   api/api.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
