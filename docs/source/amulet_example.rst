Example for Scripts that are submitted via Amulet
=================================================

The following example shows a simple PyTorch Lightning trainer script that makes use of the Amulet environment.

You will need a folder that contains the following files:

- A Conda environment definition, like :download:`this environment <amulet/environment.yml>`.
- An Amulet configuration file, like :download:`this configuration <amulet/config.yml>`.
- A script that uses the Amulet environment, like :download:`this script <amulet/amulet_example.py>`.

The script has a large number of comments around the correct use of the Amulet environment - please read them
carefully if you want to base your training code on that.

To submit this script via Amulet, follow the onboarding instructions in the :ref:`amulet_overview` section.
Once Amulet is installed, submit the example jobs with the following commands:

.. code-block:: bash

    amlt run config.yml <experiment_name> -t <name_of_your_compute_target>
