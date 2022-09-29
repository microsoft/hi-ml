# Example Trainer Script for Amulet

The following example shows a simple PyTorch Lightning trainer script that makes use of the Amulet environment.

You will need a folder that contains the following files:

- A Conda environment definition, like [this](amulet/environment.rst).
- An Amulet configuration file, like [this](amulet/amulet_config.rst). The configuration file should live in
  your repository's root folder, and to the Conda environment definition.
- A script that uses the Amulet environment, like [this](amulet/amulet_script.rst)

The script has a large number of comments around the correct use of the Amulet environment - please read them
carefully if you want to base your training code on that example.

To submit this example script as-is via Amulet, follow these steps:

- Check out the [hi-ml](https://github.com/microsoft/hi-ml) repository.
- Follow the onboarding instructions in the [Amulet overview](amulet_overview.md) section, and create an Amulet
  project in the root folder of the repository.
- Modify `docs/source/amulet/config.yml` to point to your storage account: replace the `<storage_account_name>`
  placeholder with the name of your storage account.
- Once Amulet is installed, submit the example jobs with the following command:

```bash
amlt run docs/source/amulet/config.yml <experiment_name> -t <name_of_your_compute_target>
```
