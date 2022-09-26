# HI-ML Multimodal Toolbox

This toolbox provides models for working with multi-modal health data.
The code is available on [GitHub][1] and [Hugging Face 🤗][6].

## Getting started

The best way to get started is by running the [phrase grounding notebook][2].
All the dependencies will be installed upon execution, so Python 3.7 and [Jupyter][3] are the only requirements to get started.

The notebook can also be run on [Binder][4], without the need to download any code or install any libraries:

[![Binder](https://mybinder.org/badge_logo.svg)][4]

## Installation

The latest version can be installed using `pip`:

```console
pip install "git+https://github.com/microsoft/hi-ml.git#subdirectory=hi-ml-multimodal"
```

### Development

For development, it is recommended to clone the repository and set up the environment using [`conda`][5]:

```console
git clone https://github.com/microsoft/hi-ml.git
cd hi-ml-multimodal
make env
```

This will create a `conda` environment named `multimodal` and install all the dependencies to run and test the package.

You can visit the [API documentation][9] for a deeper understanding of our tools.

## Examples

For zero-shot classification of images using text prompts, please refer to the [example
script](./test_multimodal/vlp/test_zero_shot_classification.py) that utilises a small subset of [Open-Indiana CXR
dataset][10] for pneumonia detection in Chest X-ray images. Please note that the examples and models are not intended for
deployed use cases -- commercial or otherwise -- which is currently out-of-scope.

## Hugging Face 🤗

While the [GitHub repository][1] provides examples and pipelines to use our models,
the weights and model cards are hosted on [Hugging Face 🤗][6].

## Credit

If you use our code or models in your research, please cite [the manuscript][7] (accepted to be presented at the [**European Conference on Computer Vision (ECCV) 2022**][8]).

### APA

> Boecking, B., Usuyama, N., Bannur, S., Castro, D., Schwaighofer, A., Hyland, S., Wetscherek, M., Naumann, T., Nori, A., Alvarez-Valle, J., Poon, H., & Oktay, O. (2022). *Making the Most of Text Semantics to Improve Biomedical Vision–Language Processing* ([preprint][7])

### BibTeX

```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.09817,
  doi = {10.48550/ARXIV.2204.09817},
  url = {https://arxiv.org/abs/2204.09817},
  author = {Boecking, Benedikt and Usuyama, Naoto and Bannur, Shruthi and Castro, Daniel C. and Schwaighofer, Anton and Hyland, Stephanie and Wetscherek, Maria and Naumann, Tristan and Nori, Aditya and Alvarez-Valle, Javier and Poon, Hoifung and Oktay, Ozan},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing},
  publisher = {arXiv},
  year = {2022},
}
```

[1]: https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal
[2]: https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal/notebooks/phrase_grounding.ipynb
[3]: https://jupyter.org/
[4]: https://mybinder.org/v2/gh/microsoft/hi-ml/HEAD?labpath=hi-ml-multimodal%2Fnotebooks%2Fphrase_grounding.ipynb
[5]: https://docs.conda.io/en/latest/miniconda.html
[6]: https://aka.ms/biovil-models
[7]: https://arxiv.org/abs/2204.09817
[8]: https://eccv2022.ecva.net/
[9]: https://hi-ml.readthedocs.io/en/latest/api/multimodal.html
[10]: https://openi.nlm.nih.gov/faq
