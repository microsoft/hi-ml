# HI-ML Multimodal Toolbox

This toolbox provides models for multimodal health data.
The code is available on [GitHub][1] and [Hugging Face 🤗][6].

## Getting started

The best way to get started is by running the [phrase grounding notebook][2] and the [examples](#examples).
All the dependencies will be installed upon execution, so Python 3.7 and [Jupyter][3] are the only requirements to get started.

The notebook can also be run on [Binder][4], without the need to download any code or install any libraries:

[![Binder](https://mybinder.org/badge_logo.svg)][4]

## Installation

The latest version can be installed using `pip`:

```console
pip install --upgrade hi-ml-multimodal
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

For zero-shot classification of images using text prompts, please refer to the [example script](./test_multimodal/vlp/test_zero_shot_classification.py) that utilises a small subset of [Open-Indiana CXR
dataset][10] for pneumonia detection in chest X-ray images.
Please note that the examples and models are not intended for deployed use cases (commercial or otherwise), which is currently out-of-scope.

## Hugging Face 🤗

While the [GitHub repository][1] provides examples and pipelines to use our models,
the weights and model cards are hosted on [Hugging Face 🤗][6].

## Credit

If you use our code or models in your research, please cite our recent ECCV and CVPR papers:

> Boecking, B., Usuyama, N. et al. (2022). [Making the Most of Text Semantics to Improve Biomedical Vision–Language Processing][7]. In: Avidan, S., Brostow, G., Cissé, M., Farinella, G.M., Hassner, T. (eds) Computer Vision – ECCV 2022. ECCV 2022. Lecture Notes in Computer Science, vol 13696. Springer, Cham. [https://doi.org/10.1007/978-3-031-20059-5_1][7]

> Bannur, S., Hyland, S., et al. (2023). [Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing][8]. In: CVPR 2023.

### BibTeX

```bibtex
@InProceedings{10.1007/978-3-031-20059-5_1,
    author="Boecking, Benedikt and Usuyama, Naoto and Bannur, Shruthi and Castro, Daniel C. and Schwaighofer, Anton and Hyland, Stephanie and Wetscherek, Maria and Naumann, Tristan and Nori, Aditya and Alvarez-Valle, Javier and Poon, Hoifung and Oktay, Ozan",
    editor="Avidan, Shai and Brostow, Gabriel and Ciss{\'e}, Moustapha and Farinella, Giovanni Maria and Hassner, Tal",
    title="Making the Most of Text Semantics to Improve Biomedical Vision--Language Processing",
    booktitle="Computer Vision -- ECCV 2022",
    year="2022",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="1--21",
    isbn="978-3-031-20059-5"
}

@inproceedings{bannur2023learning,
    title={Learning to Exploit Temporal Structure for Biomedical Vision{\textendash}Language Processing},
    author={Shruthi Bannur and Stephanie Hyland and Qianchu Liu and Fernando P\'{e}rez-Garc\'{i}a and Maximilian Ilse and Daniel C. Castro and Benedikt Boecking and Harshita Sharma and Kenza Bouzid and Anja Thieme and Anton Schwaighofer and Maria Wetscherek and Matthew P. Lungren and Aditya Nori and Javier Alvarez-Valle and Ozan Oktay},
    booktitle={Conference on Computer Vision and Pattern Recognition 2023},
    year={2023},
    url={https://openreview.net/forum?id=5jScn5xsbo}
}
```

[1]: https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal
[2]: https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal/notebooks/phrase_grounding.ipynb
[3]: https://jupyter.org/
[4]: https://mybinder.org/v2/gh/microsoft/hi-ml/HEAD?labpath=hi-ml-multimodal%2Fnotebooks%2Fphrase_grounding.ipynb
[5]: https://docs.conda.io/en/latest/miniconda.html
[6]: https://aka.ms/biovil-models
[7]: https://doi.org/10.1007/978-3-031-20059-5_1
[8]: https://arxiv.org/abs/2301.04558
[9]: https://hi-ml.readthedocs.io/en/latest/api/multimodal.html
[10]: https://openi.nlm.nih.gov/faq
