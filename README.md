<p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/dev/peps/"><img src="https://img.shields.io/badge/code_style-standard-brightgreen.svg" alt="Standard - Python Style Guide"></a>
    <a href="http://opensources.co"><img src="https://img.shields.io/badge/Data-OpenSources-blue.svg" alt="OpenSources Data"></a>
    <a href="https://docs.python.org/3/"><img src="https://img.shields.io/badge/python-3.5%7C3.6%7C3.7-blue.svg" alt="Python Version"></a>
</p>


# mugshots

The following papers are used for reference:

| :------------: | :------------------------------: |
| Image-to-Image | https://arxiv.org/abs/1611.07004 |
| U-Net          | https://arxiv.org/abs/1505.04597 |
| Noise2Noise    | https://arxiv.org/abs/1803.04189 |

For this project I used the [mugshots](https://www.nist.gov/srd/nist-special-database-18) dataset from the National Institute of Standards and Technology (NIST).


![generated mugshot](images/demo_profile.jpg)


The main idea behind the "mugshots" project is to be able to generate a side view from a given image showing the front view of a face. I know what you are probably thinking, "what are you thinking, that's not possible."  Nonetheless, after reading the Image-to-Image paper and U-Net paper and some others covering the topic of a generative adversarial network (GAN), I thought it would be a fun project to try it out -- anyway. The GAN model I implemented is based on these papers with some alterations.
