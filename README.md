# Introduction to Modulus-UW
Modulus-uw is a fork of [`NVIDIA/modulus`](https://github.com/NVIDIA/modulus) maintained by NVIDIA and the Department of Atmospheric and Climate Sciences at the University of Washington. Its purpose is to facilitate efficient development of deep learning weather and climate models. 

# Collaboration guidelines
The following points outline development practices of `modulus-uw` and describe permanent branches:
- Before contributing to modulus-uw please review `NVIDIA/modulus`'s [contribution guide](https://github.com/NVIDIA/modulus/blob/main/CONTRIBUTING.md). The goal of developments within modulus-uw is to be ultimately published in `NVIDIA/modulus`. As such, we defer to their standards. 
- `modulus-uw/dev` features new models, couplers, data modules, and other essential building blocks for our weather and climate models. Contributions to this branch require a pull request and code review. Feature development should be done in branches off `modulus-uw/dev`.
- `modulus-uw/main` remains an up-to-date copy of `NVIDIA/modulus`. When a new feature is finalized in `modulus-uw/dev`, rebase on `modulus-uw/main` before submitting a pull request to `NVIDIA/modulus`. This will facilitate cleaner PRs. 
- Finally, `modulus-uw/docs` will host actively maintained documentation of `modulus-uw`. This branch is purely a place for docs, and as such does not contain up-to-date code. It is a stand-alone branch and will not be merged with other branches.  

# Helpful Schematics
<!-- markdownlint-disable -->
<p align="center">
  <img src=docs/img/UW/FlowChart.png alt="Modulus"/>
</p>