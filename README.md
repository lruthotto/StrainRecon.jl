# StrainRecon.jl

[![Build Status](https://travis-ci.org/lruthotto/StrainRecon.jl.svg?branch=master)](https://travis-ci.org/lruthotto/StrainRecon.jl)
[![Coverage Status](https://coveralls.io/repos/github/lruthotto/StrainRecon.jl/badge.svg?branch=master)](https://coveralls.io/github/lruthotto/StrainRecon.jl?branch=master)

Julia code for strain reconstruction from mixed samples. 
The problem is formulating as a Bayesian inverse problem involving binary constraints. 
The code contains methods for building the posterior and exploring it by MAP estimation and some uncertainty quantification.  

The code is described in detail in:

```
@article{mustonen2018bayesian,
  title={A Bayesian framework for molecular strain identification from mixed diagnostic samples},
  author={Mustonen, Lauri and Gao, Xiangxi and Santana, Asteroide and Mitchell, Rebecca and Vigfusson, Ymir and Ruthotto, Lars},
  journal={arXiv preprint arXiv:1803.02916},
  year={2018}
}
```

## Getting started
To use the code type:

```
Pkg.clone("https://github.com/lruthotto/StrainRecon.jl")
Pkg.test("StrainRecon")
```

Some advanced methods require the installation of other packages. For full functionality type 

```
Pkg.clone("https://github.com/JuliaInv/jInv.jl")
Pkg.clone("https://github.com/JuliaInv/jInvVis.jl")
Pkg.add("JuMP")
Pkg.add("Gurobi")
```

## Acknowledgements
This material is in part based upon work supported by the National Science Foundation under Grant Number 1522599 and by the Advanced Molecular Detection fund from the CDC under contract CNS 1553579. 
Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of any of the funding agencies.
