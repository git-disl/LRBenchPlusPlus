# LRBench++: A framework for effective learning rate tuning and benchmarking

<!--- a href=""><img src="" alt=""></a --->
<!-- -----------------
[![GitHub license](https://img.shields.io/badge/license-apache-green.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)
[![Version](https://img.shields.io/badge/version-0.0.1-red.svg?style=flat)]() -->
<!---
[![Travis Status]()]()
[![Jenkins Status]()]()
[![Coverage Status]()]()
--->
## Introduction

LRBench++ is a framework for effective learning rate benchmarking and tuning, which will help practitioners efficiently evaluate, select, and compose good learning rate policies for training DNNs.

### The impact of learning rates

The following figure shows the impacts of different learning rates. The FIX (black, k=0.025) reached the local optimum, while the NSTEP (red, k=0.05, γ=0.1, l=[150, 180]) converged to the global optimum. For TRIEXP (yellow, k0=0.05, k1=0.3, γ=0.1, l=100), even though it was the fastest, it failed to converge with high fluctuation.

![Comparison of three learning rate functions: FIX, NSTEP, and TRIEXP](assets/visualization/FIX-NSTEP-TRIEXP-Comparison.gif)

 
## Problem


## Installation

## Supported Platforms


## Development / Contributing


## Issues


## Status


## Contributors

See the [people page](https://github.com/git-disl/LRBenchPlusPlus/graphs/contributors) for the full listing of contributors.
