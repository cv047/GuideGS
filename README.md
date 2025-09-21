# GuideGS: Enhancing 3D Gaussian Splatting with Vision Foundation Models for Sparse-View Synthesis
> Official implements of GuideGS

## Pipeline
<p align="center">
  <img src="assets\framework.jpg" alt="pipeline" width="100%">
</p>

## Setup
#### 1. Recommended environment
```
git clone https://github.com/prstrive/GuideGS.git

conda env create --file environment.yml
conda activate guidegs

git clone https://github.com/ashawkey/diff-gaussian-rasterization --recursive submodules
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules
pip install submodules/diff-gaussian-rasterization submodules/simple-knn

git clone https://github.com/enesmsahin/simple-lama-inpainting.git
pip install simple-lama-inpainting
```
We use CUDA 11.8 as our environment.

#### 2. Dataset Download

**LLFF dataset**

Download LLFF dataset from the official [download link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

**Tanks&Temples Dataset**

For Tanks dataset, you can download them from [here](https://www.robots.ox.ac.uk/~wenjing/Tanks.zip).

**IBRNet Dataset**

For the IBTNet data set, please refer to [here](https://github.com/googleinterns/IBRNet).

#### 3. Depth Prior Extraction

In our project, we use Depth-Anything-V2 to acquire the monocular depth. Download the model checkpoints from [here](https://github.com/DepthAnything/Depth-Anything-V2) and put them in [depthanythingv2\checkpoints](depthanythingv2\checkpoints).Then,run:
```
python run.py --encoder <vits | vitb | vitl | vitg> --img-path <path> --outdir <outdir> --grayscale
```



#### 3. Match Prior Extraction

Our matching prior extraction method is the same as [SCGaussian](https://github.com/prstrive/SCGaussian).See the SCGaussian for details.

Clone the [GIM](https://github.com/xuelunshen/gim) model first and setup the environment.Put ```get_match_info.py``` in the GIM directory. You can use this script to extract match prior.
## 🚀 Evaluation

#### 1. Optimization

Optimize the model for the specific scene first:
```
python train.py -s <path to scene> -m <path to save outputs> -r 8 --eval
```

#### 2. Rendering

Then render the novel view synthesis results:
```
python render.py -m <path to save outputs>
```

#### 3. Metrics

Compute the quantitative results:
```
python metrics.py -m <path to save outputs>
```


## Acknowledgements

Thanks to the following awesome open source projects!

- [SCGaussian](https://github.com/prstrive/SCGaussian)
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [lama](https://github.com/advimman/lama)


