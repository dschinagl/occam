# OccAM's Laser

This is the demo code for the paper:

**OccAM's Laser: Occlusion-based Attribution Maps for 3D Object Detectors on LiDAR Data**
<br>
[David Schinagl](https://dschinagl.github.io), [Georg Krispel](https://scholar.google.at/citations?user=Vt2vlgIAAAAJ&hl=de), 
[Horst Possegger](https://snototter.github.io/research/), [Peter M. Roth](https://scholar.google.at/citations?user=CgboCBAAAAAJ&hl=de),
and [Horst Bischof](https://scholar.google.com/citations?user=_pq05Q4AAAAJ&hl=de)
<br>
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022<br>
**[[Project Page]](https://dschinagl.github.io/occam/)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Paper]](https://arxiv.org/pdf/2204.06577.pdf)**<br>
<a href="https://dschinagl.github.io/occam/">
<img width="80%" src="docs/images/teaser.jpg"/>
</a>

<br>

> While 3D object detection in LiDAR point clouds is well-established in academia
> and industry, the explainability of these models is a largely unexplored 
> field. In this paper, we propose a method to generate attribution maps for the
> detected objects in order to better understand the behavior of such models. 
> These maps indicate the importance of each 3D point in predicting the specific 
> objects. Our method works with black-box models: We do not require any prior 
> knowledge of the architecture nor access to the model's internals, like 
> parameters, activations or gradients. Our efficient perturbation-based 
> approach empirically estimates the importance of each point by testing 
> the model with randomly generated subsets of the input point cloud. 
> Our sub-sampling strategy takes into account the special characteristics of 
> LiDAR data, such as the depth-dependent point density.

<img width=540 src="docs/images/example_progression.gif" /><br>

This repository is based on [`[OpenPCDet]`](https://github.com/open-mmlab/OpenPCDet) (version `v0.5.2`)

## Overview
- [Requirements](#requirements)
- [Installation](#installation)
- [Demo Example](#demo-example)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


## Requirements
* Linux (tested on Ubuntu 20.04 LTS)
* Python 3.8+
* CUDA (tested with version `11.3`)
* PyTorch (tested with version `1.10`)
* spconv 2.1.x (tested with version `2.1.21`)

## Installation

#### 1) Clone this repository
```shell
git clone https://github.com/dschinagl/occam.git
```

#### 2) Install PyTorch and SpConv depending on your environment
* [PyTorch](https://pytorch.org/get-started/locally/)
* [SpConv 2.1.x](https://github.com/traveller59/spconv)

#### 3) Install required packages
```shell
pip install -r requirements.txt
```

#### 4) Install `pcdet`
```shell
python setup.py develop
```

## Demo Example
We provide a quick demo to create attribution maps for detections of a 
[KITTI](http://www.cvlibs.net/datasets/kitti/) pretrained 
[PointPillars](https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html) 
model on custom point cloud data and to visualize the resulting maps.

1. Download the pretrained KITTI PointPillars model from the OpenPCDet 
[model-zoo](https://github.com/open-mmlab/OpenPCDet#model-zoo).

2. Prepare your custom point cloud data (skip this step if you use the provided sample `demo_pcl.npy`): 
   * You need to transform the coordinate of your custom point cloud to 
   the unified normative coordinate of `OpenPCDet` (x-axis front,y-axis left and z-axis top). 
   * The origin of your coordinate system should be about 1.6m above the ground, 
   since the provided model is trained on the KITTI dataset. 
   * The point cloud shape should be (N, 4) -> [x, y, z, intensity]
  
3. Run the demo as follows:
```shell
python occam_demo.py --ckpt ${PRETRAINED_POINTPILLARS_MODEL} \
    --source_file_path ${POINT_CLOUD_DATA}
```
`${POINT_CLOUD_DATA}` could be:
* The provided sample `demo_pcl.npy`
* A numpy array file (N, 4) like `my_data.npy`
* Original KITTI `.bin` data like `data/kitti/training/velodyne/000008.bin`

## Acknowledgement
We thank the authors of [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) for their open source release of their codebase.

## Citation

If you find this code useful for your research, please cite

```
@inproceedings{Schinagl2022OccAM,
  title={OccAM's Laser: Occlusion-based Attribution Maps for 3D Object Detectors on LiDAR Data},
  author={Schinagl, David and Krispel, Georg and Possegger, Horst and Roth, Peter M. and Bischof, Horst},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
