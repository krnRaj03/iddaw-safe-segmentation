# Safety-Aware Segmentation of Indian Roads in Adverse Weather

Capstone project for QM640 Data Analytics, Walsh College (Fall 2026).
Author: Karan Raj Modgil. Mentor: Sharath Srivatsa.

> **Status:** synopsis stage. Code and results will be added as the project progresses.

## About
Self-driving systems need to read road scenes accurately, but Indian roads are unstructured
and bad weather makes things harder. Standard mIoU scores also treat every error alike, even
though labelling a pedestrian as road is far worse than confusing two types of vehicle.

This project uses [IDD-AW](https://iddaw.github.io/) to ask four questions:

1. How big is the gap between mIoU and Safe mIoU (SmIoU) in rain, fog, low light and snow?
2. Does adding the near-infrared (NIR) image to RGB improve accuracy?
3. Does a loss based on label-tree distance improve SmIoU without hurting mIoU?
4. Does the best model make fewer dangerous errors, such as pedestrians, riders or animals
   predicted as drivable road?

## Data
The images are not stored here. To get them, register on the
[IDD portal](http://idd.insaan.iiit.ac.in/dataset/download/) and request a download token.
Check the dataset licence at download time.

The study uses the ICPR 2024 competition split: 3,430 train, 475 validation and
1,095 test images.

## Repository layout
`data/` download notes, labels and manifest · `notebooks/` · `src/` · `configs/` ·
`results/` · `reports/`

## Reproducing the results
Coming with the final project: environment file, download steps and run order.

## References
- Shaik et al. (2024). *IDD-AW: A Benchmark for Safe and Robust Segmentation of Drive Scenes
  in Unstructured Traffic and Adverse Weather.* WACV 2024.
- Shaik et al. (2024). *ICPR 2024 Competition on Safe Segmentation of Drive Scenes in
  Unstructured Traffic and Adverse Weather Conditions.* arXiv:2409.05327.

## Licence
Code is released under the MIT licence (or your choice). The dataset keeps its own licence.
