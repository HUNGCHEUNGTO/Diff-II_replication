# Inversion Circle Interpolation: Diffusion-based Image Augmentation for Data-scarce Classification (CVPR 2025)

**Diff-II is a novel and effective generative data augmentation method.**

![](teaser.jpg)


> [**Inversion Circle Interpolation: Diffusion-based Image Augmentation for Data-scarce Classification**](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Inversion_Circle_Interpolation_Diffusion-based_Image_Augmentation_for_Data-scarce_Classification_CVPR_2025_paper.pdf)
> 
> Yanghao Wang, Long Chen               

# Updates
- (02/2025) Diff-II was accepted by CVPR 2025.
- (12/2024) Code is released.


# Get Start
## Environment
Create a conda environment and activate it with the following command:
```bash
conda env create -f environment.yaml
conda activate diff-ii
```
If this takes too long, `conda config --set solver libmamba` sets conda to use the `libmamba` solver and could speed up installation.

## Download datasets, learned concepts, and suffixes
This may take some time.
```bash
bash scripts/download.sh
```

## Run code
We implemented it on 8 x 3090Ti GPUs. Thus, you can adjust the number of GPUs by modifying the code's hyperparameters.
Depending on your needs, modify the scripts/run.sh file and run.
```bash
bash scripts/run.sh
```

# Citing Diff-II

If you use Diff-II in your research or wish to refer to the baseline results published here, please use the following BibTeX entry.

```BibTeX
@inproceedings{wang2025inversion,
  title={Inversion circle interpolation: Diffusion-based image augmentation for data-scarce classification},
  author={Wang, Yanghao and Chen, Long},
  booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={25560--25569},
  year={2025},
  organization={IEEE}
}
```