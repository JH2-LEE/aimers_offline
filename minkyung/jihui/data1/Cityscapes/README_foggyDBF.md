# Foggy Cityscapes-DBF

*Foggy Cityscapes-DBF* derives from the Cityscapes dataset and constitutes a collection of partially synthetic foggy images that automatically inherit the semantic annotations of their real, clear counterparts in Cityscapes. These foggy images are generated from the clear-weather images of Cityscapes with the fog simulation pipeline using semantics that is presented in the relevant publication (see Citation below). This pipeline involves computation of a denoised and complete depth map, followed by refinement of the corresponding transmittance map with a novel Dual-reference cross-Bilateral Filter (DBF) that uses both color and semantics as reference, from which the dataset borrows its name. *Foggy Cityscapes-DBF* constitutes the successor of the *Foggy Cityscapes* dataset, which had been previously available on the Cityscapes server and was presented in a preceding publication as listed on: 
https://www.vision.ee.ethz.ch/~csakarid/SFSU_synthetic/
*Foggy Cityscapes-DBF* contains synthetic foggy images with better adherence to semantic boundaries in the scene than its predecessor *Foggy Cityscapes*.

Details and **further downloads** for *Foggy Cityscapes-DBF* (including transmittance maps) are available at: 
https://www.vision.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/
For details on the original Cityscapes dataset, please refer to:
www.cityscapes-dataset.net


### Dataset Structure

The folder structure of *Foggy Cityscapes-DBF* follows that of Cityscapes, with minor extensions. Please refer to the README file of the Cityscapes git repository for a detailed presentation of this structure: 
https://github.com/mcordts/cityscapesScripts

In particular, following the notation in the aforementioned README, the extension field `ext` of the synthetic foggy versions of left 8-bit Cityscapes images includes additional information on the attenuation coefficient `beta` which was used to render them, as shown in the following sample image name:
```
erfurt_000000_000019_leftImg8bit_foggy_beta_0.01.png
```
Foggy images in the `train` and `val` splits are available for `beta` equal to `0.005`, `0.01`, and `0.02`. No foggy images are available for the `test` split.


### Foggy Cityscapes-refined

A refined list of 550 Cityscapes images (498 `train` plus 52 `val`) that yield high-quality synthetic foggy images is provided in the file `foggy_trainval_refined_filenames.txt`.


### Citation

If you use *Foggy Cityscapes-DBF* in your research, please cite both the Cityscapes publication and the publication that introduces *Foggy Cityscapes-DBF* as listed on the relevant website:
https://www.vision.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/


### Contact

Christos Sakaridis
csakarid@vision.ee.ethz.ch
https://www.vision.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/
