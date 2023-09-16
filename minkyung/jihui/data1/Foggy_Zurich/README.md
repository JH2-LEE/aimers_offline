# The Foggy Zurich Dataset

*Foggy Zurich* is a collection of 3808 real-world foggy road scenes in the city of Zurich and its suburbs. We provide semantic segmentation annotations for a diverse test subset *Foggy Zurich-test* with 40 scenes containing dense fog, which serves as a first benchmark for the domain of dense foggy weather. These scenes are annotated with fine pixel-level semantic annotations for the 19 evaluation classes of the [Cityscapes][cityscapes] dataset.

The above annotations have been created by using the annotation protocol of Cityscapes. Thus, the evaluation code that is provided with Cityscapes is easily adjustable for evaluation of semantic segmentation results on *Foggy Zurich-test*.

Details and **download** are available at our relevant project page: www.vision.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense


### Scope of Usage of Foggy Zurich

The larger, unlabeled part of *Foggy Zurich* is highly relevant for unsupervised or semi-supervised approaches. On the other hand, its smaller test subset *Foggy Zurich-test* is meant for **evaluation** purposes. Although its annotations for semantic segmentation could also be used to train respective models, we recommend against such usage, so that the dataset preserves its character as an objective, real-world evaluation benchmark for semantic scene understanding in the particularly challenging setting of dense fog.

**Important note**: when **evaluating on *Foggy Zurich-test***, it is highly recommended to only use the splits `light` and `medium` that we provide for the unlabeled part of the dataset (see the [splits section](#lists-of-files-for-splits) below for details), and not the entire set of unlabeled images.


### Dataset Structure

The directory structure of *Foggy Zurich* is as follows:
```
{root}/{type}/{sequence}/{id}{ext}
```

The meaning of the individual elements is:
 - `root`   the root directory where the dataset is stored.
 - `type`   the type/modality of data, e.g. `gt_labelIds` for semantic annotations in Cityscapes labelIds format, or `RGB` for 8-bit RGB images.
 - `sequence` the sequence which this part of the dataset has been extracted from, e.g. `GOPR0229`.
 - `id`     the identifier of the image: a non-empty sequence of letters and digits. The combination of `record` and `id` identifies each image in the dataset uniquely.
 - `ext`    the extension of the file , e.g. `.png` for semantic annotations.

Possible values of `type`
 - `RGB`               the RGB images in 8-bit format.
 - `gt_labelIds`       the fine semantic annotations, available for 40 testing images. Annotations are encoded using `png` images, where pixel values encode labels in Cityscapes IDs format. Please refer to the script `helpers/labels.py` in the [Cityscapes GitHub repository][cityscapesGithub] for details.
 - `gt_labelTrainIds`  the fine semantic annotations, available for 40 testing images, in which labels are encoded in Cityscapes trainIDs format.
 - `gt_color`          the fine semantic annotations, available for 40 testing images, in which labels are encoded in Cityscapes color format.


### Lists of Files for Splits

*Foggy Zurich* comprises multiple splits, related to the existence of semantic ground truth or the estimated fog density for each image. We do not place each split in a separate directory below each `type` directory, but provide lists of paths to all files in each split, with consistent order across different `type`s. These lists are contained as `txt` files in the directory `lists_file_names` and their naming format is the following:
```
{type}_{split}_filenames.txt
```

Possible values of `split`
 - `testv1`     first version of the test subset, containing 16 images. Evaluations in our ECCV 2018 publication have been carried out on this version.
 - `testv2`     second version of the test subset, containing 40 images, which consist of the 16 images of `testv1` plus 24 additional images.
 - `light`      subset of the unlabeled part of the dataset, containing 1556 images. For these images, fog density is estimated as light.
 - `medium`     subset of the unlabeled part of the dataset, containing 1498 images. For these images, fog density is estimated as medium.

We have manually cleaned the `light` and `medium` splits in order to exclude from them images which bear resemblance to any image in the `testv1` and `testv2` splits. This ensures a sound training and evaluation process when training on `light` or `medium` and evaluating on either of the test splits.


### License

*Foggy Zurich* is made available for non-commercial use under the license agreement which is contained in the attached file LICENSE.txt


### Contact

Please feel free to contact us with any questions or comments:

Christos Sakaridis
csakarid@vision.ee.ethz.ch
www.vision.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense

[cityscapes]: <https://www.cityscapes-dataset.com/>
[cityscapesGithub]: <https://github.com/mcordts/cityscapesScripts>
