# The Foggy Driving Dataset

*Foggy Driving* is a collection of 101 real-world foggy road scenes with annotations for semantic segmentation and object detection, used as a benchmark for the domain of foggy weather. It features:
- dense, pixel-level semantic annotations of the images for the 19 evaluation classes of the [Cityscapes][cityscapes] dataset
- bounding box annotations for objects belonging to 8 of the above classes that correspond to humans or vehicles, i.e. *car*, *person*, *bicycle*, *bus*, *truck*, *train*, *motorcycle*, and *rider*.

The above annotations have been created by using the annotation protocol of Cityscapes. Thus, the evaluation code that is provided with Cityscapes is directly usable for evaluation of semantic segmentation results for *Foggy Driving* (see the [scripts section](#scripts) below for details).

Details and **download** are available at our relevant project page: people.ee.ethz.ch/~csakarid/SFSU_synthetic


### Foggy Driving Is Meant for Evaluation

*Foggy Driving* is meant for **evaluation** purposes, given its moderate scale. Although its annotations for semantic segmentation and object detection could also be used to train respective models, we recommend against such usage, so that the dataset preserves its character as an objective, real-world evaluation benchmark for semantic understanding of foggy scenes.


### Dataset Structure

The directory structure of *Foggy Driving* is as follows:
```
{root}/{type}/{split}/{record}/{record}_{id1}_{id2}{ext}
```

The meaning of the individual elements is:
 - `root`   the root directory where the dataset is stored.
 - `type`   the type/modality of data, e.g. `gtFine` for fine semantic annotations, or `leftImg8bit` for 8-bit RGB images.
 - `split`  the split, e.g. `test_extra`. Note that not all types of data exist for all splits.
 - `record` the context in which this part of the dataset was collected, e.g. `pedestrian` for images recorded from a pedestrian point of view.
 - `id1`    the first identifier of each image: a non-empty sequence of letters and digits.
 - `id2`    the second identifier of each image: a non-empty sequence of letters and digits. The combination of `record`, `id1`, and `id2` identifies each image in the dataset uniquely.
 - `ext`    an optional suffix `_{suffix}` followed by the extension of the file , e.g. `_gtCoarse_labelIds.png` for coarse semantic annotations, or `.txt` for bounding box annotations.

Possible values of `type`
 - `leftImg8bit`  the RGB images in 8-bit format.
 - `gtFine`       the fine semantic annotations, available for 33 testing images (`test`). Annotations are encoded using `png` images, where pixel values encode labels. Please refer to the script `helpers/labels.py` in the [Cityscapes GitHub repository][cityscapesGithub] for details.
 - `gtCoarse`     the coarse semantic annotations, available for 68 extra testing images (`test_extra`). The same specifications as for `gtFine` apply here.
 - `bboxGt`       the bounding box annotations induced from the above semantic annotations, available for all 101 images of the dataset. Annotations are encoded as `txt` files, in which each line corresponds to a single object and is formatted as
   ```
   {class} {xmin} {ymin} {xmax} {ymax}
   ```
   `class` stands for the ID of the class this object belongs to, and the rest four elements encode the extent of its bounding box in 1-based integer pixel coordinates. The 8 relevant classes are encoded with the following IDs:
   - *car*: 0
   - *person*: 1
   - *bicycle*: 2
   - *bus*: 3
   - *truck*: 4
   - *train*: 5
   - *motorcycle*: 6
   - *rider*: 7

   We recommend using only the bounding box annotations of the `test` split for object detection evaluation, since those of the `test_extra` split derive from *coarse* pixel-level semantic annotations and are not equally accurate.

Possible values of `split`
 - `test`       used for evaluation. Contains 33 images with fine pixel-level semantic annotations and induced bounding box annotations.
 - `test_extra` used for evaluation. Contains 68 images with coarse pixel-level semantic annotations and induced bounding box annotations.

Possible values of `record`
 - `pedestrian` used for images that were recorded from a pedestrian point of view.
 - `public`     used for images that were recorded from inside a public means of transportation.
 - `web`        used for images that were collected from the web and recorded in various contexts.

We also include the directory `lists_file_names`, which includes text files, each containing a **list of relative paths** to images in the corresponding subset of *Foggy Driving* (one image per line of the text file). These lists of file names are particularly useful for conducting experiments across various programming frameworks.
 - `leftImg8bit_testall_filenames.txt`		all 101 images of the dataset.
 - `leftImg8bit_testfine_filenames.txt`		33 images with fine pixel-level semantic annotations.
 - `leftImg8bit_testdense_filenames.txt`	21 images depicting scenes with dense fog, identified manually.

### Scripts

We additionally provide
 - `evalPixelLevelSemanticLabelingFoggyDriving.py` a Python script for evaluation of semantic segmentation results on the *complete* set of 101 images. Has been adjusted from the corresponding evaluation script provided with Cityscapes.
 - `run_evalPixelLevelSemanticLabelingFoggyDriving.sh` a Linux shell script template for configuring the execution of the above Python script.

Please download first the scripts that are provided in the [Cityscapes GitHub repository][cityscapesGithub] (in the directory `cityscapesscripts`), since our aforementioned scripts depend on them.


### License

*Foggy Driving* is made available for non-commercial use under the license agreement which is contained in the attached file LICENSE.txt


### Contact

Please feel free to contact us with any questions or comments:

Christos Sakaridis
csakarid@vision.ee.ethz.ch
people.ee.ethz.ch/~csakarid/SFSU_synthetic

[cityscapes]: <https://www.cityscapes-dataset.com/>
[cityscapesGithub]: <https://github.com/mcordts/cityscapesScripts>
