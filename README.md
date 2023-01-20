# Learning Multi-Scale Exposure Correction (LMSPEC) for Endoscopy

## Training

### Patches Extraction

Before training the model we must create random patches with different dimensions from both training and validation set.
Therefore, before continue with the extraction of patches we may want to locate our datasets as follows:

~~~
exposure_dataset/
      training/
          exp_images/
          gt_images/
      validation/
          exp_images/
          gt_images/
~~~

The given code in python includes a script called *patches_extraction.py* with two arguments: 1) *exposure_dataset* that
receives the directory that contains both training and validation datasets, and 2) *patches_size_pow* that receives a 
list with the powers of 2 that will express the size of the patches, besides to define the max number of patches (for 
instance, *"7 8"* for 128x128 and 256x256 patches, respectively). Then it will create directories with the patches for 
each patch size from all datasets. For example:

~~~
python patches_extraction.py --exposure_dataset "/path/to/exposure_dataset/" --patches_size_pow 7 8
~~~

