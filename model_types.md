# Models

## Usage
In general, option `--model` controls the type of model used in training:
```
deepliif train ... --model <model_name>
```
During testing or inference, the model type information will automatically be fetched from the recorded options produced during training.

Note that there could be additional configurable options for different model types. See below for details.


## Available Models
Currently we provide 5 models ready to be used with `deepliif` package. 

Input training data typically is a row of same-shaped square patches stitched together. Using DeepLIIF as an example, the input in the publicated paper is (IHC, Hematoxylin, DAPI, Lap2, Marker, Seg). This can be understood as (base mod, mod 1, mod 2, mod 3, mod 4, seg).

|Model Name|Full Model Name|Tasks|Description|Training Data|Additional Config|
|----------|---------------|-----|-----------|-------------|-----------------|
|DeepLIIF|DeepLIIF|modality translation, segmentation|The original DeepLIIF model as published in [Nature MI'22](https://rdcu.be/cKSBz). This model class has been adapted to accept a configurable number of modalities.|Original: (base mod, mod 1, mod 2, mod 3, mod 4, seg)<br>New training: (base mod, ..., seg)|`--modalities-no`: the number of modalities to translate to (can be 0)|
|DeepLIIFExt|DeepLIIF Extension|modality translation, segmentation (optional)|An extended version of DeepLIIF model that allows for more flexibility.|(base mod, mod 1, ..., seg 1, ...)|`--seg-gen`: whether involving segmentation task<br>`--modalities-no`: the number of modalities to translate to (>=1)|
|DeepLIIFKD|DeepLIIF Knowledge Distillation|knowledge distillation|This creates a student model (usually much smaller or simpler) and learns the output from the target large teacher model. This currently has only been tested with `DeepLIIF` as the teacher model, and may not be compatible with other model types.|(base mod, mod 1, mod 2, mod 3, mod 4, seg)|`--model-dir-teacher`: the directory of a trained large model to be used as the teacher|
|SDG|Synthetic Data Generation|modality translation|This model has been published as part of [MICCAI'24](https://arxiv.org/abs/2405.08169). It translates provided modalities to the target modalities. In the paper, we used it to increase the resolution of stitched WSI from video frames. **This is going to be merged with DeepLIIF.**|||
|CycleGAN|CycleGAN|**unpaired** modality translation|This model is adapted from CycleGAN, where the main usage is to learn the translation from input domain to the target domain without paired ground truth data.|(base mod, mod 1)|``|


## Model Details
### 1. DeepLIIF
DeepLIIF uses IHC as training input as well as 4 additional modalities (Hematoxylin, DAPI, Lap2, protein marker) to learn the translation task (4 generators, 1 for each modality), followed by a segmentation ground truth predicted collectively by 5 generators, each relying on one modality (IHC plus 4 translated modalities). The actual training data consists of wide images, where each has IHC + 4 modalities + segmentation stitched together in a row (see the Datasets folder for examples).

During inference, only the IHC input is needed.

The original setting employs **ResNet-9block** as the backbone for translation generators and **UNet-512 (9 down layers)** for segmentation generatros. 

#### DeepLIIF with a configurable number of modalities
We recently updated DeepLIIF model class to allow an arbitrary number of modalities (can even be 0). `--modalities-no 0` practically means 0 modality to translate to, and there will only be 1 generator that uses the base input mod (e.g., IHC) for the segmentation task (i.e., the whole DeepLIIF model in this case will only have 1 generator, rather than 4+5=9 generators in the original setting).

### 2. DeepLIIFExt
Mainly based on DeepLIIF, this extension model allows to
- learn only the modality translation task (`--seg-gen false`)
- use modality-wise segmentation ground truth (e.g., if use IHC as the base input, Lap2 and Marker as the additional modalities to learn translation to, then the training image should include 2 segmentation ground truth, 1 for each additional modalities)
- train any number of modalities (e.g., `--modalities-no 2` for 2 modalities to learn translation task for)
- modify loss function (DeepLIIF uses BCE for translation and LSGAN for segmentation)

Some other noticeable differences include:
1. The training data to DeepLIIFExt requires **1 segmentation ground truth per modality**, rather than 1 final segmentation as in DeepLIIF.
2. The input to segmentation generator is not only the original base image (IHC in DeepLIIF) or a translated modality, but a concatenated vector of **original IHC, the first translated modality, and the current translated modality**. For example, if the sequence of patches in the training image is (IHC, Lap2, Marker, Lap2-Seg, Marker-Seg), then the input to the first segmentation generator will be (IHC, translated Lap2, translated Lap2), and the input to the second segmentation generator will be (IHC, translated Lap2, translated Marker).
3. The input to segmentation discriminator, as part of conditional GAN's practice, includes more context. In DeepLIIF, the context is the original IHC or a real modality image combined with the tensor to be evaluated in this context which is the aggregated final segmentation (fake case) or the real segmentation output (true case). In DeepLIIFExt, the context includes **the first real modality, and the current real modality**. Using the same example above, the input to the first discriminator will be (IHC, real Lap2, real Lap2, generated Lap2-Seg / real Lap2-Seg), and the input to the second discriminator will be (IHC, real Lap2, real Marker, generated Marker-Seg / real Marker-Seg).
4. DeepLIIF model includes a segmentation generator for the base input modality (e.g., IHC), while DeepLIIFExt does not.


|Input to|Original DeepLIIF|DeepLIIFExt|
|--------|--------|-----------|
|Translation generators|(IHC)|(base mod)|
|Segmentation generators|(IHC)<br>(generated Hema)<br>...|(base mod, generated mod 1, generated mod 1)<br>(base mod, generated mod 1, generated mod 2)<br>...|

### 3. DeepLIIFKD
Our current KD approach simply flattens the output RGB tensors `(3, 512, 512)` into vectors `(1, 3*512*512)` and then applies KL divergence loss to the student’s output and the teacher’s output. The KL divergence loss is summed up for all 10 outputs (4 modality translations, 5 intermediate segmentations, and 1 final segmentation), and then added to the final loss term for back propagation.

If we view the whole deepliif model set as one big model that produces the aggregated segmentation image as final output, then the 4 modality translation outputs and 5 intermediate segmentation outputs can be understood as intermediate features, which helps the student model to mimic how the teacher model arrives at the final segmentation output. In this sense, we do not incorporate "real" intermediate feature losses as other approaches did by comparing middle-layer output tensors of both models, but still effectively achieve the same purpose.

The input to DeepLIIFKD is the same as that to DeepLIIF.

### 4. SDG
To be deleted if SDG can be merged with DeepLIIF.

### 5. CycleGAN
The DeepLIIF model family requires paired images to learn the mapping. This, however, is not always achievable during data collection or might require considerably more efforts. Hence we apply the idea of CycleGAN for unpaired modality translation. 

The core idea is for the model to learn `f(A) = B` and `g(f(A)) = A`, where `A` and `B` denotes data from the input and target domain, and `f(x)` and `g(x)` denotes two mapping functions approximated by neural networks. For example, the input domain can be IHC and the target domain can be KI67. Generator `f(x)` learns how to map IHC to KI67 and generator `g(x)` learns how to map the translated KI67 back to the original IHC. In this case, we only need generator `f(x)` for inference.

Our implementation of CycleGAN supports learning the mapping to multiple domains at once. Essentially, this becomes a multi-task learning: for each target domain `B1`, `B2`, ..., we create a separate generator set `f(x)` and `g(x)`. The losses from each generator set then gets combined for back propagation.


