### DeepLIIF: Deep-Learning Inferred Multiplex Immunofluoresence for IHC Quantification

[Read Link](https://www.biorxiv.org/content/10.1101/2021.05.01.442219v1) | [Google CoLab Demo](https://colab.research.google.com/drive/12zFfL7rDAtXfzBwArh9hb0jvA38L_ODK?usp=sharing)

*Reporting biomarkers assessed by routine immunohistochemical (IHC) staining of tissue is broadly used in diagnostic pathology laboratories for patient care. To date, clinical reporting is predominantly qualitative or semi-quantitative. By creating a multitask deep learning framework referred to as DeepLIIF, we are presenting a single step solution to nuclear segmentation and quantitative single-cell IHC scoring. Leveraging a unique de novo dataset of co-registered IHC and multiplex immunoflourescence (mpIF) data generated from the same tissue section, we simultaneously segment and translate low-cost and prevalent IHC slides to more expensive-yet-informative mpIF images. Moreover, a nuclear-pore marker, LAP2beta, is co-registered to improve cell segmentation and protein expression quantification on IHC slides. By formulating the IHC quantification as cell instance segmentation/classification rather than cell detection problem, we show that our model trained on clean IHC Ki67 data can generalize to more noisy and artifact-ridden images as well as other nuclear and non-nuclear markers such as CD3, CD8, BCL2, BCL6, MYC, MUM1, CD10 and TP53. We thoroughly evaluate our method on publicly available benchmark datasets as well as against pathologists’ semi-quantitative scoring.*

© This code is made available for non-commercial academic purposes.

![overview_image](./images/overview.png)*Figure 1. Overview of DeepLIIF pipeline and sample input IHCs (different brown/DAB markers -- BCL2, BCL6, CD10, CD3/CD8, Ki67) with corresponding DeepLIIF-generated hematoxylin/mpIF modalities and classified (positive (red) and negative (blue) cell) segmentation masks. (a) Overview of DeepLIIF. Given an IHC input, our multitask deep learning framework simultaneously infers corresponding Hematoxylin channel, mpIF DAPI, mpIF protein expression (Ki67, CD3, CD8, etc.), and the positive/negative protein cell segmentation, baking explainability and interpretability into the model itself rather than relying on coarse activation/attention maps. In the segmentation mask, the red cells denote cells with positive protein expression (brown/DAB cells in the input IHC), whereas blue cells represent negative cells (blue cells in the input IHC). (b) Example DeepLIIF-generated hematoxylin/mpIF modalities and segmentation masks for different IHC markers. DeepLIIF, trained on clean IHC Ki67 nuclear marker images, can generalize to noisier as well as other IHC nuclear/cytoplasmic marker images.*

## Prerequisites:
```del
Python 3
tk
Pillow>=7.1.2
matplotlib>=3.3.3
```
## Registration Framework
* To open the registration framework please run the following command:
```
python Registration_App.py
```

* To register the images, you have to first load the base image using the 'Open Base Image' button, and load the image you want to register over the base image using 'Open Moving Image' button.
* On the left side, you can see three viewers for visualizing: 1) the base image, 2) the moving image, 3) the moving image overlaid on the base image.
* Ater loading both images, you will see the moving image overlaid on top of the base image.
* **Translation:** Using the 'Left', 'Right', 'Up', and 'Down' buttons, you can move the moving image to the left, right, up, and down. You can change the translation value using the text box in the middle of the alignment buttons.
* **Scaling:** Using the 'Zoom in' and 'Zoom out' button, you can change the scale of the moving image. You can set the vertical and horizontal scaling value using the provided text boxes under the zoom buttons.
* **Rotation:** You can rotate the moving image using the provided buttons. You can set the rotation angle in the provided text box between two buttons!


## Issues
Please report all issues on the public forum.

## License
© DeepLIIF code is distributed under **Apache 2.0 with Commons Clause** license, and is available for non-commercial academic purposes. 

## Acknowledgments
* This code is inspired by [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our paper:
```
@article{ghahremani2021deepliif,
  title={DeepLIIF: Deep Learning-Inferred Multiplex ImmunoFluorescence for IHC Quantification},
  author={Ghahremani, Parmida and Li, Yanyun and Kaufman, Arie and Vanguri, Rami and Greenwald, Noah and Angelo, Michael and Hollmann, Travis J and Nadeem, Saad},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
