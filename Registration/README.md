### DeepLIIF: Deep-Learning Inferred Multiplex Immunofluoresence for IHC Image Quantification

© This code is made available for non-commercial academic purposes.

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
© [Nadeem Lab](https://nadeemlab.org/) - DeepLIIF code is distributed under **Apache 2.0 with Commons Clause** license, and is available for non-commercial academic purposes. 

## Reference
If you find our work useful in your research or if you use parts of this code, please cite our paper:
```
@article{ghahremani2021deepliif,
  title={DeepLIIF: Deep Learning-Inferred Multiplex ImmunoFluorescence for IHC Quantification},
  author={Ghahremani, Parmida and Li, Yanyun and Kaufman, Arie and Vanguri, Rami and Greenwald, Noah and Angelo, Michael and Hollmann, Travis J and Nadeem, Saad},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
