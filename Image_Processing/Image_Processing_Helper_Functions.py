import os

import cv2
import numpy as np
from skimage.color import rgb2hed, hed2rgb

from Augmentation import Augmentation

from PIL import Image
import javabridge
import bioformats
import tifffile as tf

import bioformats.omexml as ome
import sys


def create_training_testing_dataset_from_given_directory(input_dir, output_dir, post_fix_names=['IHC', 'DAPI', 'Hema', 'Lap2', 'Marker', 'Seg'], subsets={'train': 0.7, 'val': 0.15, 'test': 0.15}, tile_size=512):
    """
    Create a dataset containing 'train'/'val'/'test' subsets from the files in a directory by concatenating the images (side-by-side).

    Sample input directory:
        ├── im1_IHC.png
        ├──im1_DAPI.png
        ├── im1_Hema.png
        ├── im1_Lap2.png
        ├── im1_Marker.png
        ├── im1_Seg.png
        ├──...
        └── im5_Seg.png

    Sample output directory:
        ├── train
            ├── im1.png (concatenated of im1_IHC.png/im1_DAPI.png/im1_Hema.png/im1_Lap2.png/im1_Marker.png/im1_Seg.png
            ├── im2.png (concatenated of im2_IHC.png/im2_DAPI.png/im2_Hema.png/im2_Lap2.png/im2_Marker.png/im2_Seg.png
            └── im3.png (concatenated of im3_IHC.png/im3_DAPI.png/im3_Hema.png/im3_Lap2.png/im3_Marker.png/im3_Seg.png
        ├── val
            └── im4.png (concatenated of im4_IHC.png/im4_DAPI.png/im4_Hema.png/im4_Lap2.png/im4_Marker.png/im4_Seg.png
        └── test
            └── im5.png (concatenated of im5_IHC.png/im5_DAPI.png/im5_Hema.png/im5_Lap2.png/im5_Marker.png/im5_Seg.png

    :param input_dir: The directory containing the original data.
    :param output_dir: The directory for saving the concatenated images in the corresponding subdirectories.
    :param post_fix_names: Postfix names of the original images in the directory.
    :param subsets: A dictionary containing subdirectory names including training, validation and testing sets (or any other subsets) with the corresponding split value.
    :param tile_size: Size of each image used for training and testing (the original image is resized to this size).
    :return:
    """
    # Create subdirectories if not exists.
    all_dirs = []
    for subdir in subsets.keys():
        curr_dir = os.path.join(output_dir, subdir)
        all_dirs.append(curr_dir)
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)

    # Read images and concatenate them.
    print('Started concatenating images!')
    images = os.listdir(input_dir)
    all_images = {}
    for img in images:
        if post_fix_names[0] in img:
            try:
                images = []
                for post_fix in post_fix_names:
                    images.append(cv2.resize(cv2.imread(os.path.join(input_dir, img.replace(post_fix_names[0], post_fix))), (tile_size, tile_size)))
                all_images[img.replace(post_fix_names[0], '')] = np.concatenate(images, 1)
            except Exception as e:
                print('Cannot find all modalities for image ' + img)
    print('Finished concatenating images!')

    # Split the images into the given subsets according to the given split values by the user.
    print('Creating dataset splits!')
    allFileNames = list(all_images.keys())
    split_values = []
    curr_val = 0
    for key, value in subsets.items():
        curr_val += value
        split_values.append(int(len(allFileNames) * curr_val))
    split_FileNames = np.split(np.array(allFileNames), split_values)

    # Saving the images into the corresponding subdirectories
    print('Saving images!')
    for i in range(len(split_FileNames)):
        filenames = split_FileNames[i]
        for filename in filenames:
            cv2.imwrite(os.path.join(all_dirs[i], filename), all_images[filename])


def augment_set(input_dir, output_dir, aug_no=9, modality_types=['hematoxylin', 'CD3', 'PanCK'], tile_size=512):
    """
    This function augments a co-aligned dataset.

    Sample input directory:
        ├── im1.png
        ├── im2.png
        └── im3.png

    :param input_dir: The directory containing original images.
    :param output_dir: The directory for saving the concatenated images in the corresponding subdirectories.
    :param aug_no: Number of augmentation applied on each image.
    :param modality_types: Modality types.
    :param tile_size: Size of each tile in the concatenated image.
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images = os.listdir(input_dir)
    for img in images:
        augmented = 0
        while augmented < aug_no:
            images = {}
            image = cv2.imread(os.path.join(input_dir, img))
            for i in range(0, len(modality_types)):
                images[modality_types[i]] = image[:, i * tile_size: (i + 1) * tile_size]
            new_images = images.copy()
            aug = Augmentation(new_images)
            aug.pipeline()
            cv2.imwrite(os.path.join(output_dir, img.replace('.png', '_' + str(augmented) + '.png')),
                        np.concatenate(list(new_images.values()), 1))
            augmented += 1


def augment_created_dataset(input_dir, output_dir, aug_no=9, modality_types=['hematoxylin', 'CD3', 'PanCK'], tile_size=512):
    """
    This function goes through all subsets in the given input directory and calls augment_set function to augment each co-aligned subset.

    Sample input directory:
        ├── train
            ├── im1.png
            ├── im2.png
            ├── im3.png
            └── im4.png
        ├── val
            ├── im5.png
            └── im6.png
        └── test
            ├── im7.png
            └── im8.png

    :param input_dir: The directory containing original images.
    :param output_dir: The directory for saving the concatenated images in the corresponding subdirectories.
    :param aug_no: Number of augmentation applied on each image.
    :param modality_types: Modality types.
    :param tile_size: Size of each tile in the concatenated image.
    :return:
    """
    subdirs = os.listdir(input_dir)
    for subdir in subdirs:
        if os.path.isdir(os.path.join(input_dir, subdir)):
            print('Started augmenting images in ' + subdir + ' set')
            augment_set(os.path.join(input_dir, subdir), os.path.join(output_dir, subdir), aug_no=aug_no, modality_types=modality_types, tile_size=tile_size)
            print('Finished augmenting images in ' + subdir + ' set')


def imadjust(x, gamma=0.7, c=0, d=1):
    """
    This funstion adjusts the contrast of an image.
    :param x: The given image.
    :param gamma: The gamma value.
    :param c: The minimum value.
    :param d: The maximum value.
    :return: The adjusted image (float).
    """
    a = x.min()
    b = x.max()
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def stain_deconvolution(ihc_rgb):
    """
    This function performs stain deconvolution on input ihc image.
    :param ihc_rgb: The input ihc image.
    :return: The deconvolved brown stain.
    """
    ihc_hed = rgb2hed(np.array(ihc_rgb))

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    ihc_d = cv2.cvtColor((np.sqrt(ihc_d) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # ihc_d = np.max(ihc_d) - ihc_d
    # ihc_d = cv2.cvtColor((ihc_d * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # ihc_d = np.max(ihc_d) - ihc_d
    # ihc_d = self.imadjust(ihc_d, 1)
    ihc_d = cv2.cvtColor(ihc_d.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
    return ihc_d


def create_multi_channel_image(input_dir, output_dir, stacked=True, image_channels_names=None):
    """
    This function provide two options for visualizing the generated modalities from the model.
    In the first option (stacked=False), the function puts all images side by side.
    In the second option (stacked=True), the function create a multi-channel image readable by QuPath.
    Each channel in this image represents a modality.
    The user should provide the image postfix names and the channels in each image
    that should be included in the multi-channel image.

    For example consider 'real_A' is the IHC image and user wants to save all three channels of it,
    then the user must give the channel names of all three channels in the image_channel_names dictionary.
    Consider 'real_B_1' is the DAPI image which is a grayscale image, then user should only specify one channel name.

    :param input_dir: The directory containing input images.
    :param output_dir: The directory containing output images.
    :param stacked: The option for saving images as a stack of images or a strip of images.
    :param image_channels_names: A dictionary containing post-fix names of the images and the channel names
                                 for saving in the stacked image.
    :return:
    """
    if image_channels_names is None:
        image_channels_names = {'real_A': ['IHC_R', 'IHC_G', 'IHC_B'], 'real_B_1': ['DAPI_Orig'], 'fake_B_1': ['DAPI_Gen']}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images = os.listdir(input_dir)

    javabridge.start_vm(class_path=bioformats.JARS)
    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", "WARN", "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

    image_names = list(image_channels_names.keys())
    for img in images:
        if image_names[0] in img:
            try:
                if not stacked:
                    images = []
                    for img_name in image_names:
                        images.append(cv2.cvtColor(cv2.imread(os.path.join(input_dir, img.replace(image_names[0], img_name))), cv2.COLOR_RGB2BGR))
                    image = np.concatenate(images, 1)
                    cv2.imwrite(os.path.join(output_dir, img.replace(image_names[0], '')), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                else:
                    channel_names = []
                    all_images = []
                    filepath = os.path.join(output_dir, img.replace(image_names[0] + img.split(image_names[0])[-1], '.ome.tif'))
                    for img_postfix in image_names:
                        image = cv2.imread(os.path.join(input_dir, img.replace(image_names[0], img_postfix)))
                        current_image_channel_names = image_channels_names[img_postfix]
                        channel_names.extend(current_image_channel_names)
                        if len(current_image_channel_names) > 1:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            for i in range(len(current_image_channel_names)):
                                all_images.append(image[:, :, i])
                        elif len(current_image_channel_names) == 1:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            all_images.append(image)
                        else:
                            continue
                    all_images = np.array(all_images)
                    all_images = all_images.reshape((1, 1, all_images.shape[0], all_images.shape[1], all_images.shape[2]))
                    write_ome_tiff_file(all_images, filepath, SizeC=len(channel_names), channel_names=channel_names)
            except Exception as e:
                print(e)
                pass
    javabridge.kill_vm()


def write_ome_tiff_file(img, output_file, SizeT=1, SizeZ=1, SizeC=1, SizeX=2048, SizeY=2048, channel_names=[], Series=0, scalex=0.10833, scaley=0.10833, scalez=0.3, pixeltype='uint8', dimorder='TZCYX'):
    """
    This function writes an ome tiff image along with the corresponding xml file.

    :param img: The image array.
    :param output_file: The address for saving the output image.
    :param SizeT: Size t
    :param SizeZ: Size z
    :param SizeC: Size c
    :param SizeX: Size x
    :param SizeY: Size y
    :param channel_names: The name of the channels to be saved.
    :param Series: The series number.
    :param scalex: Physical Size x
    :param scaley: Physical Size y
    :param scalez: Physical Size z
    :param pixeltype: The pixeltype.
    :param dimorder: The dimension order to save the image (default: TZCYX).
    :return:
    """
    def writeplanes(pixel, SizeT=1, SizeZ=1, SizeC=1, order='TZCYX', verbose=False):
        if order == 'TZCYX':
            p.DimensionOrder = ome.DO_XYCZT
            counter = 0
            for t in range(SizeT):
                for z in range(SizeZ):
                    for c in range(SizeC):

                        if verbose:
                            print('Write PlaneTable: ', t, z, c),
                            sys.stdout.flush()

                        pixel.Plane(counter).TheT = t
                        pixel.Plane(counter).TheZ = z
                        pixel.Plane(counter).TheC = c
                        counter = counter + 1

        return pixel

    print('Started writing ome tiff file -> ' + output_file)
    # Getting metadata info
    omexml = ome.OMEXML()
    omexml.image(Series).Name = output_file
    p = omexml.image(Series).Pixels
    p.SizeX, p.SizeY, p.SizeT, p.SizeC = SizeX, SizeY, SizeT, SizeC
    p.PhysicalSizeX, p.PhysicalSizeY, p.PhysicalSizeZ = scalex, scaley, scalez
    p.PixelType = pixeltype
    p.channel_count = SizeC

    for i in range(len(channel_names)):
        p.Channel(i).set_Name(channel_names[i])
        p.Channel(i).set_ID(channel_names[i])
    p.plane_count = SizeZ * SizeT * SizeC
    p = writeplanes(p, SizeT=SizeT, SizeZ=SizeZ, SizeC=SizeC, order=dimorder)

    for c in range(SizeC):
        if pixeltype == 'unit8':
            p.Channel(c).SamplesPerPixel = 1
        if pixeltype == 'unit16':
            p.Channel(c).SamplesPerPixel = 2

    omexml.structured_annotations.add_original_metadata(
        ome.OM_SAMPLES_PER_PIXEL, str(SizeC))

    # Converting to omexml
    xml = omexml.to_xml()

    with tf.TiffWriter(output_file
                       # , bigtiff=True
                       # , imagej=True
                       ) as tif:
        for t in range(SizeT):
            for z in range(SizeZ):
                for c in range(SizeC):
                    tif.write(img[t, z, c, :, :]
                              , description=xml
                              , photometric='minisblack'
                              , metadata={'axes': 'TZCYX'
                             , 'DimensionOrder': 'TZCYX'
                             , 'Resolution': 0.10833
                             , 'Channel': {'Name': channel_names[c]}}
                              )

    print('Finished writing ome tiff file -> ' + output_file)


def read_bioformats_image(filename, channel=0):
    """
    This function reads an image using bioformats and return an Image pillow object and the pixels array.

    :param filename: The address to the image.
    :param channel: The channel number in the original image.
    :return: Image pillow object, pixels array
    """
    javabridge.start_vm(class_path=bioformats.JARS)

    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", "WARN", "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

    metadata = bioformats.get_omexml_metadata(filename)
    omexml = bioformats.OMEXML(metadata)
    print('Started Reading Image ' + filename)
    print('SizeX:', omexml.image().Pixels.SizeX)
    print('SizeY:', omexml.image().Pixels.SizeY)
    print('SizeZ:', omexml.image().Pixels.SizeZ)
    print('SizeC:', omexml.image().Pixels.SizeC)
    print('SizeT:', omexml.image().Pixels.SizeT)
    print('PixelType:', omexml.image().Pixels.PixelType)

    if omexml.image().Pixels.PixelType == 'uint8':
        print('int image')
        pixels = bioformats.load_image(filename, rescale=False, t=channel)
        image = Image.fromarray(pixels)
    else:
        print('float image')
        pixels = bioformats.load_image(filename, rescale=True, t=channel)
        pixels *= 255
        pixels = np.rint(pixels).astype(np.uint8)
        image = Image.fromarray(pixels)
    print('Done Reading Image ' + filename)
    return image, pixels


def read_region_of_image_using_bioformats(path, channel=0, region=(0,0,0,0)):
    """
    Using this function, you can read a specific region of a large image by giving the region bounding box (XYWH format)
    and the channel number.

    :param path: The address to the file.
    :param channel: The channel number.
    :param region: The bounding box around the region of interest (XYWH format).
    :return: The specified region of interest image (numpy array).
    """
    javabridge.start_vm(class_path=bioformats.JARS)

    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", "WARN", "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

    with bioformats.ImageReader(path) as reader:
        return reader.read(t=channel, XYWH=region)


def get_ome_information(filename):
    """
    This function reads all information in the xml of the given ome image.

    :param filename: The address to the ome image.
    :return: size_x, size_y, size_z, size_c, size_t, pixel_type
    """
    javabridge.start_vm(class_path=bioformats.JARS)

    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", "WARN", "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

    metadata = bioformats.get_omexml_metadata(filename)
    omexml = bioformats.OMEXML(metadata)
    size_x, size_y, size_z, size_c, size_t, pixel_type = omexml.image().Pixels.SizeX, \
                                                         omexml.image().Pixels.SizeY, \
                                                         omexml.image().Pixels.SizeZ, \
                                                         omexml.image().Pixels.SizeC, \
                                                         omexml.image().Pixels.SizeT, \
                                                         omexml.image().Pixels.PixelType
    print('SizeX:', size_x)
    print('SizeY:', size_y)
    print('SizeZ:', size_z)
    print('SizeC:', size_c)
    print('SizeT:', size_t)
    print('PixelType:', pixel_type)
    return size_x, size_y, size_z, size_c, size_t, pixel_type


def create_dataset_from_ome_multi_channel_image(ome_dir, output_addr, tile_size=1024, img_channel_modality=None):
    """
    This function creates a dataset from given directory containing OME tiff files with multiple channels.
    The user can specify the size of the tiles to be extracted from the OME image and saved in the output directory.
    The user must give the information of the channels of the OME tiff files in the format of the dictionary,
    where the key is the channel name and the value is the corresponding channel number.
    If multiple channels should be stacked to create the final image, the user can give all the channel numbers in a list.
    For example, channels of the IHC image are saved independently in the OME tiff file.
    So, the user can specify the R, G, and B channel numbers as a list ('IHC':[2,3,4]).

    :param ome_dir: The path to the directory containing ome tiffs.
    :param output_addr: The path to the directory containing the output tiles.
    :param tile_size: The size of the tiles extracted from the WSI.
    :param img_channel_modality: A dictionary containing the information about channels in OME file.
                                 The keys are channel names and values are the corresponding channel numbers.
                                 Sample img_channel_modality: {'DAPI': 0, 'PD1': 1, 'IHC': [2, 3, 4]}
    :return: 
    """
    if img_channel_modality is None:
        print('img_channel_modality not given!')
        return 
    if not os.path.exists(output_addr):
        os.makedirs(output_addr)
    region_size = 10240
    omes = os.listdir(ome_dir)
    for ome in omes:
        current_addr = os.path.join(ome_dir, ome)
        size_x, size_y, size_z, size_c, size_t, pixel_type = get_ome_information(current_addr)
        images_dict = {}
        for i in range(0, size_x, region_size):
            for j in range(0, size_y, region_size):
                for img_type in img_channel_modality:
                    img_channels = img_channel_modality[img_type]
                    images = []
                    if type(img_channels) == list:
                        for img_channel in img_channels:
                            images.append(read_region_of_image_using_bioformats(current_addr,
                                                                                channel=img_channel,
                                                                                region=(i, j, min(region_size, size_x - i), min(region_size, size_y - j))))
                        
                        images_dict[img_type] = np.dstack(images)
                    else:
                        images_dict[img_type] = read_region_of_image_using_bioformats(current_addr,
                                                                                      channel=img_channels,
                                                                                      region=(i, j, min(region_size, size_x - i), min(region_size, size_y - j)))

                create_dataset_from_WSI_regions(images_dict, output_addr,
                                                ome.split('_')[0], tile_size=tile_size, start_i=i, start_j=j)

    javabridge.kill_vm()


def create_dataset_from_WSI_regions(WSI_images, output_addr, ome_name, tile_size=1024, start_i=0, start_j=0):
    start_index = [0, 0]
    image_shape = list(WSI_images.values())[0].shape
    print(ome_name, start_i, start_j, image_shape)
    while start_index[0] + tile_size <= image_shape[0]:
        while start_index[1] + tile_size <= image_shape[1]:
            dapi_tile = None
            if 'DAPI' in WSI_images.keys():
                dapi_tile = WSI_images['DAPI'][start_index[0]: start_index[0] + tile_size, start_index[1]: start_index[1] + tile_size]
            if (dapi_tile.any() and np.mean(dapi_tile) > 0.0) or not dapi_tile.any():
                for img_type, WSI_image in WSI_images.items():
                    tile = WSI_image[start_index[0]: start_index[0] + tile_size, start_index[1]: start_index[1] + tile_size]
                    tile = (imadjust(tile, 1, 0, 255)).astype(np.uint8)
                    Image.fromarray(tile).save(os.path.join(output_addr, ome_name + '_' +
                                                            str(start_i + start_index[0]) + '_' + str(start_j + start_index[1]) + '_' + img_type + '.png'))
            start_index[1] += tile_size
        start_index[1] = 0
        start_index[0] += tile_size


# input_dir = '/Users/pghahremani/Documents/test_dataset'
# output_dir = '/Users/pghahremani/Documents/results/'
# create_training_testing_dataset_from_given_directory(input_dir, output_dir,
#                                                      post_fix_names=['DAPI', 'hematoxylin', 'CD3', 'CD8', 'DAPI', 'FoxP3', 'PDL1', 'PanCK'],
#                                                      subsets={'train': 0.8, 'test': 0.1, 'val': 0.1})


# input_dir = '/Users/pghahremani/Documents/results/'
# output_dir = '/Users/pghahremani/Documents/augmented/'
# augment_created_dataset(input_dir, output_dir, aug_no=3, modality_types = ['DAPI', 'hematoxylin', 'CD3', 'CD8', 'DAPI', 'FoxP3', 'PDL1', 'PanCK'], tile_size=512)


# image_addr = '/Users/pghahremani/Downloads/Moffitt/mpif2/Case1_M1_1_0_FoxP3.png'
# image = Image.open(image_addr)
# adjusted_image1 = imadjust(np.array(image), gamma=0.6, c=0, d=255).astype(np.uint8)
# adjusted_image2 = imadjust(np.array(image), gamma=1.4, c=0, d=255).astype(np.uint8)
# cv2.imshow('original image', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
# cv2.imshow('adjusted image (gamma=0.6)', cv2.cvtColor(adjusted_image1, cv2.COLOR_RGB2BGR))
# cv2.imshow('adjusted image (gamma=1.4)', cv2.cvtColor(adjusted_image2, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)


# image_addr = '/Volumes/NadeemLab/Parmida/Projects-Datasets-BestPractices/DeepLIIF/images/target.png'
# image = Image.open(image_addr)
# deconvolved_stain = stain_deconvolution(np.array(image)).astype(np.uint8)
# cv2.imshow('Original image', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
# cv2.imshow('deconvolved image', cv2.cvtColor(deconvolved_stain, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)


# create_multi_channel_image('/nadeem_lab/Parmida/DeepLIIF/results/Travis_All_Markers/test_latest/images',
#                            '/nadeem_lab/NadeemLab/Parmida/DeepLIIF/results/Travis_All_Markers/test_latest/stacked3', stacked=True,
#                            image_channels_names={'real_A': ['IHC_R', 'IHC_G', 'IHC_B'],
#                                                  'real_B_1': ['DAPI_Orig'], 'fake_B_1': ['DAPI_Gen'],
#                                                  'real_B_2': ['CD3_Orig'], 'fake_B_2': ['CD3_Gen'],
#                                                  'real_B_3': ['CD68_Orig'], 'fake_B_3': ['CD68_Gen'],
#                                                  'real_B_4': ['Sox10_Orig'], 'fake_B_4': ['Sox10_Gen'],
#                                                  'real_B_5': ['PDL1_Orig'], 'fake_B_5': ['PDL1_Gen'],
#                                                  'real_B_6': ['CD8_Orig'], 'fake_B_6': ['CD8_Gen']})

# input_dir = ''
# output_dir = ''
# create_dataset_from_ome_multi_channel_image(input_dir, output_dir, tile_size=1024,
#                                             img_channel_modality={'DAPI': 0, 'PD1': 1, 'PDL1': 2, 'CD68': 3, 'CD8': 4,
#                                                                   'CD4': 6, 'FoxP3': 7, 'Sox10': 8, 'CD3': 9,
#                                                                   'IHC': [13, 12, 11]})
