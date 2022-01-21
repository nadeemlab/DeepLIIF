from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import cv2
import numpy as np
import csv
from numba import cuda

from Segmentation_Metrics import compute_segmentation_metrics
from fid_official_tf import calculate_fid_given_paths
from inception_score import calculate_inception_score

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage import img_as_float, io, measure
from skimage.color import rgb2gray
import collections
from swd import compute_swd

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--gt_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--model_name', type=str, required=False, default='DeepLIIF')
parser.add_argument('--mode', type=str, default='Segmentation',
                    help='Mode of the statistics computation including Segmentation, ImageSynthesis, All')
parser.add_argument('--raw_segmentation', action='store_true')
parser.add_argument('--device', type=str, default='cuda', help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of processes to use for data loading')
parser.add_argument('--image_types', type=str, default='Hema,DAPI,Lap2,Marker')


class Statistics:
    def __init__(self, args):
        self.gt_path = args.gt_path
        self.model_path = args.model_path
        self.output_path = args.output_path
        self.model_name = args.model_name
        self.mode = args.mode
        self.raw_segmentation = args.raw_segmentation
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = args.device
        self.image_types = args.image_types.replace(' ', '').split(',')

        # Image Similarity Metrics
        self.inception_avg = collections.defaultdict(float)
        self.inception_std = collections.defaultdict(float)

        self.mse_avg = collections.defaultdict(float)
        self.mse_std = collections.defaultdict(float)

        self.ssim_avg = collections.defaultdict(float)
        self.ssim_std = collections.defaultdict(float)

        self.fid_value = collections.defaultdict(float)
        self.swd_value = collections.defaultdict(float)

        self.all_info = {}
        self.all_info['Model'] = self.model_name

        # Segmentation Metrics
        self.segmentation_metrics = collections.defaultdict(float)
        self.segmentation_info = None

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def compute_mse_ssim_scores(self):
        for img_type in self.image_types:
            images = os.listdir(self.model_path)
            mse_arr = []
            ssim_arr = []
            # mse_info = []
            for img_name in images:
                if img_type in img_name:
                    orig_img = img_as_float(rgb2gray(io.imread(os.path.join(self.gt_path, img_name))))
                    mask_img = img_as_float(rgb2gray(io.imread(os.path.join(self.model_path, img_name))))

                    mse_mask = mean_squared_error(orig_img, mask_img)
                    ssim_mask = ssim(orig_img, mask_img, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)

                    mse_arr.append(mse_mask)
                    ssim_arr.append(ssim_mask)
                    # mse_info.append({'image_name': img_name, 'image_type':img_type, 'mse': mse_mask, 'ssim': ssim_mask})
            # self.write_list_to_csv(mse_info, mse_info[0].keys(),
            #                        filename='inference_info_' + img_type + '_' + self.model_name + '.csv')
            self.mse_avg[img_type], self.mse_std[img_type] = np.mean(mse_arr), np.std(mse_arr)
            self.ssim_avg[img_type], self.ssim_std[img_type] = np.mean(ssim_arr), np.std(ssim_arr)

    def compute_inception_score(self):
        for img_type in self.image_types:
            images = os.listdir(self.model_path)
            real_images_array = []
            for img in images:
                if img_type in img:
                    image = cv2.imread(os.path.join(self.model_path, img))
                    image = cv2.resize(image, (299, 299))
                    real_images_array.append(image)
            real_images_array = np.array(real_images_array)
            self.inception_avg[img_type], self.inception_std[img_type] = calculate_inception_score(real_images_array)

    def compute_fid_score(self):
        for img_type in self.image_types:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            self.fid_value[img_type] = calculate_fid_given_paths([self.gt_path, self.model_path], None, low_profile=False)
            print("FID: ", self.fid_value[img_type])
            # self.fid_value[img_type] = calculate_fid_given_paths(paths=[self.gt_path, self.model_path], batch_size=self.batch_size, dims=self.fid_dims, num_workers=self.num_workers, mod_type='_' + img_type)

            device = cuda.get_current_device()
            device.reset()

    def compute_swd(self):
        for img_type in self.image_types:
            orig_images = []
            mask_images = []
            images = os.listdir(self.model_path)
            for img_name in images:
                if img_type in img_name:
                    orig_img = cv2.cvtColor(cv2.imread(os.path.join(self.gt_path, img_name)), cv2.COLOR_BGR2RGB)
                    mask_img = cv2.cvtColor(cv2.imread(os.path.join(self.model_path, img_name)), cv2.COLOR_BGR2RGB)
                    orig_images.append(orig_img)
                    mask_images.append(mask_img)

            self.swd_value[img_type] = compute_swd(np.array(orig_images), np.array(mask_images), self.device)

    def compute_image_similarity_metrics(self):
        self.compute_mse_ssim_scores()
        print('SSIM Computed')
        self.compute_inception_score()
        print('inception Computed')
        self.compute_fid_score()
        print('fid Computed')
        self.compute_swd()
        print('swd Computed')

        for key in self.mse_avg:
            self.all_info[key + '_' + 'MSE_avg'] = self.mse_avg[key]
            self.all_info[key + '_' + 'MSE_std'] = self.mse_std[key]
            self.all_info[key + '_' + 'ssim_avg'] = self.ssim_avg[key]
            self.all_info[key + '_' + 'ssim_std'] = self.ssim_std[key]
            self.all_info[key + '_' + 'inception_avg'] = self.inception_avg[key]
            self.all_info[key + '_' + 'inception_std'] = self.inception_std[key]
            self.all_info[key + '_' + 'fid_value'] = self.fid_value[key]
            self.all_info[key + '_' + 'swd_value'] = self.swd_value[key]

    def compute_IHC_scoring(self):
        images = os.listdir(self.gt_path)
        IHC_info = []
        for img in images:
            gt_image = cv2.cvtColor(cv2.imread(os.path.join(self.gt_path, img)), cv2.COLOR_BGR2RGB)
            if 'DeepLIIF' in self.model_name:
                mask_image = cv2.cvtColor(cv2.imread(os.path.join(self.model_path, img.replace('_Seg', '_Seg_Refined'))), cv2.COLOR_BGR2RGB)
            else:
                mask_image = cv2.cvtColor(cv2.imread(os.path.join(self.model_path, img)), cv2.COLOR_BGR2RGB)
            gt_image[gt_image < 10] = 0
            label_image_red_gt = measure.label(gt_image[:, :, 0], background=0)
            label_image_blue_gt = measure.label(gt_image[:, :, 2], background=0)
            number_of_positive_cells_gt = (len(np.unique(label_image_red_gt)) - 1)
            number_of_negative_cells_gt = (len(np.unique(label_image_blue_gt)) - 1)
            number_of_all_cells_gt = number_of_positive_cells_gt + number_of_negative_cells_gt
            gt_IHC_score = number_of_positive_cells_gt / number_of_all_cells_gt if number_of_all_cells_gt > 0 else 0

            mask_image[mask_image < 10] = 0
            label_image_red_mask = measure.label(mask_image[:, :, 0], background=0)
            label_image_blue_mask = measure.label(mask_image[:, :, 2], background=0)
            number_of_positive_cells_mask = (len(np.unique(label_image_red_mask)) - 1)
            number_of_negative_cells_mask = (len(np.unique(label_image_blue_mask)) - 1)
            number_of_all_cells_mask = number_of_positive_cells_mask + number_of_negative_cells_mask
            mask_IHC_score = number_of_positive_cells_mask / number_of_all_cells_mask if number_of_all_cells_mask > 0 else 0
            diff = abs(gt_IHC_score * 100 - mask_IHC_score * 100)
            IHC_info.append({'Model': self.model_name, 'Sample': img, 'Diff_IHC_Score': diff})
        self.write_list_to_csv(IHC_info, IHC_info[0].keys(),
                               filename='IHC_Scoring_info_' + self.mode + '_' + self.model_name + '.csv')

    def compute_segmentation_metrics(self):
        # max_dice = [0, 0, 0]
        # max_AJI = [0, 0, 0]
        # for thresh in range(60, 150, 10):
        #     for noise_size in range(10, 80, 20):
        thresh = 100
        boundary_thresh = 100
        noise_size = 50
        print(thresh, noise_size)
        self.segmentation_info, self.segmentation_metrics = compute_segmentation_metrics(self.gt_path, self.model_path, self.model_name, image_size=512, thresh=thresh, boundary_thresh=boundary_thresh, small_object_size=noise_size, raw_segmentation=self.raw_segmentation)
        self.write_list_to_csv(self.segmentation_info, self.segmentation_info[0].keys(),
                               filename='segmentation_info_' + self.mode + '_' + self.model_name + '_' + str(thresh) + '_' + str(noise_size) + '.csv')
        for key in self.segmentation_metrics:
            self.all_info[key] = self.segmentation_metrics[key]
            print(key, self.all_info[key])
        print('-------------------------------------------------------')

    def create_all_info(self):
        self.write_dict_to_csv(self.all_info, list(self.all_info.keys()), filename='metrics_' + self.mode + '_' + self.model_name + '.csv')

    def compute_statistics(self):
        self.compute_image_similarity_metrics()
        self.compute_segmentation_metrics()
        self.create_all_info()

    def write_dict_to_csv(self, info_dict, csv_columns, filename='info.csv'):
        print('Writing in csv')
        info_csv = open(os.path.join(self.output_path, filename), 'w')
        writer = csv.DictWriter(info_csv, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(info_dict)

    def write_list_to_csv(self, info_dict, csv_columns, filename='info.csv'):
        print('Writing in csv')
        info_csv = open(os.path.join(self.output_path, filename), 'w')
        writer = csv.DictWriter(info_csv, fieldnames=csv_columns)
        writer.writeheader()
        for data in info_dict:
            writer.writerow(data)


if __name__ == '__main__':
    args = parser.parse_args()
    stat = Statistics(args)
    print(stat.mode)
    if stat.mode == 'All':
        stat.compute_statistics()
        stat.compute_IHC_scoring()
    elif stat.mode == 'Segmentation':
        stat.compute_segmentation_metrics()
    elif stat.mode == 'ImageSynthesis':
        stat.compute_image_similarity_metrics()
