import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = '/ssd_4/my/data/mscr/' #opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True




import os
import warnings
import numpy as np
from tqdm import tqdm
from natsort import natsorted

from datetime import datetime
to_date   = lambda string: datetime.strptime(string, '%Y-%m-%d')
S1_LAUNCH = to_date('2014-04-03')

import rasterio
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
#from s2cloudless import S2PixelCloudDetector
#from util.detect_cloudshadow import get_cloud_mask, get_shadow_mask


# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS
def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    return tif

def read_img(tif):
    return tif.read().astype(np.float32)

def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img      = (img - oldMin) / oldRange
    return img

def process_MS(img, method):
    if method=='default':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img = rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img /= 2000                                        # project to [0,5], preserve global intensities (across patches)
    return img

def process_SAR(img, method):
    if method=='default':
        dB_min, dB_max = -25, 0                            # define a reasonable range of SAR dB
        img = np.clip(img, dB_min, dB_max)                 # intensity clipping to a global unified SAR dB range
        img = rescale(img, dB_min, dB_max)                 # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        # project SAR to [0, 2] range
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        img = np.concatenate([(2 * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0]) / (dB_max[0] - dB_min[0]))[None, ...],
                              (2 * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1]) / (dB_max[1] - dB_min[1]))[None, ...]], axis=0)
    return img

from feature_detectors import get_cloud_cloudshadow_mask
import numpy as np
import scipy
import scipy.signal as scisig





def normalized_difference(channel1, channel2):
    subchan = channel1 - channel2
    sumchan = channel1 + channel2
    sumchan[sumchan == 0] = 0.001  # checking for 0 divisions
    return subchan / sumchan


def get_shadow_mask(data_image):
    data_image = data_image / 10000.

    (ch, r, c) = data_image.shape
    shadowmask = np.zeros((r, c)).astype('float32')

    BB     = data_image[1]
    BNIR   = data_image[7]
    BSWIR1 = data_image[11]

    CSI = (BNIR + BSWIR1) / 2.

    t3 = 3/4 # cloud-score index threshold
    T3 = np.min(CSI) + t3 * (np.mean(CSI) - np.min(CSI))

    t4 = 5 / 6  # water-body index threshold
    T4 = np.min(BB) + t4 * (np.mean(BB) - np.min(BB))

    shadow_tf = np.logical_and(CSI < T3, BB < T4)

    shadowmask[shadow_tf] = -1
    shadowmask = scisig.medfilt2d(shadowmask, 5)

    return shadowmask


def get_cloud_mask(data_image, cloud_threshold, binarize=False, use_moist_check=False):
    '''Adapted from https://github.com/samsammurphy/cloud-masking-sentinel2/blob/master/cloud-masking-sentinel2.ipynb'''

    data_image = data_image / 10000.
    (ch, r, c) = data_image.shape

    # Cloud until proven otherwise
    score = np.ones((r, c)).astype('float32')
    # Clouds are reasonably bright in the blue and aerosol/cirrus bands.
    score = np.minimum(score, rescale(data_image[1], [0.1, 0.5]))
    score = np.minimum(score, rescale(data_image[0], [0.1, 0.3]))
    score = np.minimum(score, rescale((data_image[0] + data_image[10]), [0.4, 0.9]))
    score = np.minimum(score, rescale((data_image[3] + data_image[2] + data_image[1]), [0.2, 0.8]))

    if use_moist_check:
        # Clouds are moist
        ndmi = normalized_difference(data_image[7], data_image[11])
        score = np.minimum(score, rescale(ndmi, [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = normalized_difference(data_image[2], data_image[11])
    score = np.minimum(score, rescale(ndsi, [0.8, 0.6]))

    boxsize = 7
    box = np.ones((boxsize, boxsize)) / (boxsize ** 2)

    score = scipy.ndimage.morphology.grey_closing(score, size=(5, 5))
    score = scisig.convolve2d(score, box, mode='same')

    score = np.clip(score, 0.00001, 1.0)

    if binarize:
        score[score >= cloud_threshold] = 1
        score[score < cloud_threshold]  = 0

    return score


""" SEN12MSCRTS data loader class, inherits from torch.utils.data.Dataset

    IN: 
    root:               str, path to your copy of the SEN12MS-CR-TS data set
    split:              str, in [all | train | val | test]
    region:             str, [all | africa | america | asiaEast | asiaWest | europa]
    cloud_masks:        str, type of cloud mask detector to run on optical data, in []
    sample_type:        str, [generic | cloudy_cloudfree]
    n_input_samples:    int, number of input samples in time series
    rescale_method:     str, [default | resnet]
    min_cov:            float, in [0.0, 1.0]
    max_cov:            float, in [0.0, 1.0]
    import_data_path:   str, path to importing the suppl. file specifying what time points to load for input and output
    export_data_path:   str, path to export the suppl. file specifying what time points to load for input and output
    
    OUT:
    data_loader:        SEN12MSCRTS instance, implements an iterator that can be traversed via __getitem__(pdx),
                        which returns the pdx-th dictionary of patch-samples (whose structure depends on sample_type)
"""

class SEN12MSCRTS(Dataset):
    def __init__(self, root, split="all", region='all', cloud_masks='s2cloudless_mask', sample_type='cloudy_cloudfree', n_input_samples=3, rescale_method='default', min_cov=0.0, max_cov=1.0, import_data_path=None, export_data_path=None):
        
        self.root_dir = root   # set root directory which contains all ROI
        self.region   = region # region according to which the ROI are selected
        self.ROI      = {'ROIs1158': ['106'],
                         'ROIs1868': ['17', '36', '56', '73', '85', '100', '114', '119', '121', '126', '127', '139', '142', '143'],
                         'ROIs1970': ['20', '21', '35', '40', '57', '65', '71', '82', '83', '91', '112', '116', '119', '128', '132', '133', '135', '139', '142', '144', '149'],
                         'ROIs2017': ['8', '22', '25', '32', '49', '61', '63', '69', '75', '103', '108', '115', '116', '117', '130', '140', '146']}
        
        # define splits conform with SEN12MS-CR
        self.splits         = {}
        if self.region=='all':
            all_ROI             = [os.path.join(key, val) for key, vals in self.ROI.items() for val in vals]
            self.splits['test'] = [os.path.join('ROIs1868', '119'), os.path.join('ROIs1970', '139'), os.path.join('ROIs2017', '108'), os.path.join('ROIs2017', '63'), os.path.join('ROIs1158', '106'), os.path.join('ROIs1868', '73'), os.path.join('ROIs2017', '32'),
                                   os.path.join('ROIs1868', '100'), os.path.join('ROIs1970', '132'), os.path.join('ROIs2017', '103'), os.path.join('ROIs1868', '142'), os.path.join('ROIs1970', '20'), os.path.join('ROIs2017', '140')]  # official test split, across continents
            self.splits['val']  = [os.path.join('ROIs2017', '22'), os.path.join('ROIs1970', '65'), os.path.join('ROIs2017', '117'), os.path.join('ROIs1868', '127'), os.path.join('ROIs1868', '17')] # insert a validation split here
            self.splits['train']= [roi for roi in all_ROI if roi not in self.splits['val'] and roi not in self.splits['test']]  # all remaining ROI are used for training
        elif self.region=='africa':
            self.splits['test'] = [os.path.join('ROIs2017', '32'), os.path.join('ROIs2017', '140')]
            self.splits['val']  = [os.path.join('ROIs2017', '22')]
            self.splits['train']= [os.path.join('ROIs1970', '21'), os.path.join('ROIs1970', '35'), os.path.join('ROIs1970', '40'),
                                   os.path.join('ROIs2017', '8'), os.path.join('ROIs2017', '61'), os.path.join('ROIs2017', '75')]
        elif self.region=='america':
            self.splits['test'] = [os.path.join('ROIs1158', '106'), os.path.join('ROIs1970', '132')]
            self.splits['val']  = [os.path.join('ROIs1970', '65')]
            self.splits['train']= [os.path.join('ROIs1868', '36'), os.path.join('ROIs1868', '85'),
                                   os.path.join('ROIs1970', '82'), os.path.join('ROIs1970', '142'),
                                   os.path.join('ROIs2017', '49'), os.path.join('ROIs2017', '116')]
        elif self.region=='asiaEast':
            self.splits['test'] = [os.path.join('ROIs1868', '73'), os.path.join('ROIs1868', '119'), os.path.join('ROIs1970', '139')]
            self.splits['val']  = [os.path.join('ROIs2017', '117')]
            self.splits['train']= [os.path.join('ROIs1868', '114'), os.path.join('ROIs1868', '126'), os.path.join('ROIs1868', '143'), 
                                   os.path.join('ROIs1970', '116'), os.path.join('ROIs1970', '135'),
                                   os.path.join('ROIs2017', '25')]
        elif self.region=='asiaWest':
            self.splits['test'] = [os.path.join('ROIs1868', '100')]
            self.splits['val']  = [os.path.join('ROIs1868', '127')]
            self.splits['train']= [os.path.join('ROIs1970', '57'), os.path.join('ROIs1970', '83'), os.path.join('ROIs1970', '112'),
                                   os.path.join('ROIs2017', '69'), os.path.join('ROIs1970', '115'), os.path.join('ROIs1970', '130')]
        elif self.region=='europa':
            self.splits['test'] = [os.path.join('ROIs2017', '63'), os.path.join('ROIs2017', '103'), os.path.join('ROIs2017', '108'), os.path.join('ROIs1868', '142'), os.path.join('ROIs1970', '20')]
            self.splits['val']  = [os.path.join('ROIs1868', '17')]
            self.splits['train']= [os.path.join('ROIs1868', '56'), os.path.join('ROIs1868', '121'), os.path.join('ROIs1868', '139'),
                                   os.path.join('ROIs1970', '71'), os.path.join('ROIs1970', '91'), os.path.join('ROIs1970', '119'), os.path.join('ROIs1970', '128'), os.path.join('ROIs1970', '133'), os.path.join('ROIs1970', '144'), os.path.join('ROIs1970', '149'),
                                   os.path.join('ROIs2017', '146')]
        else: raise NotImplementedError

        self.splits["all"]  = self.splits["train"] + self.splits["test"] + self.splits["val"]
        self.split = split
        
        assert split in ['all', 'train', 'val', 'test'], "Input dataset must be either assigned as all, train, test, or val!"
        assert sample_type in ['generic', 'cloudy_cloudfree'], "Input data must be either generic or cloudy_cloudfree type!"
        assert cloud_masks in [None, 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'], "Unknown cloud mask type!"

        self.modalities     = ["S1", "S2"]
        self.time_points    = range(30)
        self.cloud_masks    = cloud_masks  # e.g. 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'
        self.sample_type    = sample_type if self.cloud_masks is not None else 'generic' # pick 'generic' or 'cloudy_cloudfree'
        self.n_input_t      = n_input_samples  # specifies the number of samples, if only part of the time series is used as an input

        #if self.cloud_masks in ['s2cloudless_map', 's2cloudless_mask']:
        #    self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
        #else: self.cloud_detector = None

        self.import_data_path = import_data_path
        self.export_data_path = export_data_path
        if self.export_data_path: self.data_pairs = {}

        if self.import_data_path:
            # fetch time points as specified in the imported file, expects arguments are set accordingly
            if os.path.isdir(self.import_data_path):
                import_here = os.path.join(self.import_data_path, f'{self.n_input_t}_{self.split}_{self.cloud_masks}.npy')
            else:
                import_here = self.import_data_path
            self.data_pairs = np.load(import_here, allow_pickle=True).item()
            print(f'Importing data pairings for split {self.split} from {import_here}.')

        self.paths          = self.get_paths()
        self.n_samples      = len(self.paths)

        # raise a warning that no data has been found
        if not self.n_samples: self.throw_warn()

        self.method         = rescale_method
        self.min_cov, self.max_cov = min_cov, max_cov

    def throw_warn(self):
        warnings.warn("""No data samples found! Please use the following directory structure:
                        
        path/to/your/SEN12MSCRTS/directory:
        ├───ROIs1158
        ├───ROIs1868
        ├───ROIs1970
        │   ├───20
        │   ├───21
        │   │   ├───S1
        │   │   └───S2
        │   │       ├───0
        │   │       ├───1
        │   │       │   └─── ... *.tif files
        │   │       └───30
        │   ...
        └───ROIs2017
                        
        Note: the data is provided by ROI geo-spatially separated and sensor modalities individually.
        You can simply merge the downloaded & extracted archives' subdirectories via 'mv */* .' in the parent directory
        to obtain the required structure specified above, which the data loader expects.
        """)

    # indexes all patches contained in the current data split
    def get_paths(self):  # assuming for the same ROI+num, the patch numbers are the same
        print(f'\nProcessing paths for {self.split} split of region {self.region}')

        paths = []
        for roi_dir, rois in self.ROI.items():
            for roi in tqdm(rois):
                roi_path = os.path.join(self.root_dir, roi_dir, roi)
                # skip non-existent ROI or ROI not part of the current data split
                if not os.path.isdir(roi_path) or os.path.join(roi_dir, roi) not in self.splits[self.split]: continue
                path_s1_t, path_s2_t = [], [],
                for tdx in self.time_points:
                    # working with directory under time stamp tdx
                    path_s1_complete = os.path.join(roi_path, self.modalities[0], str(tdx))
                    path_s2_complete = os.path.join(roi_path, self.modalities[1], str(tdx))

                    # same as complete paths, truncating root directory's path
                    path_s1 = os.path.join(roi_dir, roi, self.modalities[0], str(tdx))
                    path_s2 = os.path.join(roi_dir, roi, self.modalities[1], str(tdx))

                    # get list of files which contains all the patches at time tdx
                    s1_t = natsorted([os.path.join(path_s1, f) for f in os.listdir(path_s1_complete) if (os.path.isfile(os.path.join(path_s1_complete, f)) and ".tif" in f)])
                    s2_t = natsorted([os.path.join(path_s2, f) for f in os.listdir(path_s2_complete) if (os.path.isfile(os.path.join(path_s2_complete, f)) and ".tif" in f)])

                    # same number of patches
                    assert len(s1_t) == len(s2_t)

                    # sort via file names according to patch number and store
                    path_s1_t.append(s1_t)
                    path_s2_t.append(s2_t)

                # for each patch of the ROI, collect its time points and make this one sample
                for pdx in range(len(path_s1_t[0])):
                    sample = {"S1": [path_s1_t[tdx][pdx] for tdx in self.time_points],
                              "S2": [path_s2_t[tdx][pdx] for tdx in self.time_points]}
                    paths.append(sample)

        return paths
    def lp(self, mask):
        mask[mask != 0] = 1
        return mask
    def __getitem__(self, pdx):  # get the time series of one patch

        # get images
        s1_tif          = [read_tif(os.path.join(self.root_dir, img)) for img in self.paths[pdx]['S1']]
        s2_tif          = [read_tif(os.path.join(self.root_dir, img)) for img in self.paths[pdx]['S2']]
        coord           = [list(tif.bounds) for tif in s2_tif]
        s1              = [process_SAR(read_img(img), self.method) for img in s1_tif]
        s2              = [read_img(img) for img in s2_tif]  # note: pre-processing happens after cloud detection
        masks           = [self.lp(get_cloud_cloudshadow_mask(img, 0.2)) for img in s2]
        # get statistics and additional meta information
        coverage    = [np.mean(mask) for mask in masks]
        s1_dates    = [to_date(img.split('/')[-1].split('_')[5]) for img in self.paths[pdx]['S1']]
        s2_dates    = [to_date(img.split('/')[-1].split('_')[5]) for img in self.paths[pdx]['S2']]
        s1_td       = [(date-S1_LAUNCH).days for date in s1_dates]
        s2_td       = [(date-S1_LAUNCH).days for date in s2_dates]

        # generate data of ((cloudy_t1, cloudy_t2, ..., cloudy_tn), cloud-free) pairings
        # note: filtering the data (e.g. according to cloud coverage etc) and may only use a fraction of the data set
        #       if you wish to train or test on additional samples, then this filtering needs to be adjusted
        if self.sample_type == 'cloudy_cloudfree':
            if self.import_data_path:
                # read indices
                inputs_idx    = self.data_pairs[pdx]['input']
                cloudless_idx = self.data_pairs[pdx]['target']
                target_s1, target_s2, target_mask = np.array(s1)[cloudless_idx], np.array(s2)[cloudless_idx], np.array(masks)[cloudless_idx]
                input_s1, input_s2, input_masks   = np.array(s1)[inputs_idx], np.array(s2)[inputs_idx], np.array(masks)[inputs_idx]
                coverage_match = True

            else:  # sample custom time points from the current patch space in the current split
                # sort observation indices according to cloud coverage, ascendingly
                coverage_idx = np.argsort(coverage)
                cloudless_idx = coverage_idx[0]
                # take the (earliest, in case of draw) least cloudy time point as target
                target_s1, target_s2, target_mask = np.array(s1)[cloudless_idx], np.array(s2)[cloudless_idx], np.array(masks)[cloudless_idx]
                # take the first n_input_t samples with cloud coverage e.g. in [0.1, 0.5], ...
                inputs_idx = [pdx for pdx, perc in enumerate(coverage) if perc >= self.min_cov and perc <= self.max_cov][:self.n_input_t]
                coverage_match = True  # assume the requested amount of cloud coverage is met
                
                if len(inputs_idx) < self.n_input_t:
                    # ... if not exists then take the first n_input_t samples (except target patch)
                    inputs_idx = [pdx for pdx in range(len(coverage)) if pdx!=cloudless_idx][:self.n_input_t]
                    coverage_match = False  # flag input samples that didn't meet the required cloud coverage
                input_s1, input_s2, input_masks = np.array(s1)[inputs_idx], np.array(s2)[inputs_idx], np.array(masks)[inputs_idx]

                if self.export_data_path:
                    # performs repeated writing to file, only use this for processes dedicated for exporting
                    # and if so, only use a single thread of workers (--num_threads 1), this ain't thread-safe
                    self.data_pairs[pdx] = {'input': inputs_idx, 'target': cloudless_idx,
                                            'paths': {'input': {'S1': [self.paths[pdx]['S1'][idx] for idx in inputs_idx],
                                                                'S2': [self.paths[pdx]['S2'][idx] for idx in inputs_idx]},
                                                      'output': {'S1': self.paths[pdx]['S1'][cloudless_idx],
                                                                 'S2': self.paths[pdx]['S2'][cloudless_idx]}}}
                    if os.path.isdir(self.export_data_path):
                        export_here = os.path.join(self.export_data_path, f'{self.n_input_t}_{self.split}_{self.cloud_masks}.npy')
                    else:
                        export_here = self.export_data_path
                    np.save(export_here, self.data_pairs)

            sample = {'input': {'S1': list(input_s1),
                                'S2': [process_MS(img, self.method) for img in input_s2],
                                'masks': list(input_masks),
                                'coverage': [np.mean(mask) for mask in input_masks],
                                'S1 TD': [s1_td[idx] for idx in inputs_idx],
                                'S2 TD': [s2_td[idx] for idx in inputs_idx],
                                'S1 path': [os.path.join(self.root_dir, self.paths[pdx]['S1'][idx]) for idx in inputs_idx],
                                'S2 path': [os.path.join(self.root_dir, self.paths[pdx]['S2'][idx]) for idx in inputs_idx],
                                'coord': [coord[idx] for idx in inputs_idx],
                                },
                      'target': {'S1': [target_s1],
                                 'S2': [process_MS(target_s2, self.method)],
                                 'masks': [target_mask],
                                 'coverage': [np.mean(target_mask)],
                                 'S1 TD': [s1_td[cloudless_idx]],
                                 'S2 TD': [s2_td[cloudless_idx]],
                                 'S1 path': [os.path.join(self.root_dir, self.paths[pdx]['S1'][cloudless_idx])],
                                 'S2 path': [os.path.join(self.root_dir, self.paths[pdx]['S2'][cloudless_idx])],
                                 'coord': [coord[cloudless_idx]],
                                 },
                       'coverage bin': coverage_match
                      }

        elif self.sample_type == 'generic':  # this returns the whole, unfiltered sequence of S1 & S2 observations
            sample = {'S1': s1,
                      'S2': [process_MS(img, self.method) for img in s2],
                      'masks': masks,
                      'coverage': coverage,
                      'S1 TD': s1_td,
                      'S2 TD': s2_td,
                      'S1 path': [os.path.join(self.root_dir, self.paths[pdx]['S1'][idx]) for idx in self.time_points],
                      'S2 path': [os.path.join(self.root_dir, self.paths[pdx]['S2'][idx]) for idx in self.time_points],
                      'coord': coord
                      }
        return sample

    def __len__(self):
        # length of generated list
        #print(self.n_samples)
        return self.n_samples



""" SEN12MSCR data loader class, inherits from torch.utils.data.Dataset

    IN: 
    root:               str, path to your copy of the SEN12MS-CR-TS data set
    split:              str, in [all | train | val | test]
    region:             str, [all | africa | america | asiaEast | asiaWest | europa]
    cloud_masks:        str, type of cloud mask detector to run on optical data, in []
    sample_type:        str, [generic | cloudy_cloudfree]
    n_input_samples:    int, number of input samples in time series
    rescale_method:     str, [default | resnet]
    
    OUT:
    data_loader:        SEN12MSCRTS instance, implements an iterator that can be traversed via __getitem__(pdx),
                        which returns the pdx-th dictionary of patch-samples (whose structure depends on sample_type)
"""



import numpy as np
import random
#from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
#from data.image_folder import make_dataset
#from data.dataLoader import SEN12MSCRTS
import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

class Sen12mscrtsDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=100000 , new_dataset_option=2.0)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)

        #if opt.alter_initial_model or opt.benchmark_resnet_model:
        #    self.rescale_method = 'resnet' # rescale SAR to [0,2] and optical to [0,5]
        #else:
        self.rescale_method = 'default' # rescale all to [-1,1] (gets rescaled to [0,1])

        self.opt 			= opt
        self.dataroot = '/ssd_4/my/data/mscr/'
        dir_SEN12MSCR   = '/ssd_4/my/data/mscr/'
        #sen12mscr       = SEN12MSCRTS(dir_SEN12MSCR, split='train', region='all')
        self.data_loader 	= SEN12MSCRTS(dir_SEN12MSCR, split='train', region='all')#SEN12MSCRTS(opt.dataroot, split=opt.input_type, region=opt.region, cloud_masks=opt.cloud_masks, sample_type=opt.sample_type, n_input_samples=opt.n_input_samples, rescale_method=self.rescale_method, min_cov=opt.min_cov, max_cov=opt.max_cov, import_data_path=opt.import_data_path, export_data_path=opt.export_data_path)
        self.max_bands		= 13

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        
        # call data loader to get item
        cloudy_cloudfree = self.data_loader.__getitem__(index)

        if self.opt.include_S1:
            input_channels = [i for i in range(self.max_bands)]
            
            # for each input sample, collect the SAR data
            A_S1 = []
            for i in range(self.opt.n_input_samples):
                A_S1_01 = cloudy_cloudfree['input']['S1'][i]

                if self.rescale_method == 'default':
                    A_S1.append((A_S1_01 * 2) - 1)  # rescale from [0,1] to [-1,+1]
                elif self.rescale_method == 'resnet':
                    A_S1.append(A_S1_01)  # no need to rescale, keep at [0,2]
                    
            # fetch the target S1 image (and optionally rescale)
            B_S1_01 = cloudy_cloudfree['target']['S1'][0]
            if self.rescale_method == 'default':
                B_S1 = (B_S1_01 * 2) - 1  # rescale from [0,1] to [-1,+1]
            elif self.rescale_method == 'resnet':
                B_S1 = B_S1_01  # no need to rescale, keep at [0,2]
        
        else: # not containing any S1
            assert self.opt.input_nc <= self.max_bands, Exception("MS input channel number larger than 13 (S1 not included)!")
            input_channels = [i for i in range(self.opt.input_nc)]

        # use only NIR+BGR channels when training STGAN
        #if self.opt.model == "temporal_branched_ir_modified": input_channels = [7, 1, 2, 3]

        A_S2, A_S2_mask = [], []

        if self.opt.in_only_S1:  # using only S1 input
            input_channels = [i for i in range(self.max_bands)]
            for i in range(self.opt.n_input_samples):
                A_S2_01 = cloudy_cloudfree['input']['S1'][i]
                if self.rescale_method == 'default':
                    A_S2.append((A_S2_01 * 2) - 1)   # rescale from [0,1] to [-1,+1]
                elif self.rescale_method == 'resnet':
                    A_S2.append(A_S2_01)  # no need to rescale, keep at [0,5]
                A_S2_mask.append(cloudy_cloudfree['target']['masks'][0].reshape((1, 256, 256)))
        else: # this is the typical case
            for i in range(self.opt.n_input_samples):
                A_S2_01 = cloudy_cloudfree['input']['S2'][i][input_channels]
                if self.rescale_method == 'default':
                    A_S2.append((A_S2_01 * 2) - 1)  # rescale from 0,1 to -1,+1
                elif self.rescale_method == 'resnet':
                    A_S2.append(A_S2_01)  # no need to rescale, keep at [0,5]
                A_S2_mask.append(cloudy_cloudfree['input']['masks'][i].reshape((1, 256, 256)))

        # get the target cloud-free optical image
        B_01   = cloudy_cloudfree['target']['S2'][0]
        #if self.opt.output_nc == 4: B_01 = B_01[input_channels]
        if self.rescale_method == 'default':
            B = (B_01 * 2) - 1  # rescale from [0,1] to [-1,+1]
        elif self.rescale_method == 'resnet':
            B = B_01  # no need to rescale, keep at [0,5]
        B_mask = cloudy_cloudfree['target']['masks'][0].reshape((1, 256, 256))
        image_path = cloudy_cloudfree['target']['S2 path']
        
        coverage_bin = True
        if "coverage bin" in cloudy_cloudfree: coverage_bin = cloudy_cloudfree["coverage bin"]

        if self.opt.include_S1:
            return {'A_S1': A_S1, 'A_S2': A_S2, 'A_mask': A_S2_mask, 'B': B, 'B_S1': B_S1, 'B_mask': B_mask, 'image_path': image_path, "coverage_bin": coverage_bin}
        else:
            return {'A_S2': A_S2, 'A_mask': A_S2_mask, 'B': B, 'B_mask': B_mask, 'image_path': image_path, "coverage_bin": coverage_bin}

    def __len__(self):
        """Return the total number of images."""
        return len(self.data_loader)
		
		
class Sen12mscrtsDataset_test(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=100000 , new_dataset_option=2.0)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)

        #if opt.alter_initial_model or opt.benchmark_resnet_model:
        #    self.rescale_method = 'resnet' # rescale SAR to [0,2] and optical to [0,5]
        #else:
        self.rescale_method = 'default' # rescale all to [-1,1] (gets rescaled to [0,1])

        self.opt 			= opt
        self.dataroot = '/ssd_4/my/data/mscr/'
        dir_SEN12MSCR   = '/ssd_4/my/data/mscr/'
        #sen12mscr       = SEN12MSCRTS(dir_SEN12MSCR, split='train', region='all')
        self.data_loader 	= SEN12MSCRTS(dir_SEN12MSCR, split='test', region='all')#SEN12MSCRTS(opt.dataroot, split=opt.input_type, region=opt.region, cloud_masks=opt.cloud_masks, sample_type=opt.sample_type, n_input_samples=opt.n_input_samples, rescale_method=self.rescale_method, min_cov=opt.min_cov, max_cov=opt.max_cov, import_data_path=opt.import_data_path, export_data_path=opt.export_data_path)
        self.max_bands		= 13

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        
        # call data loader to get item
        cloudy_cloudfree = self.data_loader.__getitem__(index)

        if self.opt.include_S1:
            input_channels = [i for i in range(self.max_bands)]
            
            # for each input sample, collect the SAR data
            A_S1 = []
            for i in range(self.opt.n_input_samples):
                A_S1_01 = cloudy_cloudfree['input']['S1'][i]

                if self.rescale_method == 'default':
                    A_S1.append((A_S1_01 * 2) - 1)  # rescale from [0,1] to [-1,+1]
                elif self.rescale_method == 'resnet':
                    A_S1.append(A_S1_01)  # no need to rescale, keep at [0,2]
                    
            # fetch the target S1 image (and optionally rescale)
            B_S1_01 = cloudy_cloudfree['target']['S1'][0]
            if self.rescale_method == 'default':
                B_S1 = (B_S1_01 * 2) - 1  # rescale from [0,1] to [-1,+1]
            elif self.rescale_method == 'resnet':
                B_S1 = B_S1_01  # no need to rescale, keep at [0,2]
        
        else: # not containing any S1
            assert self.opt.input_nc <= self.max_bands, Exception("MS input channel number larger than 13 (S1 not included)!")
            input_channels = [i for i in range(self.opt.input_nc)]

        # use only NIR+BGR channels when training STGAN
        #if self.opt.model == "temporal_branched_ir_modified": input_channels = [7, 1, 2, 3]

        A_S2, A_S2_mask = [], []

        if self.opt.in_only_S1:  # using only S1 input
            input_channels = [i for i in range(self.max_bands)]
            for i in range(self.opt.n_input_samples):
                A_S2_01 = cloudy_cloudfree['input']['S1'][i]
                if self.rescale_method == 'default':
                    A_S2.append((A_S2_01 * 2) - 1)   # rescale from [0,1] to [-1,+1]
                elif self.rescale_method == 'resnet':
                    A_S2.append(A_S2_01)  # no need to rescale, keep at [0,5]
                A_S2_mask.append(cloudy_cloudfree['target']['masks'][0].reshape((1, 256, 256)))
        else: # this is the typical case
            for i in range(self.opt.n_input_samples):
                A_S2_01 = cloudy_cloudfree['input']['S2'][i][input_channels]
                if self.rescale_method == 'default':
                    A_S2.append((A_S2_01 * 2) - 1)  # rescale from 0,1 to -1,+1
                elif self.rescale_method == 'resnet':
                    A_S2.append(A_S2_01)  # no need to rescale, keep at [0,5]
                A_S2_mask.append(cloudy_cloudfree['input']['masks'][i].reshape((1, 256, 256)))

        # get the target cloud-free optical image
        B_01   = cloudy_cloudfree['target']['S2'][0]
        #if self.opt.output_nc == 4: B_01 = B_01[input_channels]
        if self.rescale_method == 'default':
            B = (B_01 * 2) - 1  # rescale from [0,1] to [-1,+1]
        elif self.rescale_method == 'resnet':
            B = B_01  # no need to rescale, keep at [0,5]
        B_mask = cloudy_cloudfree['target']['masks'][0].reshape((1, 256, 256))
        image_path = cloudy_cloudfree['target']['S2 path']
        
        coverage_bin = True
        if "coverage bin" in cloudy_cloudfree: coverage_bin = cloudy_cloudfree["coverage bin"]

        if self.opt.include_S1:
            return {'A_S1': A_S1, 'A_S2': A_S2, 'A_mask': A_S2_mask, 'B': B, 'B_S1': B_S1, 'B_mask': B_mask, 'image_path': image_path, "coverage_bin": coverage_bin}
        else:
            return {'A_S2': A_S2, 'A_mask': A_S2_mask, 'B': B, 'B_mask': B_mask, 'image_path': image_path, "coverage_bin": coverage_bin}

    def __len__(self):
        """Return the total number of images."""
        return len(self.data_loader)