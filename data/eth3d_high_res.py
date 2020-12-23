import os
import cv2
import numpy as np

import torch.utils.data as data

from utils.io_utils import load_cam, load_pfm, load_pair, cam_adjust_max_d, load_cam_eth3d
from utils.preproc import to_channel_first, resize, center_crop, image_net_center as center_image
from data.data_utils import dict_collate
from utils.utils import print_dict


class ETH3DHighRes(data.Dataset):

    def __init__(self, root, num_src, subset, read, transforms):
        self.root = root
        self.num_src = num_src
        self.read = read
        self.transforms = transforms
        self.subset = subset
        assert self.subset in ["training", "test"]
        
        # self.scene_names = ["courtyard", "delivery_area", "electro", "facade", "kicker", "meadow",
        #                     "office", "pipes", "playground", "relief", "relief_2", "terrace", "terrains"]
        # self.scene_idx = int(os.environ['SCAN'])
        #
        # self.pair_file = os.path.join(self.root, self.subset, "{}/pair.txt".format(self.scene_names[self.scene_idx]))
        # self.pair = load_pair(self.pair_file)

        self.scene = str(os.environ['SCENE'])
        self.pair_file = os.path.join(self.root, self.subset, "{}/pair.txt".format(self.scene))
        self.pair = load_pair(self.pair_file)
        
    def __len__(self):
        return len(self.pair)
        
    def __getitem__(self, i):
        ref_idx = i
        src_idxs = self.pair[ref_idx][:self.num_src]
        
        # ref, *srcs = [os.path.join(self.root, f'{self.subset}/{self.scene_names[self.scene_idx]}/images/{idx:08}.jpg') for idx in [ref_idx] + src_idxs]
        # ref_cam, *srcs_cam = [os.path.join(self.root, f'{self.subset}/{self.scene_names[self.scene_idx]}/cams/{idx:08}_cam.txt') for idx in [ref_idx] + src_idxs]
        
        ref, *srcs = [os.path.join(self.root, f'{self.subset}/{self.scene}/images/{idx:08}.jpg') for idx in [ref_idx] + src_idxs]
        ref_cam, *srcs_cam = [os.path.join(self.root, f'{self.subset}/{self.scene}/cams/{idx:08}_cam.txt') for idx in [ref_idx] + src_idxs]
        
        skip = 0

        sample = self.read({'ref':ref, 'ref_cam':ref_cam, 'srcs':srcs, 'srcs_cam':srcs_cam, 'skip':skip})
        for t in self.transforms:
            sample = t(sample)
        return sample


def read(filenames, max_d, interval_scale):
    ref_name, ref_cam_name, srcs_name, srcs_cam_name, skip = [filenames[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'skip']]
    ref, *srcs = [cv2.imread(fn) for fn in [ref_name] + srcs_name]
    ref_cam, *srcs_cam = [load_cam_eth3d(fn, max_d, interval_scale) for fn in [ref_cam_name] + srcs_cam_name]
    gt = np.zeros((ref.shape[0], ref.shape[1], 1))
    masks = [np.zeros((ref.shape[0], ref.shape[1], 1)) for i in range(len(srcs))]
    return {
        'ref': ref,
        'ref_cam': ref_cam,
        'srcs': srcs,
        'srcs_cam': srcs_cam,
        'gt': gt,
        'masks': masks,
        'skip': skip
    }


# input sample preprocessing     
def val_preproc(sample, preproc_args):
    ref, ref_cam, srcs, srcs_cam, gt, masks, skip = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks', 'skip']]

    ref, *srcs = [center_image(img) for img in [ref] + srcs]
    ref, ref_cam, srcs, srcs_cam, gt, masks = resize([ref, ref_cam, srcs, srcs_cam, gt, masks], preproc_args['resize_width'], preproc_args['resize_height'])
    ref, ref_cam, srcs, srcs_cam, gt, masks = center_crop([ref, ref_cam, srcs, srcs_cam, gt, masks], preproc_args['crop_width'], preproc_args['crop_height'])
    ref, *srcs, gt = to_channel_first([ref] + srcs + [gt])
    masks = to_channel_first(masks)

    srcs, srcs_cam, masks = [np.stack(arr_list, axis=0) for arr_list in [srcs, srcs_cam, masks]]

    return {
        'ref': ref,  # 3hw
        'ref_cam': ref_cam,  # 244
        'srcs': srcs,  # v3hw
        'srcs_cam': srcs_cam,  # v244
        'gt': gt,  # 1hw
        'masks': masks,  # v1hw
        'skip': skip  # scalar
    }


def get_val_loader(root, num_src, subset, preproc_args):
    dataset = ETH3DHighRes(
        root, num_src, subset,
        read=lambda filenames: read(filenames, preproc_args['max_d'], preproc_args['interval_scale']),
        transforms=[lambda sample: val_preproc(sample, preproc_args)]
    )
    loader = data.DataLoader(dataset, 1, collate_fn=dict_collate, shuffle=False)
    return dataset, loader     
        
# some test code here
if __name__ == "__main__":
    # dataset, loader =
    data_root = "/mnt/B/qiyh/ETH3D/"
    num_src = 7
    #subset = "training"
    interval_scale = 1.0
    max_d = 256
    resize_width, resize_height = 1920, 1280
    crop_width, crop_height = 1920, 1280

    os.environ["SCAN"] = '0'

    dataset, loader = get_val_loader(
        data_root, num_src, #subset,
        {
            'interval_scale': interval_scale,
            'max_d': max_d,
            'resize_width': resize_width,
            'resize_height': resize_height,
            'crop_width': crop_width,
            'crop_height': crop_height
        }
    )

    sample = dataset[0]
    print("dataset: {}".format(sample.keys()))

    print_dict(sample)
    # # input rgb image
    # import cv2
    # # winname = 'ref_view'
    # # cv2.namedWindow(winname, cv2.WINDOW_KEEPRATIO)
    # # cv2.imshow(winname, sample["ref"].transpose(1, 2, 0))
    # cv2.imshow("src_view", sample["srcs"][0].transpose(1, 2, 0))
    # cv2.waitKey(0)

    print("ref_cam: {}".format(sample["ref_cam"]))

