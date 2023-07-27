# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import pickle, glob, cv2, imageio, os, sys, pdb, re, json, trimesh, copy, pdb, logging, multiprocessing, subprocess, joblib
import numpy as np
import ruamel.yaml

yaml = ruamel.yaml.YAML()
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../../")
from Utils import *

PROJ_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))
HO3D_ROOT = os.path.join(PROJ_ROOT, "datasets/HO3D_v3")
DEX_YCB_ROOT = os.path.join(PROJ_ROOT, "datasets/DexYcb")
HO_PIPE_ROOT = os.path.join(PROJ_ROOT, "datasets/HoPipe")


class YcbineoatReader:
    def __init__(self, video_dir, downscale=1, shorter_side=None):
        self.video_dir = video_dir
        self.downscale = downscale
        self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
        self.K = np.loadtxt(f"{video_dir}/cam_K.txt").reshape(3, 3)
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace(".png", "")
            self.id_strs.append(id_str)
        self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]

        if shorter_side is not None:
            self.downscale = shorter_side / min(self.H, self.W)

        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)
        self.K[:2] *= self.downscale

        self.gt_pose_files = sorted(glob.glob(f"{self.video_dir}/annotated_poses/*"))

        self.videoname_to_object = {
            "bleach0": "021_bleach_cleanser",
            "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
            "cracker_box_reorient": "003_cracker_box",
            "cracker_box_yalehand0": "003_cracker_box",
            "mustard0": "006_mustard_bottle",
            "mustard_easy_00_02": "006_mustard_bottle",
            "sugar_box1": "004_sugar_box",
            "sugar_box_yalehand0": "004_sugar_box",
            "tomato_soup_can_yalehand0": "005_tomato_soup_can",
        }

    def get_video_name(self):
        return self.video_dir.split("/")[-1]

    def __len__(self):
        return len(self.color_files)

    def get_gt_pose(self, i):
        try:
            pose = np.loadtxt(self.gt_pose_files[i]).reshape(4, 4)
            return pose
        except:
            logging.info("GT pose not found, return None")
            return None

    def get_color(self, i):
        color = imageio.imread(self.color_files[i])
        color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return color

    def get_mask(self, i):
        mask = cv2.imread(self.color_files[i].replace("rgb", "masks"), -1)
        if len(mask.shape) == 3:
            mask = (mask.sum(axis=-1) > 0).astype(np.uint8)
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return mask

    def get_depth(self, i):
        depth = cv2.imread(self.color_files[i].replace("rgb", "depth"), -1) / 1e3
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.K)
        return xyz_map

    def get_occ_mask(self, i):
        hand_mask_file = self.color_files[i].replace("rgb", "masks_hand")
        occ_mask = np.zeros((self.H, self.W), dtype=bool)
        if os.path.exists(hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(hand_mask_file, -1) > 0)

        right_hand_mask_file = self.color_files[i].replace("rgb", "masks_hand_right")
        if os.path.exists(right_hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(right_hand_mask_file, -1) > 0)

        occ_mask = cv2.resize(
            occ_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST
        )

        return occ_mask.astype(np.uint8)

    def get_gt_mesh(self):
        ob_name = self.videoname_to_object[self.get_video_name()]
        mesh = trimesh.load(
            f"/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/YCB_Video_Models/models/{ob_name}/textured_simple.obj"
        )
        return mesh


class Ho3dReader:
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.jpg"))
        meta_file = self.color_files[0].replace(".jpg", ".pkl").replace("rgb", "meta")
        self.K = pickle.load(open(meta_file, "rb"))["camMat"]

        self.id_strs = []
        for i in range(len(self.color_files)):
            id = os.path.basename(self.color_files[i]).split(".")[0]
            self.id_strs.append(id)

    def __len__(self):
        return len(self.color_files)

    def get_video_name(self):
        return os.path.dirname(os.path.abspath(self.color_files[0])).split("/")[-2]

    def get_mask(self, i):
        video_name = self.get_video_name()
        index = int(os.path.basename(self.color_files[i]).split(".")[0])
        mask = cv2.imread(f"{HO3D_ROOT}/masks_XMem/{video_name}/{index:05d}.png", -1)
        return mask

    def get_occ_mask(self, i):
        video_name = self.get_video_name()
        index = int(os.path.basename(self.color_files[i]).split(".")[0])
        mask = cv2.imread(
            f"{HO3D_ROOT}/masks_XMem/{video_name}_hand/{index:04d}.png", -1
        )
        return mask

    def get_gt_mesh(self):
        video2name = {
            "AP": "019_pitcher_base",
            "MPM": "010_potted_meat_can",
            "SB": "021_bleach_cleanser",
            "SM": "006_mustard_bottle",
        }
        video_name = self.get_video_name()
        for k in video2name:
            if video_name.startswith(k):
                ob_name = video2name[k]
                break
        mesh = trimesh.load(f"{HO3D_ROOT}/models/{ob_name}/textured_simple.obj")
        return mesh

    def get_depth(self, i):
        color = imageio.imread(self.color_files[i])
        depth_scale = 0.00012498664727900177
        depth = cv2.imread(
            self.color_files[i].replace(".jpg", ".png").replace("rgb", "depth"), -1
        )
        depth = (depth[..., 2] + depth[..., 1] * 256) * depth_scale
        return depth

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.K)
        return xyz_map

    def get_gt_pose(self, i):
        meta_file = self.color_files[i].replace(".jpg", ".pkl").replace("rgb", "meta")
        meta = pickle.load(open(meta_file, "rb"))
        ob_in_cam_gt = np.eye(4)
        if meta["objTrans"] is None:
            return None
        else:
            ob_in_cam_gt[:3, 3] = meta["objTrans"]
            ob_in_cam_gt[:3, :3] = cv2.Rodrigues(meta["objRot"].reshape(3))[0]
            ob_in_cam_gt = glcam_in_cvcam @ ob_in_cam_gt
        return ob_in_cam_gt


class DexYcbReader:
    def __init__(self, sequence_folder) -> None:
        self.sequence_folder = os.path.abspath(sequence_folder)
        self.calib_folder = os.path.abspath(
            os.path.join(self.sequence_folder, "../../calibration")
        )
        self.models_folder = os.path.abspath(
            os.path.join(self.sequence_folder, "../../models")
        )
        self.H = 480
        self.W = 640

        self.color_prefix = "color_{:06d}.jpg"
        self.depth_prefix = "aligned_depth_to_color_{:06d}.png"
        self.label_prefix = "labels_{:06d}.npz"

        # load meta_data
        data = self.load_data_from_yaml(os.path.join(self.sequence_folder, "meta.yml"))
        serials = data["serials"]
        self.num_frames = data["num_frames"]
        self.ycb_grasp_ind = data["ycb_grasp_ind"]
        self.ycb_ids = data["ycb_ids"]
        self.ycb_id = self.ycb_ids[self.ycb_grasp_ind]
        self.extr_file = os.path.join(
            self.calib_folder, f"extrinsics_{data['extrinsics']}/extrinsics.yml"
        )

        # load extrinsics
        data = self.load_data_from_yaml(self.extr_file)
        master = data["master"]
        self.serials = [master] + [s for s in serials if s != master]
        self.Ts = {
            key: np.array(
                [
                    [
                        data["extrinsics"][key][0],
                        data["extrinsics"][key][1],
                        data["extrinsics"][key][2],
                        data["extrinsics"][key][3],
                    ],
                    [
                        data["extrinsics"][key][4],
                        data["extrinsics"][key][5],
                        data["extrinsics"][key][6],
                        data["extrinsics"][key][7],
                    ],
                    [
                        data["extrinsics"][key][8],
                        data["extrinsics"][key][9],
                        data["extrinsics"][key][10],
                        data["extrinsics"][key][11],
                    ],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
            for key in data["extrinsics"]
        }

        # load intrinsics
        self.Ks = {
            s: self.load_K_from_yaml(
                os.path.join(self.calib_folder, f"intrinsics/{s}_{self.W}x{self.H}.yml")
            )
            for s in self.serials
        }

    def load_data_from_yaml(self, yaml_file, version=(1, 1)):
        yaml = ruamel.yaml.YAML()
        yaml.version = version
        with open(yaml_file, "r") as f:
            data = yaml.load(f)
        return data

    def load_K_from_yaml(self, yaml_file, camera_type="color"):
        data = self.load_data_from_yaml(yaml_file)
        K = np.array(
            [
                [data[camera_type]["fx"], 0, data[camera_type]["ppx"]],
                [0, data[camera_type]["fy"], data[camera_type]["ppy"]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        return K

    def get_color(self, serial, frame_id):
        color_file = os.path.join(
            self.sequence_folder, serial, self.color_prefix.format(frame_id)
        )
        color = cv2.imread(color_file)
        return color

    def get_depth(self, serial, frame_id):
        depth_file = os.path.join(
            self.sequence_folder, serial, self.depth_prefix.format(frame_id)
        )
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 1000.0
        return depth

    def get_mask(self, serial, frame_id):
        label_file = os.path.join(
            self.sequence_folder, serial, self.label_prefix.format(frame_id)
        )
        seg = np.load(label_file)["seg"]
        mask = seg == self.ycb_id
        return mask.astype(np.uint8)


class HoPipeReader:
    def __init__(self, sequence_folder):
        self.sequence_folder = os.path.abspath(sequence_folder)
        self.calib_folder = os.path.abspath(
            os.path.join(self.sequence_folder, "../../calibration")
        )
        self.mask_folder = os.path.abspath(
            os.path.join(self.sequence_folder, "data_processing/xmem/output")
        )

        # load meta_data
        data = self.load_data_from_yaml(os.path.join(self.sequence_folder, "meta.yml"))
        serials = data["realsense"]["rs_serials"]
        self.num_frames = data["num_frames"]
        self.H = data["realsense"]["rs_height"]
        self.W = data["realsense"]["rs_width"]
        self.extr_file = os.path.join(
            self.calib_folder, f"extrinsics/{data['calibration']['extrinsics_file']}"
        )

        # load extrinsics
        data = self.load_data_from_yaml(self.extr_file)
        master = data["rs_master"]
        self.serials = [master] + [s for s in serials if s != master]
        self.Ts = {
            key: np.array(
                [
                    [
                        data["extrinsics"][key]["rotation"][0],
                        data["extrinsics"][key]["rotation"][1],
                        data["extrinsics"][key]["rotation"][2],
                        data["extrinsics"][key]["translation"][0],
                    ],
                    [
                        data["extrinsics"][key]["rotation"][3],
                        data["extrinsics"][key]["rotation"][4],
                        data["extrinsics"][key]["rotation"][5],
                        data["extrinsics"][key]["translation"][1],
                    ],
                    [
                        data["extrinsics"][key]["rotation"][6],
                        data["extrinsics"][key]["rotation"][7],
                        data["extrinsics"][key]["rotation"][8],
                        data["extrinsics"][key]["translation"][2],
                    ],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
            for key in data["extrinsics"]
        }

        # load intrinsics
        self.Ks = {
            s: self.load_K_from_yaml(
                os.path.join(self.calib_folder, f"intrinsics/{s}.yml")
            )
            for s in self.serials
        }

    def load_data_from_yaml(self, yaml_file, version=(1, 1)):
        yaml = ruamel.yaml.YAML()
        yaml.version = version
        with open(yaml_file, "r") as f:
            data = yaml.load(f)
        return data

    def load_K_from_yaml(self, yaml_file, camera_type="color"):
        data = self.load_data_from_yaml(yaml_file)
        K = np.array(
            [
                [data[camera_type]["fx"], 0, data[camera_type]["ppx"]],
                [0, data[camera_type]["fy"], data[camera_type]["ppy"]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        return K

    def get_color(self, serial, i):
        color_file = os.path.join(self.sequence_folder, serial, f"color_{i:06d}.jpg")
        color = cv2.imread(color_file)
        return color

    def get_depth(self, serial, i):
        depth_file = os.path.join(self.sequence_folder, serial, f"depth_{i:06d}.png")
        depth = cv2.imread(depth_file, -1).astype(np.float32) / 1000.0
        return depth

    def get_mask(self, serial, i):
        mask_file = os.path.join(self.mask_folder, serial, f"color_{i:06d}.png")
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = mask > 0
        return mask.astype(np.uint8)
