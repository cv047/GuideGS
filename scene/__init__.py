#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
# from scene.gaussian_model_xyz import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.virtual_poses import interpolate_virtual_poses, interpolate_virtual_poses_bspline_with_match,generate_offset_camera_poses
from scene.cameras import PseudoCamera

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}
        self.match_data = {}

        print("os.path.join(args.source_path:", os.path.join(args.source_path))
        if "Waymo" in args.source_path:
            scene_info = sceneLoadTypeCallbacks["Waymo"](args.source_path, args.images, args.eval)
        elif "basketball" in args.source_path:
            scene_info = sceneLoadTypeCallbacks["Basketball"](args.source_path, args.images, args.eval)
        elif "Scannet" in args.source_path:
            scene_info = sceneLoadTypeCallbacks["Scannet"](args.source_path, args.images, args.eval)
        elif "Tanks" in args.source_path:
            scene_info = sceneLoadTypeCallbacks["Tanks"](args.source_path, args.images, args.eval)
        elif "360" in args.source_path:
            scene_info = sceneLoadTypeCallbacks["Mip360"](args.source_path, args.images, args.eval)
        elif "dtu" in args.source_path:
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self.match_data = scene_info.match_data

        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"] # 相机的范围

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras,scene_info.match_data, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, scene_info.match_data, resolution_scale, args)

            print("Loading Pseudo Cameras")
            n_per_segment = 8
            pseudo_cam_poses= interpolate_virtual_poses_bspline_with_match(self.train_cameras[1.0], n_per_segment)
            pseudo_cams = []
            for pose in pseudo_cam_poses:
                # pseudo_image, pseudo_image_mask = self.wrap_image(pose,self.train_cameras[1.0],scene_info.match_data)
                pseudo_cams.append(PseudoCamera(
                    R=pose['R'].T, T=pose['T'], FoVx=self.train_cameras[1.0][0].FoVx, 
                    FoVy=self.train_cameras[1.0][0].FoVy, width=self.train_cameras[1.0][0].image_width, 
                    height=self.train_cameras[1.0][0].image_height,nearest_image=pose['nearest_img']))
            
            pseudo_cam_poses_offset = generate_offset_camera_poses(self.train_cameras[1.0])
            for pose in pseudo_cam_poses_offset:
                # pseudo_image, pseudo_image_mask = self.wrap_image(pose,self.train_cameras[1.0],scene_info.match_data)
                pseudo_cams.append(PseudoCamera(
                    R=pose['R'].T, T=pose['T'], FoVx=self.train_cameras[1.0][0].FoVx, 
                    FoVy=self.train_cameras[1.0][0].FoVy, width=self.train_cameras[1.0][0].image_width, 
                    height=self.train_cameras[1.0][0].image_height,nearest_image=pose['nearest_img']))
            
            self.pseudo_cameras[resolution_scale] = pseudo_cams
            
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_mono(scene_info.base_cameras, scene_info.match_data, self.cameras_extent) # 根据匹配信息进行点云初始化

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
    def save_init(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "init_point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply_at_matchpoint(os.path.join(point_cloud_path, "point_cloud_matchpoint.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getPseudoCameras(self,scale=1.0):
        return self.pseudo_cameras[scale]
    
    def get_match_data(self):
        return self.match_data
    

class VideoScene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        print("os.path.join(args.source_path:", os.path.join(args.source_path))
        if "basketball" in args.source_path:
            scene_info = sceneLoadTypeCallbacks["BasketballVideo"](args.source_path)
        elif "Tanks" in args.source_path:
            scene_info = sceneLoadTypeCallbacks["TanksVideo"](args.source_path)
        else:
            scene_info = sceneLoadTypeCallbacks["LLFFVideo"](args.source_path)

        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.create_from_mono(scene_info.base_cameras, scene_info.match_data, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
    def save_init(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "init_point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply_at_matchpoint(os.path.join(point_cloud_path, "point_cloud_matchpoint.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    