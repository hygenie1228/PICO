import cv2
import numpy as np
import trimesh
import json
import smplx
import torch

from src.constants import IMAGE_SIZE, SMPLX_FACES_PATH
from src.utils.structs import HumanParams, ObjectParams


model_type='smplx'
model_folder="data/human_model_files/smplx/SMPLX_NEUTRAL.npz"
layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False,
             'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False,
             'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
smplx_model = smplx.create(model_folder, model_type=model_type,
                     gender='neutral',
                     num_betas=10,
                     num_expression_coeffs=10, use_pca=False, use_face_contour=True, **layer_arg)
smplx_model = smplx_model

def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize image
    h, w = image.shape[:2]
    r = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
    w = int(r * w)
    h = int(r * h)
    image = cv2.resize(image, (w, h))

    return image


def load_human_params(human_inference_file: str, human_detection_file: str, imgsize: np.ndarray) -> HumanParams:
    with open(human_inference_file) as f:
        smplx_params = json.load(f)
    
    smplx_params['betas'] = torch.tensor(smplx_params['shape'])
    smplx_params['global_orient'] = torch.tensor(smplx_params['root_pose'])
    smplx_params['body_pose'] = torch.tensor(smplx_params['body_pose'])
    smplx_params['left_hand_pose'] = torch.tensor(smplx_params['lhand_pose'])
    smplx_params['right_hand_pose'] = torch.tensor(smplx_params['rhand_pose'])
    smplx_params['jaw_pose'] = torch.tensor(smplx_params['jaw_pose'])
    smplx_params['leye_pose'] = torch.zeros((1,3))
    smplx_params['reye_pose'] = torch.zeros((1,3))
    smplx_params['expression'] = torch.tensor(smplx_params['expr'])


    output = smplx_model(
            global_orient=smplx_params['global_orient'],
            body_pose=smplx_params['body_pose'],
            left_hand_pose=smplx_params['left_hand_pose'],
            right_hand_pose=smplx_params['right_hand_pose'],
            jaw_pose=smplx_params['jaw_pose'],
            leye_pose=smplx_params['leye_pose'],
            reye_pose=smplx_params['reye_pose'],
            expression=smplx_params['expression'],
            betas=smplx_params['betas']
        )

    vertices = output.vertices[0].cpu().numpy()
    faces = smplx_model.faces

    human_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    centroid_offset = np.array(smplx_params['cam_trans'])

    # resize to image size
    mask = cv2.imread(human_detection_file, 0) / 255.0
    mask = cv2.resize(mask, (imgsize[1], imgsize[0]))

    non_zero_pixels = np.where(mask > 0)
    y_min = np.min(non_zero_pixels[0])
    y_max = np.max(non_zero_pixels[0])
    x_min = np.min(non_zero_pixels[1])
    x_max = np.max(non_zero_pixels[1])

    # Calculate width and height
    w = x_max - x_min
    h = y_max - y_min

    # Create the bounding box
    human_bbox = np.array([x_min, y_min, w, h])

    human_params = HumanParams(
        vertices = human_mesh.vertices,
        faces = faces.astype(np.int32),
        centroid_offset = centroid_offset.copy(),
        bbox = human_bbox,
        mask = mask,
        smplx_params = smplx_params,
        intrinsic = {'focal': smplx_params['focal'], 'princpt': smplx_params['princpt']}
    )

    human_params.to_cuda()
    return human_params
    

def load_object_params(object_mesh_file: str, object_detection_file: str, imgsize: np.ndarray) -> ObjectParams:
    obj_mesh = trimesh.load(object_mesh_file)

    # rotate object 90 degrees around x-axis (mostly upright in objaverse)
    # obj_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))

    # center the mesh
    obj_mesh.apply_translation(-obj_mesh.centroid)

    # load object mask and resize to image size
    # with open(object_detection_file, 'r') as f:
    #     detection = json.load(f)
    # mask = np.array(detection['mask']).astype(float)

    mask = cv2.imread(object_detection_file, 0) / 255.0
    mask = cv2.resize(mask, (imgsize[1], imgsize[0]))

    object_params = ObjectParams(
        vertices = obj_mesh.vertices,
        faces = obj_mesh.faces,
        mask = mask,
        scale = obj_mesh.extents.max()
    )

    object_params.to_cuda()
    return object_params

