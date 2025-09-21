import numpy as np
import pickle
import json
import torch

from src.constants import SMPL_TO_SMPLX_MATRIX_PATH
from src.utils.geometry import rot6d_to_matrix

import json
smplx_idxs = json.load(open("data/human_model_files/smplx_vert_segmentation.json"))


def load_contact_mapping(contact_map_path1: str, contact_map_path2: str, convert_to_smplx: bool = True) -> dict:

    # contact_transfer_map = {
    #     "objShaperighthand": [],
    #     "objShapelefthand": [],
    #     "humanShaperighthand": [],
    #     "humanShapelefthand": []
    # }

    
    human_contact = np.load(contact_map_path1)
    object_contact = np.load(contact_map_path2)

    contact_transfer_map = {
        "human": human_contact,
        "object": object_contact
    }

    return contact_transfer_map


def convert_contact_map_to_smplx(contact_transfer_map: dict) -> dict:
    # load smplx to smpl matrix
    with open(SMPL_TO_SMPLX_MATRIX_PATH, 'rb') as f:
        smpl_to_smplx = pickle.load(f)

    for key in contact_transfer_map:
        if key.startswith('human'):
            smplx_contact = []
            for v in contact_transfer_map[key]:
                v = int(v[1:])
                smpl_vertices = np.zeros((6890))
                smpl_vertices[v] = 1
                smplx_vertices = smpl_vertices @ smpl_to_smplx['matrix'].T
                    
                best = np.argmax(smplx_vertices)
                smplx_contact.append('v ' + str(best))

            contact_transfer_map[key] = smplx_contact

    return contact_transfer_map


def interpret_contact_points(contact_map, human_vertices, obj_mesh):
    # human_points = []
    # object_points = []
    # for shapekey in contact_map:
    #     if shapekey.startswith('humanShape'):  # Processing human mesh points
    #         for point in contact_map[shapekey]:
    #             if point.startswith('v'):  # Single vertex
    #                 idx = int(point.split()[1])
    #                 human_points.append(human_vertices[idx])
    #     elif shapekey.startswith('objShape'):  # Processing object mesh points
    #         for point in contact_map[shapekey]:
    #             if point.startswith('f'):  # Face with Barycentric coordinates
    #                 parts = point.split()
    #                 face_idx = int(parts[1])
    #                 bary_coords = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
    #                 # Find the vertices of the face
    #                 vertices = obj_mesh.vertices[obj_mesh.faces[face_idx]]
    #                 # Calculate the point using Barycentric coordinates
    #                 point = vertices[0] * bary_coords[0] + vertices[1] * bary_coords[1] + vertices[2] * (1 - bary_coords[0] - bary_coords[1])
    #                 object_points.append(point)
    #             elif point.startswith('e'):  # Edge
    #                 raise NotImplementedError("Edge contact points are not supported yet")
    #             elif point.startswith('v'):
    #                 raise NotImplementedError("Single vertex contact points are not supported yet")

    # human_points = torch.from_numpy(np.asarray(human_points)).float().cuda()
    # object_points = torch.from_numpy(np.asarray(object_points)).float().cuda()

    sample_num = min(contact_map['human'].sum(), contact_map['object'].sum())

    h_mask = np.asarray(contact_map['human']).squeeze()
    o_mask = np.asarray(contact_map['object']).squeeze()

    h_cands = np.flatnonzero(h_mask)
    o_cands = np.flatnonzero(o_mask)

    k_h = min(sample_num, len(h_cands))
    k_o = min(sample_num, len(o_cands))

    picked_human  = np.random.choice(h_cands, size=k_h, replace=False) if k_h > 0 else np.array([], dtype=int)
    picked_object = np.random.choice(o_cands, size=k_o, replace=False) if k_o > 0 else np.array([], dtype=int)

    human_points = human_vertices[picked_human]
    object_points = obj_mesh.vertices[picked_object]

    human_points = torch.from_numpy(np.asarray(human_points)).float().cuda()
    object_points = torch.from_numpy(np.asarray(object_points)).float().cuda()

    new_contact_map = {
        'human': picked_human,
        'object': picked_object,
    }

    return human_points, object_points, new_contact_map


def calculate_human_points(transformed_vertices, contact_data):
    # human_points = []
    # for shapekey in contact_data:
    #     if shapekey.startswith('humanShape'):  # Processing human mesh points
    #         for point in contact_data[shapekey]:
    #             if point.startswith('v'):  # Single vertex
    #                 idx = int(point.split()[1])
    #                 human_points.append(transformed_vertices[idx])
    # # Stack the list of tensors into a single tensor
    
    # try:
    #     human_points_tensor = torch.stack(human_points)
    # except:
    #     human_points_tensor = torch.tensor([]).cuda()

    human_contact = torch.tensor(contact_data['human'])
    human_points_tensor = transformed_vertices[human_contact]    
    return human_points_tensor


def calculate_object_points(transformed_vertices, contact_data, mesh_obj_faces):
    # object_points = []
    # for shapekey in contact_data:
    #     if shapekey.startswith('objShape'):  # Processing object mesh points
    #         for point in contact_data[shapekey]:
    #             if point.startswith('f'):  # Face with Barycentric coordinates
    #                 parts = point.split()
    #                 face_idx = int(parts[1])
    #                 bary_coords = torch.tensor([float(parts[2]), float(parts[3]), float(parts[4])], dtype=torch.float32)
                    
    #                 # Find the vertices of the face using PyTorch indexing
    #                 try:
    #                     fvi = mesh_obj_faces[face_idx]
    #                     face_vertex_indices = fvi.clone().detach().to(torch.long)
    #                     vertices = transformed_vertices[face_vertex_indices]
    #                 except:
    #                     print(f"face_idx: {face_idx}, len(mesh_obj_faces): {mesh_obj_faces.shape}")
                        
                    
    #                 # Calculate the point using Barycentric coordinates with PyTorch operations
    #                 point = vertices[0] * bary_coords[0] + vertices[1] * bary_coords[1] + vertices[2] * (1 - bary_coords[0] - bary_coords[1])
    #                 object_points.append(point)

    # # Stack the list of tensors into a single tensor
    # try:
    #     object_points_tensor = torch.stack(object_points)
    # except:
    #     object_points_tensor = torch.tensor([]).cuda()

    object_contact = torch.tensor(contact_data['object'])
    object_points_tensor = transformed_vertices[object_contact]    
    return object_points_tensor


def apply_transformation(vertices, rot, translation, scaling=1.0, rotmat=False):
    """Apply transformation using 6D rotation representation.
    inputs:
    - vertices: tensor of shape (N, 3) containing the vertices of the mesh
    - rot6d: tensor of shape (6,) containing the 6D rotation representation
    - translation: tensor of shape (3,) containing the translation values
    - scaling: scalar tensor containing the scaling factor
    returns:
    - transformed_vertices: tensor of shape (N, 3) containing the transformed vertices
    """
    if not rotmat:
        rot_matrix = rot6d_to_matrix(rot).view(1, 3, 3)
    else:
        rot_matrix = rot
    scaled_vertices = vertices * scaling
    rotated_vertices = torch.matmul(scaled_vertices.unsqueeze(1), rot_matrix).squeeze(1)
    transformed_vertices = rotated_vertices + translation
    
    return transformed_vertices 


def select_pose_parameters(contact_mapping: dict):
    parts = list(smplx_idxs.keys())
    human_contact = contact_mapping['human']

    contact_dict = {}
    for part in parts:
        idxs = smplx_idxs[part]
        is_contact =  human_contact[idxs].any()
        contact_dict[part] = is_contact

    # # get body parts that are in contact
    # contact_bodyparts = []
    # for k, _ in contact_mapping.items():
    #     if 'humanShape' in k:
    #         contact_bodyparts.append(k.replace('humanShape', ''))

    # select the joints to optimize
    joint_ids = []
    left_hand_opt = False
    right_hand_opt = False

    if contact_dict['leftHand'] or contact_dict['leftHandIndex1']:
        left_hand_opt = True
        joint_ids.extend([20, 18, 16])
    elif contact_dict['leftForeArm']:
        joint_ids.extend([18, 16])
    elif contact_dict['leftArm'] or contact_dict['leftShoulder']:
        joint_ids.extend([16])

    if contact_dict['rightHand'] or contact_dict['rightHandIndex1']:
        right_hand_opt = True
        joint_ids.extend([21, 19, 17])
    elif contact_dict['rightForeArm']:
        joint_ids.extend([19, 17])
    elif contact_dict['rightArm'] or contact_dict['rightShoulder']:
        joint_ids.extend([17])

    if contact_dict['leftFoot'] or contact_dict['leftToeBase']:
        joint_ids.extend([10, 7, 4, 1])
    elif contact_dict['leftLeg']:
        joint_ids.extend([4, 1])
    elif contact_dict['leftUpLeg']:
        joint_ids.extend([1])

    if contact_dict['rightFoot'] or contact_dict['rightToeBase']:
        joint_ids.extend([11, 8, 5, 2])
    elif contact_dict['rightLeg']:
        joint_ids.extend([5, 2])
    elif contact_dict['rightUpLeg']:
        joint_ids.extend([2])

    # remove duplicates
    joint_ids = list(set(joint_ids))
    # convert joint ids to smplx body pose indices
    body_pose_indices_to_opt = []
    for j in joint_ids:
        body_pose_indices_to_opt.extend([(j-1)*3, (j-1)*3+1, (j-1)*3+2])
    

    return body_pose_indices_to_opt, left_hand_opt, right_hand_opt
