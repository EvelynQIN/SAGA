import argparse
import os
import sys

import numpy as np
import open3d as o3d

from visualization_utils import *
# from psbody.mesh import Mesh
import smplx
import torch
import mano
from utils.utils import rotmat2aa

# Helper functions and variables.

def get_sbj(res, gender):
    # ---- (*) BODY
    sbj_m = smplx.create(model_path='../body_utils/body_models',
                        model_type='smplx',
                        gender=gender,
                        num_pca_comps=np.array(24),
                        batch_size=1).to('cpu').eval()
    body_params = {'transl': torch.Tensor(res['transl'].reshape(1,3)),
                   'global_orient': torch.Tensor(res['global_orient'].reshape(1,3)),
                   'body_pose': torch.Tensor(res['pose'].reshape(1,63))}
    bm = sbj_m(**body_params)
    body_vertices = bm.vertices.detach().cpu().numpy()[0]
    body_vertices[..., 2] -= body_vertices[..., 2].min(-1, keepdims=True)[0]
    return body_vertices, sbj_m.faces


def get_rh(res, gender):
    # ---- (*) RHAND
    rh_m = mano.load(model_path='../body_utils/body_models/mano',
                     is_right=True,
                     model_type='mano',
                     gender=gender,
                     num_pca_comps=45,
                     flat_hand_mean=True,
                     batch_size=1).to('cpu').eval()
    return res['rh_verts'].reshape(-1, 3), rh_m.faces

def get_ground(grnd_size=5, xmean=0, ymean=0):
    """
    Return the mesh for ground.
    :param grnd_size    (int)
    :param offset       (int)
    :return grnd_mesh   (psbody mesh)
    """
    x_range = np.arange(-1 * grnd_size, grnd_size, 1)
    y_range = np.arange(-1 * grnd_size, grnd_size, 1)
    z_range = np.arange(0, 1, 1)
    x_range += int(xmean)
    y_range += int(ymean)
    gp_lines, gp_pcd = create_lineset(x_range, y_range, z_range)
    gp_lines.paint_uniform_color(color_hex2rgb('#bdbfbe'))   # grey
    gp_pcd.paint_uniform_color(color_hex2rgb('#bdbfbe'))     # grey
    return gp_lines, gp_pcd


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='grabpose-visualization')

    parser.add_argument('--exp_name', default = None, type=str,
                        help='experiment name')

    parser.add_argument('--gender', default = None, type=str,
                        help='gender')

    parser.add_argument('--object', default = None, type=str,
                        help='object name')
    
    parser.add_argument('--receptacle_name', default = None, type=str,
                        help='receptacle_name')
    
    parser.add_argument('--ornt', default = 'up', type=str,
                        help='up or all')
    
    parser.add_argument('--index', default = None, type=int,
                        help='0/1/2')
        
    # parser.add_argument('--visual_cue', default = 'contactmap', type=str,
    #                     help='objectmesh / contactmap')

    args = parser.parse_args()


    cwd = os.getcwd()

    load_path = f'../results/{args.exp_name}/GraspPose/{args.object}/{args.receptacle_name}/{args.ornt}_{args.index}/flex_fitting_results.npz'
    body_model_path = '../body_utils/body_models'
    contact_meshes_path = '../dataset/contact_meshes'
    receptacles_path = '../dataset/replicagrasp/receptacles.npz'
    dset_path = '../dataset/replicagrasp/dset_info.npz'

     # Load flex fitting results
    res = dict(np.load(load_path, allow_pickle=True))

    # Load replica dataset from file
    recept_dict = dict(np.load(receptacles_path, allow_pickle=1))
    dset_info_dict = dict(np.load(dset_path, allow_pickle=1))

    # Load recept & object info
    transl_grab, orient_grab, recept_idx = dset_info_dict[f'{args.object}_{args.receptacle_name}_{args.ornt}_{args.index}']
    recept_v, recept_f = recept_dict[args.receptacle_name][recept_idx][0], recept_dict[args.receptacle_name][recept_idx][1]

    # Visualize ground for context (center it to mean of rigid/articulated object).
    xmean, ymean, _ = recept_v.mean(0)
    gp_lines, gp_pcd = get_ground(grnd_size=20, xmean=xmean, ymean=ymean)

    # Visualize object to be grasped in scene.
    object_mesh = o3d.io.read_triangle_mesh(f'{contact_meshes_path}/{args.object}.ply')
    obj_verts = np.matmul(object_mesh.vertices, orient_grab.T) + transl_grab
    grab_obj = o3d.geometry.TriangleMesh()
    grab_obj.vertices = o3d.utility.Vector3dVector(obj_verts)
    grab_obj.triangles = object_mesh.triangles
    grab_obj.compute_vertex_normals()
    grab_obj.paint_uniform_color(color_hex2rgb('#3399ff')) # blue

    # Visualize receptacles.
    receptacle = o3d.geometry.TriangleMesh()
    receptacle.vertices = o3d.utility.Vector3dVector(recept_v)
    receptacle.triangles = o3d.utility.Vector3iVector(recept_f)
    receptacle.compute_vertex_normals()
    receptacle.paint_uniform_color(color_hex2rgb('#cc9966')) # brown


    visualization_list = [gp_lines, gp_pcd, grab_obj, receptacle]

    # Get top-5 result.
    for idx, color in enumerate(['#00cc00', '#ff3399', '#999966', '#ff6600', '#6600ff']):
        res_i = dict(res['arr_0'].item())['final_results'][idx]
        res_i_new = res_i.copy()
        res_i_new['global_orient'] = rotmat2aa(torch.Tensor(res_i['global_orient']).reshape(1,1,1,9)).reshape(1, 3).numpy()
        # Visualize human.
        sbj_v, sbj_f = get_sbj(res_i_new, args.gender)
        body_i = o3d.geometry.TriangleMesh()
        body_i.vertices = o3d.utility.Vector3dVector(sbj_v)
        body_i.triangles = o3d.utility.Vector3iVector(sbj_f)
        body_i.compute_vertex_normals()
        body_i.paint_uniform_color(color_hex2rgb(color))
        visualization_list.append(body_i)
    o3d.visualization.draw_geometries(visualization_list)

