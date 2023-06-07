import argparse
import os
import sys

import numpy as np
import open3d as o3d

from visualization_utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='grabpose-Testing')

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
    
    

    args = parser.parse_args()


    cwd = os.getcwd()

    load_path = f'../results/{args.exp_name}/GraspPose/{args.object}/{args.receptacle_name}/{args.ornt}_{args.index}/saga_fitting_results.npz'
    body_model_path = '../body_utils/body_models'
    contact_meshes_path = '../dataset/contact_meshes'
    receptacles_path = '../dataset/replicagrasp/receptacles.npz'
    dset_path = '../dataset/replicagrasp/dset_info.npz'

    data = np.load(load_path, allow_pickle=True)
    gender = args.gender
    object_name = args.object

    # Load replica dataset from file
    recept_dict = dict(np.load(receptacles_path, allow_pickle=1))
    dset_info_dict = dict(np.load(dset_path, allow_pickle=1))

    # Load recept & object info
    transl_grab, orient_grab, recept_idx = dset_info_dict[f'{args.object}_{args.receptacle_name}_{args.ornt}_{args.index}']
    recept_v, recept_f = recept_dict[args.receptacle_name][recept_idx][0], recept_dict[args.receptacle_name][recept_idx][1]


    n_samples = len(data['markers'])

    # Prepare mesh and pcd
    object_pcd = object_pcd = get_pcd(data['object'][()]['verts_object'][:n_samples])  # data['contact'][()]['object'][:n_samples], together with the contact map info
    object_mesh = get_object_mesh(contact_meshes_path, object_name, 'GRAB', data['object'][()]['transl'][:n_samples], data['object'][()]['global_orient'][:n_samples], n_samples)
    body_mesh, _ = get_body_mesh(body_model_path, data['body'][()], gender, n_samples)

    # Visualize receptacles.
    receptacle = o3d.geometry.TriangleMesh()
    receptacle.vertices = o3d.utility.Vector3dVector(recept_v)
    receptacle.triangles = o3d.utility.Vector3iVector(recept_f)
    receptacle.compute_vertex_normals()
    receptacle.paint_uniform_color(color_hex2rgb('#cc9966')) # brown


    # ground
    x_range = np.arange(-5, 50, 1)
    y_range = np.arange(-5, 50, 1)
    z_range = np.arange(0, 1, 1)
    gp_lines, gp_pcd = create_lineset(x_range, y_range, z_range)
    gp_lines.paint_uniform_color(color_hex2rgb('#bdbfbe'))   # grey
    gp_pcd.paint_uniform_color(color_hex2rgb('#bdbfbe'))     # grey
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    for i in range(n_samples):
        print(body_mesh[i])
        visualization_list = [body_mesh[i], object_mesh[i], coord, gp_lines, gp_pcd, receptacle]
        o3d.visualization.draw_geometries(visualization_list)

