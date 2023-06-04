import numpy as np
import argparse
# import pyigl as igl
import igl as igl
# from  iglhelpers import e2p, p2e
# import trimesh
import torch
import json
import os 
import open3d as o3d
from tqdm import tqdm

import kaolin

from visualization.visualization_utils import *
from scipy.spatial.distance import pdist

def set_torch(deter=False):
    '''
    `deter` means use deterministic algorithms for GPU training reproducibility, 
    if set `deter=True`, please set the environment variable `CUBLAS_WORKSPACE_CONFIG` in advance
    '''
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(deter)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def uniform_box_sampling(min_corner, max_corner, res = 0.005):
    x_min = min_corner[0] - res
    x_max = max_corner[0] + res
    y_min = min_corner[1] - res
    y_max = max_corner[1] + res
    z_min = min_corner[2] - res
    z_max = max_corner[2] + res

    h = int((x_max-x_min)/res)+1
    l = int((y_max-y_min)/res)+1
    w = int((z_max-z_min)/res)+1

    # print('Sampling size: %d x %d x %d'%(h, l, w))

    with torch.no_grad():
        xyz = torch.zeros(h, l, w, 3, dtype=torch.float32) + torch.tensor([x_min, y_min, z_min], dtype=torch.float32)
        for i in range(1,h):
            xyz[i,0,0] = xyz[i-1,0,0] + torch.tensor([res,0,0])
        for i in range(1,l):
            xyz[:,i,0] = xyz[:,i-1,0] + torch.tensor([0,res,0])
        for i in range(1,w):
            xyz[:,:,i] = xyz[:,:,i-1] + torch.tensor([0,0,res])
    return res, xyz


def bounding_box_intersection(min_corner0, max_corner0, min_corner1, max_corner1):
    """
    Return the boundary of the intersection of two bounding boxes
    """
    min_x = max(min_corner0[0], min_corner1[0])
    min_y = max(min_corner0[1], min_corner1[1])
    min_z = max(min_corner0[2], min_corner1[2])

    max_x = min(max_corner0[0], max_corner1[0])
    max_y = min(max_corner0[1], max_corner1[1])
    max_z = min(max_corner0[2], max_corner1[2])

    if max_x > min_x and max_y > min_y and max_z > min_z:
        # print('Intersected bounding box size: %f x %f x %f'%(max_x - min_x, max_y - min_y, max_z - min_z))
        return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])
    else:
        return np.zeros((1,3), dtype = np.float32), np.zeros((1,3), dtype = np.float32) # no intersection case


def writeOff(output, vertex, face):
    with open(output, 'w') as f:
        f.write("COFF\n")
        f.write("%d %d 0\n" %(vertex.shape[0], face.shape[0]))
        for row in range(0, vertex.shape[0]):
            f.write("%f %f %f\n" %(vertex[row, 0], vertex[row, 1], vertex[row, 2]))
        for row in range(0, face.shape[0]):
            f.write("3 %d %d %d\n" %(face[row, 0], face[row, 1], face[row, 2]))

def s2h_intersection_eval(sbj_verts, sbj_faces, obj_verts):
    """ Calculate the interpenetration dist & contact of obstacle to human
    : returns
        obj2sbj_depth: max dist of inside obs points to human mesh
        contact: 1-contact, 0-no contact
    """
    sbj_verts = torch.FloatTensor(sbj_verts).unsqueeze(0)
    sbj_faces = torch.LongTensor(sbj_faces)
    obj_verts = torch.FloatTensor(obj_verts).unsqueeze(0)
    # Kaolin to calculate sign (True if inside, False if outside)
    sign = kaolin.ops.mesh.check_sign(sbj_verts, sbj_faces, obj_verts) # (bs, N_obj)                                                                                               
                                                                       
    # Calculate absolute distance of points from mesh, and multiply by sign.
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(sbj_verts, sbj_faces)                                    # (bs, F_sbj, 3, 3)
    dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(obj_verts.contiguous(), face_vertices) # (bs, N_obj)
    obj2sbj_depth = torch.max(dist * sign)  # only care about inside dist  
    contact = 1 if obj2sbj_depth > 0 else 0
    return float(obj2sbj_depth.cpu().detach()), contact

def h2o_intersection_eval(mesh0, mesh1, res=0.005, scale=1., trans=None, visualize_flag=False, visualize_file='output.off'):
    '''Calculate intersection depth and volumn of the two input meshes (body vs obj).
    args:
        mesh1, mesh2 (Trimesh.trimesh): input meshes
        res (float): voxel resolustion in meter(m)
        scale (float): scaling factor
        trans (float) (1, 3): translation
    returns:
        vol (float): intersection volume in cm^3
        depth (float): maximum depth from the center of voxel to the surface of another mesh
        contact: 1/0 flag to indicate whether the two meshes contact
    '''
    # mesh0 = trimesh.load(mesh_file_0, process=False)
    # mesh1 = trimesh.load(mesh_file_1, process=False)

    # scale = 1 # 10
    # res = 0.5

    mesh0_vertices = np.asarray(mesh0.vertices) * scale
    mesh1_vertices = np.asarray(mesh1.vertices) * scale

    mesh0_faces = np.asarray(mesh0.triangles)
    mesh1_faces = np.asarray(mesh1.triangles)


    S, I, C = igl.signed_distance(mesh0_vertices + 1e-10, mesh1_vertices, mesh1_faces, return_normals=False)

    mesh_mesh_distance = S.min()
    # print("dist", S)
    # print("Mesh to mesh distance: %f cm" % mesh_mesh_distance)

    #### print("Mesh to mesh distance: %f" % (max(S.min(), 0)))

    if mesh_mesh_distance > 0:
        # print('No intersection!')
        return 0, mesh_mesh_distance * 1e2, 0

    # Get bounding box for each mesh:
    min_corner0 = np.array([mesh0_vertices[:,0].min(), mesh0_vertices[:,1].min(), mesh0_vertices[:,2].min()])
    max_corner0 = np.array([mesh0_vertices[:,0].max(), mesh0_vertices[:,1].max(), mesh0_vertices[:,2].max()])

    min_corner1 = np.array([mesh1_vertices[:,0].min(), mesh1_vertices[:,1].min(), mesh1_vertices[:,2].min()])
    max_corner1 = np.array([mesh1_vertices[:,0].max(), mesh1_vertices[:,1].max(), mesh1_vertices[:,2].max()])

    # Compute the intersection of two bounding boxes:
    min_corner_i, max_corner_i = bounding_box_intersection(min_corner0, max_corner0, min_corner1, max_corner1)
    if ((min_corner_i - max_corner_i)**2).sum() == 0:
        # print('No intersection!')
        return 0, mesh_mesh_distance * 1e2, 0

    # Uniformly sample the intersection bounding box:
    _, xyz = uniform_box_sampling(min_corner_i, max_corner_i, res)
    xyz = xyz.view(-1, 3)
    xyz = xyz.detach().cpu().numpy()

    S, I, C = igl.signed_distance(xyz, mesh0_vertices, mesh0_faces, return_normals=False)

    inside_sample_index = np.argwhere(S < 0.0)
    # print("inside sample index", inside_sample_index, len(inside_sample_index))

    # Compute the signed distance for inside_samples to mesh 1:
    inside_samples = xyz[inside_sample_index[:,0], :]

    S, I, C = igl.signed_distance(inside_samples, mesh1_vertices, mesh1_faces, return_normals=False)

    inside_both_sample_index = np.argwhere(S < 0)

    # Compute intersection volume:
    i_v = inside_both_sample_index.shape[0] * (res**3)
    # print("Intersected volume: %f cm^3" % (i_v))

    # Visualize intersection volume:
    if visualize_flag:
        writeOff(visualize_file, inside_samples[inside_both_sample_index[:,0], :], np.zeros((0,3)))

    # From (m) to (cm)
    vol = i_v * 1e6
    depth = mesh_mesh_distance * 1e2
    contact = int(mesh_mesh_distance<0) # 1: contact
    return vol, depth, contact

def h2o_intersection_eval_on_vf(mesh0_vertices, mesh0_faces, mesh1_vertices, mesh1_faces, res=0.005, scale=1., trans=None, visualize_flag=False, visualize_file='output.off'):
    '''Calculate intersection depth and volumn of the two input meshes (body vs obj).
    args:
        mesh1, mesh2 (Trimesh.trimesh): input meshes
        res (float): voxel resolustion in meter(m)
        scale (float): scaling factor
        trans (float) (1, 3): translation
    returns:
        vol (float): intersection volume in cm^3
        depth (float): maximum depth from the center of voxel to the surface of another mesh
        contact: 1/0 flag to indicate whether the two meshes contact
    '''
    # mesh0 = trimesh.load(mesh_file_0, process=False)
    # mesh1 = trimesh.load(mesh_file_1, process=False)

    # scale = 1 # 10
    # res = 0.5

    mesh0_vertices *= scale
    mesh1_vertices *= scale


    S, I, C = igl.signed_distance(mesh0_vertices + 1e-10, mesh1_vertices, mesh1_faces, return_normals=False)

    mesh_mesh_distance = S.min()
    # print("dist", S)
    # print("Mesh to mesh distance: %f cm" % mesh_mesh_distance)

    #### print("Mesh to mesh distance: %f" % (max(S.min(), 0)))

    if mesh_mesh_distance > 0:
        # print('No intersection!')
        return 0, mesh_mesh_distance * 1e2, 0

    # Get bounding box for each mesh:
    min_corner0 = np.array([mesh0_vertices[:,0].min(), mesh0_vertices[:,1].min(), mesh0_vertices[:,2].min()])
    max_corner0 = np.array([mesh0_vertices[:,0].max(), mesh0_vertices[:,1].max(), mesh0_vertices[:,2].max()])

    min_corner1 = np.array([mesh1_vertices[:,0].min(), mesh1_vertices[:,1].min(), mesh1_vertices[:,2].min()])
    max_corner1 = np.array([mesh1_vertices[:,0].max(), mesh1_vertices[:,1].max(), mesh1_vertices[:,2].max()])

    # Compute the intersection of two bounding boxes:
    min_corner_i, max_corner_i = bounding_box_intersection(min_corner0, max_corner0, min_corner1, max_corner1)
    if ((min_corner_i - max_corner_i)**2).sum() == 0:
        # print('No intersection!')
        return 0, mesh_mesh_distance * 1e2, 0

    # Uniformly sample the intersection bounding box:
    _, xyz = uniform_box_sampling(min_corner_i, max_corner_i, res)
    xyz = xyz.view(-1, 3)
    xyz = xyz.detach().cpu().numpy()

    S, I, C = igl.signed_distance(xyz, mesh0_vertices, mesh0_faces, return_normals=False)

    inside_sample_index = np.argwhere(S < 0.0)
    # print("inside sample index", inside_sample_index, len(inside_sample_index))

    # Compute the signed distance for inside_samples to mesh 1:
    inside_samples = xyz[inside_sample_index[:,0], :]

    S, I, C = igl.signed_distance(inside_samples, mesh1_vertices, mesh1_faces, return_normals=False)

    inside_both_sample_index = np.argwhere(S < 0)

    # Compute intersection volume:
    i_v = inside_both_sample_index.shape[0] * (res**3)
    # print("Intersected volume: %f cm^3" % (i_v))

    # Visualize intersection volume:
    if visualize_flag:
        writeOff(visualize_file, inside_samples[inside_both_sample_index[:,0], :], np.zeros((0,3)))

    # From (m) to (cm)
    vol = i_v * 1e6
    depth = mesh_mesh_distance * 1e2
    contact = int(mesh_mesh_distance<0) # 1: contact
    return vol, depth, contact

def diversity_eval(marker_samples):
    """ Average L2 distance between all pairs of grasp samples to measure diversity within samples
    args:
        marker_samples: list of body marker samples from the ONE given object scene
    returns:
        diversity: APD of these samples
    """
    if marker_samples.shape[0] == 1:
        return 0.0
    # samples_array = []
    # for i in range(n_samples):
    #     samples_array.append(np.asarray(mesh_list[i].vertices).reshape(-1))
    # samples_array = np.stack(samples_array) # (n_samples, n_verts * 3)
    # dist = pdist(samples_array) # len(dist) = n_samples * (n_samples-1) / 2
    # diversity = np.mean(dist) # convert cm to m
    dist = pdist(marker_samples.reshape(marker_samples.shape[0], -1))
    diversity = dist.mean().item()
    return diversity

def evaluate(body_model_path, contact_meshes_path, load_path, gender, object_name, n_rand_samples_per_object, recept_dict):
    """ Compute apd, iterpenatration volume & depth, contact ratio for the given fitting_results file
    args:
        body_model_path:  the path to body_models
        contact_meshes_path: the path to contact_meshes
        load_path: the path to folder that contains the fitting_results.npz resides
        gender: the gender of the human body
        object_name: the class of the object
        n_rand_samples_per_object: The number of whole-body poses random samples generated per object
        recept_dict: the dict that stores the mesh v & f of receptacles
    returns:
        eval_dict: dict of
            apd list
            inter_vol list
            inter_depth list
            contact list
    """
    result_path = os.path.join(load_path, "fitting_results.npz")
    data = np.load(result_path, allow_pickle=True)
    body_markers = data['markers']
    n_samples = body_markers.shape[0] # 10

    # Prepare the meshes of objects & bodies
    object_mesh_v, object_mesh_f = get_object_mesh_vf(contact_meshes_path, object_name, 'GRAB', data['object'][()]['transl'], data['object'][()]['global_orient'], n_samples)
    body_mesh_v, body_mesh_f = get_body_mesh_vf(body_model_path, data['body'][()], gender, n_samples)

    # load the mesh info of the obstacles
    receptacle_name, recept_idx = data['obstacle'][()]['receptacle_name'], data['obstacle'][()]['recept_idx'] 

    # compute the apd, need to group all samples from one object scene
    print("Start cal apd")
    apd = []
    for i in range(0, n_samples, n_rand_samples_per_object):
        apd.append(diversity_eval(body_markers[i:i+n_rand_samples_per_object]))
    print("APD cal done")
    
    # compute interpenetration volume & depth
    obj_inter_vol, obj_inter_depth, obj_contact = [], [], []
    obs_inter_vol, obs_inter_depth, obs_contact = [], [], []
    for i in tqdm(range(n_samples)):
        # calculate the intersection of humen VS grab object
        print("start cal h2o")
        obj_vol_i, obj_depth_i, obj_contact_i = h2o_intersection_eval_on_vf(body_mesh_v[i], body_mesh_f, object_mesh_v[i], object_mesh_f, res=0.001, visualize_flag=False)
        print("End cal h2o")
        obj_inter_vol.append(obj_vol_i) 
        obj_inter_depth.append(obj_depth_i)
        obj_contact.append(obj_contact_i)

        # calculate the intersection of humen VS obstacle object
        receptacle_name_i, recept_idx_i = receptacle_name[i,0], recept_idx[i, 0]
        recept_v_i, _ = recept_dict[receptacle_name_i][recept_idx_i][0], recept_dict[receptacle_name_i][recept_idx_i][1]
        print("start cal h2s")
        obs_depth_i, obs_contact_i = s2h_intersection_eval(body_mesh_v[i], body_mesh_f, recept_v_i)
        print(f"type of obs_depth_i: {type(obs_depth_i)}, value is {obs_depth_i}")
        print("End cal h2s")
        obs_inter_vol.append(0) 
        obs_inter_depth.append(obs_depth_i)
        obs_contact.append(obs_contact_i)
    
    # print("=================================================")
    # print("Evaluation results for {} and {} with {} samples:".format(gender, object_name, n_samples))
    # print("AVG_APD: {}".format(np.mean(apd)))
    # print("AVG_inter_vol: {}".format(np.mean(inter_vol)))
    # print("AVG_inter_depth: {}".format(np.mean(inter_depth)))
    # print("contact_ratio: {}".format(np.mean(contact)))
    # print("=================================================")

    # construct the output dict
    output = dict()
    output['apd'] = apd
    output['avg_apd'] = np.mean(apd)

    output['obj_inter_vol'] = obj_inter_vol
    output['obj_inter_depth'] = obj_inter_depth
    output['obj_contact'] = obj_contact
    output['avg_obj_inter_vol'] = np.mean(obj_inter_vol)
    output['avg_obj_inter_depth'] = np.mean(obj_inter_depth)
    output['obj_contact_ratio'] = np.mean(obj_contact)

    output['obs_inter_vol'] = obs_inter_vol
    output['obs_inter_depth'] = obs_inter_depth
    output['obs_contact'] = obs_contact
    output['avg_obs_inter_vol'] = np.mean(obs_inter_vol)
    output['avg_obs_inter_depth'] = np.mean(obs_inter_depth)
    output['obs_contact_ratio'] = np.mean(obs_contact)

    output['n_samples'] = n_samples
    output['n_rand_samples_per_object'] = n_rand_samples_per_object

    # write to the json file
    output_path = os.path.join(load_path,"eval_fitting.json")
    with open(output_path, "w") as outfile:
        json.dump(output, outfile)

    return output



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='grabpose-Testing')

    parser.add_argument('--exp_name', default = None, type=str,
                        help='experiment name')

    parser.add_argument('--gender', default = None, type=str,
                        help='gender')

    parser.add_argument('--object', default = None, type=str,
                        help='object name')
    
    parser.add_argument('--n_rand_samples_per_object', default = 1, type=int,
                        help='The number of whole-body poses random samples generated per object')

    parser.add_argument('--object_format', default = 'mesh', type=str,
                        help='pcd or mesh')
    
    parser.add_argument('--fitting_path', default = 'None', type=str,
                        help='the folder path of the fitting_results.npz')


    args = parser.parse_args()
    
    gender = args.gender
    object_name = args.object
    n_rand_samples_per_object = args.n_rand_samples_per_object

    cwd = os.getcwd()
    
    body_model_path = cwd + '/body_utils/body_models'
    contact_meshes_path = cwd + '/dataset/contact_meshes'
    receptacles_path = './dataset/replicagrasp/receptacles.npz'
    recept_dict = dict(np.load(receptacles_path, allow_pickle=1))

    evaluate(body_model_path, contact_meshes_path, args.fitting_path, gender, object_name, n_rand_samples_per_object, recept_dict)