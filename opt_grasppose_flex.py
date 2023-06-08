import argparse
import os
import sys
from collections import defaultdict

import numpy as np
# import smplx
import open3d as o3d
import torch
from smplx.lbs import batch_rodrigues
from tqdm import tqdm

from utils.cfg_parser import Config
from utils.utils import makelogger, makepath, recompose_angle, replace_topk, rotmat2aa, aa2rotmat
from WholeGraspPose.models.fittingop_with_flex import FittingOP
from WholeGraspPose.models.objectmodel import ObjectModel
from WholeGraspPose.trainer import Trainer

from psbody.mesh import Mesh
import math
from WholeGraspPose.models.flex_opt import optimize_findz
import time
from datetime import datetime
from human_body_prior.tools.model_loader import load_vposer


#### inference
def load_obj_info(test_pose, obj_name):
    
    obj_meshes_dir = './dataset/contact_meshes'
    transf_transl, global_orient = test_pose[0], test_pose[1]
    transl = torch.zeros_like(transf_transl) # for object model which is centered at object

    # get the object info
    obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(obj_meshes_dir, obj_name + '.ply'))
    obj_mesh_base.compute_vertex_normals()
    v_temp = torch.FloatTensor(np.asarray(obj_mesh_base.vertices)).to(grabpose.device).view(1, -1, 3)
    normal_temp = torch.FloatTensor(np.asarray(obj_mesh_base.vertex_normals)).to(grabpose.device).view(1, -1, 3)
    obj_model = ObjectModel(v_temp, normal_temp, 1)
    global_orient_rotmat = batch_rodrigues(global_orient.view(-1, 3)).to(grabpose.device)  

    object_output = obj_model(global_orient_rotmat, transl.to(grabpose.device), v_temp.to(grabpose.device), normal_temp.to(grabpose.device), rotmat=True)
    object_verts = object_output[0].detach().cpu().numpy()
    object_normal = object_output[1].detach().cpu().numpy()

    index = np.linspace(0, object_verts.shape[1], num=2048, endpoint=False,retstep=True,dtype=int)[0]
    
    verts_object = object_verts[:, index]
    normal_object = object_normal[:, index]
    global_orient_rotmat_6d = global_orient_rotmat.view(-1, 1, 9)[:, :, :6].detach().cpu().numpy()
    feat_object = np.concatenate([normal_object, global_orient_rotmat_6d.repeat(2048, axis=1)], axis=-1)
    
    verts_object = torch.from_numpy(verts_object).to(grabpose.device)
    feat_object = torch.from_numpy(feat_object).to(grabpose.device)
    transf_transl = transf_transl.to(grabpose.device)

    object_info = {'verts_object':verts_object, 'normal_object': normal_object, 'global_orient':global_orient, 
                   'global_orient_rotmat':global_orient_rotmat, 'feat_object':feat_object, 'transf_transl':transf_transl}

    return object_info

def inference(grabpose, test_pose, obj, n_rand_samples, save_dir):
    """ prepare test object data: [verts_object, feat_object(normal + rotmat), transf_transl] """
    ### object centered
    obj_data = load_obj_info(test_pose, obj)
    obj_data['feat_object'] = obj_data['feat_object'].permute(0,2,1)
    obj_data['verts_object'] = obj_data['verts_object'].permute(0,2,1)

    n_samples_total = obj_data['feat_object'].shape[0]

    markers_gen = []
    object_contact_gen = []
    markers_contact_gen = []
    for i in range(n_samples_total):
        sample_results = grabpose.full_grasp_net.sample(obj_data['verts_object'][None, i].repeat(n_rand_samples,1,1), obj_data['feat_object'][None, i].repeat(n_rand_samples,1,1), obj_data['transf_transl'][None, i].repeat(n_rand_samples,1))
        markers_gen.append((sample_results[0]+obj_data['transf_transl'][None, i]))
        markers_contact_gen.append(sample_results[1])
        object_contact_gen.append(sample_results[2])

    markers_gen = torch.cat(markers_gen, dim=0)   # [B, N, 3]
    object_contact_gen = torch.cat(object_contact_gen, dim=0).squeeze()   # [B, 2048]
    markers_contact_gen = torch.cat(markers_contact_gen, dim=0)   # [B, N]

    output = {}
    output['markers_gen'] = markers_gen.detach().cpu().numpy()
    output['markers_contact_gen'] = markers_contact_gen.detach().cpu().numpy()
    output['object_contact_gen'] = object_contact_gen.detach().cpu().numpy()
    output['normal_object'] = obj_data['normal_object']#.repeat(n_rand_samples, axis=0)
    output['transf_transl'] = obj_data['transf_transl'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['global_orient_object'] = obj_data['global_orient'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['global_orient_object_rotmat'] = obj_data['global_orient_rotmat'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['verts_object'] = (obj_data['verts_object']+obj_data['transf_transl'].view(-1,3,1).repeat(1,1,2048)).permute(0, 2, 1).detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)

    save_path = os.path.join(save_dir, 'markers_results.npy')
    np.save(save_path, output)
    print('Saving to {}'.format(save_path))

    return output

def fitting_data_save(save_data,
              markers,
              markers_fit,
              smplxparams,
              gender,
              object_contact, body_contact,
              object_name, verts_object, global_orient_object, transf_transl_object):
    # markers & markers_fit
    save_data['markers'].append(markers)
    save_data['markers_fit'].append(markers_fit)
    # print('markers:', markers.shape)

    # body params
    for key in save_data['body'].keys():
        # print(key, smplxparams[key].shape)
        save_data['body'][key].append(smplxparams[key].detach().cpu().numpy())
    # object name & object params
    save_data['object_name'].append(object_name)
    save_data['gender'].append(gender)
    save_data['object']['transl'].append(transf_transl_object)
    save_data['object']['global_orient'].append(global_orient_object)
    save_data['object']['verts_object'].append(verts_object)

    # contact
    save_data['contact']['body'].append(body_contact)
    save_data['contact']['object'].append(object_contact)

#### fitting

def pose_opt(grabpose, samples_results, n_random_samples, obj, gender, save_dir, logger, device):
    # prepare objects
    n_samples = len(samples_results['verts_object'])
    verts_object = torch.tensor(samples_results['verts_object'])[:n_samples].to(device)  # (n, 2048, 3)
    normals_object = torch.tensor(samples_results['normal_object'])[:n_samples].to(device)  # (n, 2048, 3)
    global_orients_object = torch.tensor(samples_results['global_orient_object'])[:n_samples].to(device)  # (n, 2048, 3)
    transf_transl_object = torch.tensor(samples_results['transf_transl'])[:n_samples].to(device)  # (n, 2048, 3)

    # prepare body markers
    markers_gen = torch.tensor(samples_results['markers_gen']).to(device)  # (n*k, 143, 3)
    object_contacts_gen = torch.tensor(samples_results['object_contact_gen']).view(markers_gen.shape[0], -1, 1).to(device)  #  (n, 2048, 1)
    markers_contacts_gen = torch.tensor(samples_results['markers_contact_gen']).view(markers_gen.shape[0], -1, 1).to(device)   #  (n, 143, 1)

    print('Fitting {} {} samples for {}...'.format(n_samples, cfg.gender, obj.upper()))

    fittingconfig={ 'init_lr_h': 0.008,
                            'num_iter': [300,400,500],
                            'batch_size': 1,
                            'num_markers': 143,
                            'device': device,
                            'cfg': cfg,
                            'verbose': False,
                            'hand_ncomps': 45, # 24 in saga, change into 45 to align with flex
                            'only_rec': False,     # True / False 
                            'contact_loss': 'contact',  # contact / prior / False
                            'logger': logger,
                            'data_type': 'markers_143',
                            
                            }
    fittingop = FittingOP(fittingconfig)

    save_data_gen = {}
    for data in [save_data_gen]:
        data['markers'] = []
        data['markers_fit'] = []
        data['body'] = {}
        for key in ['betas', 'transl', 'global_orient', 'body_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'wrist_joint_transl', 'wrist_joint_global_orient', 'vpose_rec']:
            data['body'][key] = []
        data['object'] = {}
        for key in ['transl', 'global_orient', 'verts_object']:
            data['object'][key] = []
        data['contact'] = {}
        for key in ['body', 'object']:
            data['contact'][key] = []
        data['gender'] = []
        data['object_name'] = []


    for i in tqdm(range(n_samples)):
        # prepare object 
        vert_object = verts_object[None, i, :, :]
        normal_object = normals_object[None, i, :, :]

        marker_gen = markers_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]
        object_contact_gen = object_contacts_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]
        markers_contact_gen = markers_contacts_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]

        for k in range(n_random_samples):
            print('Fitting for {}-th GEN...'.format(k+1))
            markers_fit_gen, smplxparams_gen, loss_gen = fittingop.fitting(marker_gen[None, k, :], object_contact_gen[None, k, :], markers_contact_gen[None, k], vert_object, normal_object, gender)
            fitting_data_save(save_data_gen,
                    marker_gen[k, :].detach().cpu().numpy().reshape(1, -1 ,3),
                    markers_fit_gen[-1].squeeze().reshape(1, -1 ,3),
                    smplxparams_gen[-1],
                    gender,
                    object_contact_gen[k].detach().cpu().numpy().reshape(1, -1), markers_contact_gen[k].detach().cpu().numpy().reshape(1, -1),
                    obj, vert_object.detach().cpu().numpy(), global_orients_object[i].detach().cpu().numpy(), transf_transl_object[i].detach().cpu().numpy())


    for data in [save_data_gen]:
        # for data in [save_data_gt, save_data_rec, save_data_gen]:
            data['markers'] = np.vstack(data['markers'])  
            data['markers_fit'] = np.vstack(data['markers_fit'])
            for key in ['betas', 'transl', 'global_orient', 'body_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'wrist_joint_transl', 'wrist_joint_global_orient', 'vpose_rec']:
                data['body'][key] = np.vstack(data['body'][key])
            for key in ['transl', 'global_orient', 'verts_object']:
                data['object'][key] = np.vstack(data['object'][key])
            for key in ['body', 'object']:
                data['contact'][key] = np.vstack(data['contact'][key])

    np.savez(os.path.join(save_dir, 'saga_fitting_results.npz'), **save_data_gen)
    return save_data_gen

# flex opt
# =============Main classes====================================================================
class Optimize():

    def __init__(self, cfg):
        self.device = f'cuda:{cfg.cuda_id}'
        self.cfg = cfg
        self.vposer, _ = load_vposer(self.cfg.smplx_dir + '/vposer_v1_0', vp_model='snapshot')
        self.vposer.to(self.cfg.device)
        self.vposer.eval()


    def get_obstacle_info(self, obstacles_dict):
       """
       From obstacle_list with obstacle name, position and orientation, get vertices and normals from Mesh.

       :param obstacles_dict        (dict) with keys containing obstacle names and values list of [verts, faces]

       :return obstacles_info       (list) containing dicts with keys ['o_verts', 'o_faces'] - each a torch.Tensor.
       """
       obstacles_info = []
       for _, [verts, faces] in obstacles_dict.items():
            obj_verts = torch.from_numpy(verts.astype('float32'))  # (N, 3)
            obj_faces = torch.LongTensor(faces.astype('float32'))  # (F, 3)
            obstacles_info.append({'o_verts': obj_verts, 'o_faces': obj_faces})
       return obstacles_info


    def perform_optim(self, transl_init, global_orient_init,
                      vpose_rec_init, wrist_global_orient_init, wrist_transl_init, hand_pose_init,
                      obstacles_dict, object_mesh, obj_transl, obj_global_orient, obj_name,
                      model_name='flex'):
        """
        Main controller for optimization across 4 parameters.

        :param transl_init           (torch.Tensor) -- size (b, 3) on device
        :param global_orient_init    (torch.Tensor) -- size (b, 3) on device
        :param a_init                (torch.Tensor) -- size (b, 3) on device
        :param obstacles_dict        (dict) with keys containing obstacle names and values list of [verts, faces]
        :param obj_bps               (torch.Tensor) -- size (1, 4096) object bps representation for grasping object
        :param obj_transl            (torch.Tensor) -- size (1, 3) object translation for grasping object
        :param obj_global_orient     (torch.Tensor) -- size (1, 3) object global orientation for grasping object
        :param model_name            (str) -- Model class. Either 'flex' or 'latent'

        :return curr_res             (dict) of ['pose_init', 'transl_init',  'global_orient_init', 'pose_final', 'transl_final', 'global_orient_final', 'rh_verts, 'loss_dict', 'losses']
        """
        bs = self.cfg.batch_size

        # Save obstacle info (vertices, normals) required for loss computation.
        obstacles_info = self.get_obstacle_info(obstacles_dict)

        # Define if optimization is over transl and/or global orient.
        extras = {'obj_transl': obj_transl, 'obj_global_orient': obj_global_orient,
                  'obstacles_info': obstacles_info, 'obj_name': obj_name,
                  'object_mesh': object_mesh}

        # (*) Adapt model to this example.
        out_optim = optimize_findz(
                            cfg=self.cfg,
                            gan_body=self.vposer,
                            transl_init=transl_init,
                            global_orient_init=global_orient_init,
                            vpose_rec_init=vpose_rec_init, 
                            wrist_global_orient_init=wrist_global_orient_init,
                            wrist_transl_init=wrist_transl_init, 
                            hand_pose_init=hand_pose_init,
                            num_iterations=self.cfg.n_iter,
                            display=True,
                            extras=extras,
                            model_name=model_name
                        )

        dout, loss_dict, losses = out_optim
        dout = {k: dout[k].detach() for k in dout.keys()}

        # Return loss and result dict.
        curr_res = {'vpose_rec': vpose_rec_init, 'transl_init': transl_init, 'global_orient_init': global_orient_init,
                    'pose_final': dout['pose_body'], 'transl_final': dout['transl'], 'global_orient_final': dout['global_orient'],
                    'rh_verts': dout['rh_verts'], 'human_vertices': dout['human_vertices'], 'right_hand_pose': dout['right_hand_pose']}
        curr_res = {k: v.detach().cpu() for k,v in curr_res.items()}
        curr_res['loss_dict'] = loss_dict
        curr_res['losses'] = losses  # NOTE: losses is a dict of lists.

        return curr_res


    def get_inits(self, test_pose, bs, fitting_results):
        """
        :param test_pose      (list) of 2 torch.Tensors of size (1,3) & (1,3) for transl and global_orient of object respectively
        :param obj_bps        (torch.Tensor) -- (bs, 4096)     - bps representation of object in distribution for grasping (useful for GrabNet, not penetration losses)
        :param bs             (int) - batch size - product of number of initializations for each variable

        :return t_inits       (torch.Tensor) on device (bs, 3)
        :return g_inits       (torch.Tensor) on device (bs, 3)
        """

        # random initialize translation & global orientation
        t_inits = (test_pose[0] + torch.rand(bs, 3) * 0.5).to(self.device) # (bs, 3)     
        g_inits = recompose_angle(torch.rand(bs) * math.pi, torch.zeros(bs), torch.ones(bs) * 1.5, 'aa').to(self.device) # (bs, 3)       
        # a_inits = torch.rand([bs, 3]) * 360                                                                 # (bs, 3)                                                              
        # a_inits = a_inits.to(self.device)

        # initialize body params using the fitting results from saga opt
        vpose_rec_init = torch.FloatTensor(fitting_results['body']['vpose_rec']).repeat(bs, 1).to(self.device) # (bs, 32)   
        # wrist_transl_init = (test_pose[0] + torch.rand(bs, 3) * 0.5).to(self.device) # (bs, 3) 
        wrist_global_orient_init = recompose_angle(torch.rand(bs) * math.pi, torch.zeros(bs), torch.ones(bs) * 1.5, 'aa').to(self.device)      
        # wrist_global_orient_init =  torch.FloatTensor(fitting_results['body']['wrist_joint_global_orient']).repeat(bs, 1).to(self.device) # (bs, 3)            
        wrist_transl_init = torch.FloatTensor(fitting_results['body']['wrist_joint_transl']).repeat(bs, 1).to(self.device) # (bs, 3)          
        hand_pose_init = torch.FloatTensor(fitting_results['body']['right_hand_pose']).repeat(bs, 1).to(self.device) # (bs, 45)          

        params_init = {
            't_inits': t_inits, 'g_inits': g_inits, 'vpose_rec_init': vpose_rec_init, 
            'wrist_global_orient_init': wrist_global_orient_init, ' wrist_transl_init':  wrist_transl_init, 'hand_pose_init': hand_pose_init
        }

        return params_init


    def optimize(self, obj_name, test_pose, obstacles_dict, fitting_results, bs=1, model_name='flex'):
        """Given an object, perform grid search to optimize over 4 variables: pelvis translation, pelvis global orientation, body pose and latent (z) of VPoser.
        Save the top-k results which have the lowest loss based on constraints specified in loss function in `optimize_findz`.

        :param obj_name       (str)
        :param test_pose      (list) of 2 torch.Tensors of size (1,3) & (1,3) for transl and global_orient of object respectively
        :param obstacles_dict (dict) with keys containing obstacle names and values list of [verts, faces]
        :param bs             (int) - batch size
        :param model_name     (str) - model class. Either 'flex' or 'latent'

        :return results       (dict) - keys ['pose_init', 'transl_init', 'global_orient_init', 'pose_final', 'transl_final', 'global_orient_final', 'rh_verts', 'loss_dict', 'losses']
                                       where each is a tensor of size (topk, ..) except `loss_dict` which is itself a dict of detailed losses.
        """

        object_mesh = Mesh(filename=os.path.join(self.cfg.obj_meshes_dir, obj_name + '.ply'), vscale=1.)
        object_mesh.reset_normals()
        object_mesh = [object_mesh.v, object_mesh.f]

        # NOTE: for above
        #   - a_inits stores the random rotation that we used to get obj_bps.
        #   - We will use a_inits later to tranform the predicted hand pose back in the correct coordinate system
        params_init = self.get_inits(test_pose, bs, fitting_results)

        # (*) Perform main optimization for 4 initializations.
        start_time = time.time()
        obj_transl, obj_global_orient = test_pose[0].to(self.device), test_pose[1].to(self.device)
        curr_res = self.perform_optim(params_init['t_inits'], params_init['g_inits'],
                                      params_init['vpose_rec_init'], params_init['wrist_global_orient_init'], params_init[' wrist_transl_init'],
                                      params_init['hand_pose_init'],
                                      obstacles_dict, object_mesh, obj_transl, obj_global_orient, obj_name, model_name)

        # (*) Save topk results.
        results = replace_topk(curr_res, self.cfg.topk)
        print("--- %s seconds ---" % (time.time() - start_time))

        return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='grabpose-Testing')

    parser.add_argument('--data_path', default = './dataset/GraspPose', type=str,
                        help='The path to the folder that contains grabpose data')

    parser.add_argument('--object', default = None, type=str,
                        help='object name')

    parser.add_argument('--gender', default=None, type=str,
                        help='The gender of dataset')

    parser.add_argument('--config_path', default = 'WholeGraspPose/configs/WholeGraspPose.yaml', type=str,
                        help='The path to the confguration of the trained grabpose model')

    parser.add_argument('--exp_name', default = None, type=str,
                        help='experiment name')

    parser.add_argument('--pose_ckpt_path', default = None, type=str,
                        help='checkpoint path')

    parser.add_argument('--n_rand_samples_per_object', default = 1, type=int,
                        help='The number of whole-body poses random samples generated per object')
    
    parser.add_argument('--flex_cfg_path', default='WholeGraspPose/configs/Flex.yaml', type=str, help='The default config path for flex')
    
    parser.add_argument('--receptacle_name', default = 'receptacle_aabb_Tbl2_Top2_frl_apartment_table_02', type=str, 
                        help='Name of the receptacle.')

    parser.add_argument('--ornt_name', default = 'up', type=str, 
                        help='Orientation -- all or up.')

    parser.add_argument('--index', default=0, type=int, 
                        help='0/1/2')

    args = parser.parse_args()

    cwd = os.getcwd()

    best_net = os.path.join(cwd, args.pose_ckpt_path)

    vpe_path  = '/configs/verts_per_edge.npy'
    c_weights_path = cwd + '/WholeGraspPose/configs/rhand_weight.npy'
    work_dir = cwd + '/results/{}/GraspPose'.format(args.exp_name)
    print(work_dir)
    config = {
        'dataset_dir': args.data_path,
        'work_dir':work_dir,
        'vpe_path': vpe_path,
        'c_weights_path': c_weights_path,
        'exp_name': args.exp_name,
        'gender': args.gender,
        'best_net': best_net
    }

    cfg = Config(default_cfg_path=args.config_path, **config)

    save_dir = os.path.join(work_dir, args.object, args.receptacle_name, f'{args.ornt_name}_{args.index}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)        

    logger = makelogger(makepath(os.path.join(save_dir, '%s.log' % (args.object)), isfile=True)).info
    
    grabpose = Trainer(cfg=cfg, inference=True, logger=logger)

    # === Configuration for flex
    flex_config = {
        'gender': args.gender,
        'exp_name': args.exp_name,
        'obj_name': args.object,
        'batch_size': 500
    }
    cfg_flex = Config(default_cfg_path=args.flex_cfg_path, **flex_config)       # Create config for flex optimization

    # === Load replica dataset to get receptacle & object info
    recept_dict = dict(np.load('./dataset/replicagrasp/receptacles.npz', allow_pickle=1))
    dset_info_dict = dict(np.load('./dataset/replicagrasp/dset_info.npz', allow_pickle=1))
    transl_grab, orient_grab, recept_idx = dset_info_dict[f'{args.object}_{args.receptacle_name}_{args.ornt_name}_{args.index}']
    pose = [torch.Tensor(transl_grab), rotmat2aa(torch.Tensor(orient_grab).reshape(1,1,1,9)).reshape(1,3)]
    recept_v, recept_f = recept_dict[args.receptacle_name][recept_idx][0], recept_dict[args.receptacle_name][recept_idx][1]
    obstacles_dict = {args.receptacle_name: [recept_v, recept_f]}
    
    # === Start saga optimization
    samples_results = inference(grabpose, pose, args.object, args.n_rand_samples_per_object, save_dir)
    fitting_results = pose_opt(grabpose, samples_results, args.n_rand_samples_per_object, args.object, cfg.gender, save_dir, logger, grabpose.device)

    # === Start flex optimization
    print(f'--------FLEX Processing OBJ: {args.object}, RECEPT: {args.receptacle_name}, GENDER: {args.gender} -------------')
    path_save = f'{save_dir}/flex_fitting_results.npz'
    
    # === Main optimization instance.
    tto = Optimize(cfg=cfg_flex)

    # === Run search given an input object, it's pose and obstacles info.
    res = tto.optimize(obj_name=args.object, test_pose=pose, obstacles_dict=obstacles_dict, fitting_results=fitting_results, bs=cfg_flex.bs, model_name=cfg_flex.model_name)
    final_results = []
    for i in range(cfg_flex.topk):
        final_results.append({
            'human_vertices': res['human_vertices'][i].detach().cpu().numpy(),
            'pose': res['pose_final'][i].reshape(21, 3).detach().cpu().numpy(),
            'transl': res['transl_final'][i].detach().cpu().numpy(),
            'global_orient': aa2rotmat(res['global_orient_final'])[i].view(3, 3).detach().cpu().numpy(),
            'rh_verts': res['rh_verts'][i].detach().cpu().numpy(),
        })

    # === Save results for flex opt.
    np.savez(path_save, {'final_results': final_results, 'cfg_flex': cfg_flex,
                         'args': args, 'datetime': str(datetime.now())})


