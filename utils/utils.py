import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# for flex
from psbody.mesh import Mesh
import chamfer_distance as chd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todense(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array.astype(dtype)


def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

def makelogger(log_dir,mode='w'):


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    fh = logging.FileHandler('%s'%log_dir, mode=mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def CRot2rotmat(pose):

    reshaped_input = pose.view(-1, 3, 2)

    b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack([b1, b2, b3], dim=-1)


def euler(rots, order='xyz', units='deg'):

    rots = np.asarray(rots)
    single_val = False if len(rots.shape)>1 else True
    rots = rots.reshape(-1,3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz)
        r = np.eye(3)
        for theta, axis in zip(xyz,order):
            c = np.cos(theta)
            s = np.sin(theta)
            if axis=='x':
                r = np.dot(np.array([[1,0,0],[0,c,-s],[0,s,c]]), r)
            if axis=='y':
                r = np.dot(np.array([[c,0,s],[0,1,0],[-s,0,c]]), r)
            if axis=='z':
                r = np.dot(np.array([[c,-s,0],[s,c,0],[0,0,1]]), r)
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_val:
        return rotmats[0]
    else:
        return rotmats

def batch_euler(bxyz,order='xyz', units='deg'):

    br = []
    for frame in range(bxyz.shape[0]):
        br.append(euler(bxyz[frame], order, units))
    return np.stack(br).astype(np.float32)

def rotate(points,R):
    shape = points.shape
    if len(shape)>3:
        points = points.squeeze()
    if len(shape)<3:
        points = points[:,np.newaxis]
    r_points = torch.matmul(torch.from_numpy(points).to(device), torch.from_numpy(R).to(device).transpose(1,2))
    return r_points.cpu().numpy().reshape(shape)

def rotmul(rotmat,R):

    shape = rotmat.shape
    rotmat = rotmat.squeeze()
    R = R.squeeze()
    rot = torch.matmul(torch.from_numpy(R).to(device),torch.from_numpy(rotmat).to(device))
    return rot.cpu().numpy().reshape(shape)

# import torchgeometry as tgm
# borrowed from the torchgeometry package
def rotmat2aa(rotmat):
    '''
    :param rotmat: Nx1xnum_jointsx9
    :return: Nx1xnum_jointsx3
    '''
    batch_size = rotmat.size(0)
    homogen_matrot = F.pad(rotmat.view(-1, 3, 3), [0,1])
    pose = rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
    return pose

def aa2rotmat(axis_angle):
    '''
    :param Nx1xnum_jointsx3
    :return: pose_matrot: Nx1xnum_jointsx9
    '''
    batch_size = axis_angle.size(0)
    pose_body_matrot = angle_axis_to_rotation_matrix(axis_angle.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
    return pose_body_matrot

def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4

def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


class RotConverter(nn.Module):
    '''
    this class is from smplx/vposer
    '''
    def __init__(self):
        super(RotConverter, self).__init__()

    def forward(self,module_input):
        pass


    @staticmethod
    def cont2rotmat(module_input):
        reshaped_input = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def aa2cont(module_input):
        '''
        :param NxTxnum_jointsx3
        :return: pose_matrot: NxTxnum_jointsx6
        '''
        batch_size = module_input.shape[0]
        n_frames = module_input.shape[1]
        pose_body_6d = angle_axis_to_rotation_matrix(module_input.reshape(-1, 3))[:, :3, :2].contiguous().view(batch_size, n_frames, -1, 6)

        return pose_body_6d


    @staticmethod
    def rotmat2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()

        return pose


    @staticmethod
    def aa2rotmat(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        batch_size = pose.shape[0]
        n_frames = pose.shape[1]
        pose_body_matrot = angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, n_frames, -1, 9)

        return pose_body_matrot

# ======================================================================================================================================== #
# add util functions from flex

def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
):
    """
    signed distance between two pointclouds
    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
    Returns:
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - x2y_signed: Torch.Tensor
            the sign distance from x to y
    """

    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x,y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near
    y2x = y - y_near

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    return y2x_signed, x2y_signed

def local2global(verts, rot_mat, trans):
    """
    Convert local mesh vertices to global using parameters (rot_mat, trans).

    :param verts    (torch.Tensor) -- size (b, N, 3)
    :param rot_mat  (torch.Tensor) -- size (b, 3, 3)
    :param trans    (torch.Tensor) -- size (b, 3)

    :return verts   (torch.Tensor) -- size (b, N, 3)
    """
    return torch.transpose(torch.bmm(rot_mat, torch.transpose(verts, 1, 2)), 1, 2) + trans[:, None, :]

def global2local(din_obj, rhand_verts):
    """
    Given object parameters and absolute rhand vertices,
    return new vertices of rhand in shifted frame of reference wrt object parameters.

    :param din_obj           (dict)     containing keys ['obj_glob', 'obj_tran'] torch.Tensor of sizes (1,3) & (1,3) resp.
    :param rhand_verts       (torch.Tensor) (N, 3)

    :return rhand_verts_new  (torch.Tensor) (N, 3)
    """
    obj_glob_rotmat = aa2rotmat(din_obj['obj_glob'][None, :]).reshape(3,3)
    diff = rhand_verts - din_obj['obj_tran']
    rhand_verts_new = torch.matmul(diff, obj_glob_rotmat.T)
    return rhand_verts_new


def load_obj_verts(rand_rotmat, object_mesh, n_sample_verts=10000):
    """
    Load object vertices corresponding to BPS representation which should be in the same distribution as used for GrabNet data.
    NOTE: the returned vertices are not transformed, but are simply meant to be used as input for RefineNet.

    :param cfg             (OmegaConf dict) with at least keys [obj_meshes_dir]
    :param obj             (str) e.g., 'wineglass'
    :param rand_rotmat     (torch.Tensor) -- (bs, 3, 3)
    :param scale           (float)
    :param n_sample_verts  (int)

    :return verts_sampled  (torch.Tensor) -- (bs, n_sample_verts, 3) - e.g., (250, 10000, 3)
    """
    obj_mesh_v = object_mesh[0]
    obj_mesh_f = object_mesh[1]
    # Center and scale the object.
    max_length = np.linalg.norm(obj_mesh_v, axis=1).max()
    if max_length > .3:
        re_scale = max_length/.08
        print(f'The object is very large, down-scaling by {re_scale} factor')
        obj_mesh_v = obj_mesh_v/re_scale
    object_fullpts = obj_mesh_v
    maximum = object_fullpts.max(0, keepdims=True)
    minimum = object_fullpts.min(0, keepdims=True)
    offset = ( maximum + minimum) / 2
    verts_obj = object_fullpts - offset

    # Batched rotation.
    bs = rand_rotmat.shape[0]
    obj_mesh_verts = torch.Tensor(verts_obj)[None].repeat(bs,1,1).to(rand_rotmat.device)  # (bs, 46891, 3)
    obj_mesh_verts_rotated = torch.bmm(obj_mesh_verts, torch.transpose(rand_rotmat, 1, 2))
    # Final choice.
    if verts_obj.shape[-2] < n_sample_verts:
        # Repeat vertices until n_sample_verts
        verts_sample_id = np.arange(verts_obj.shape[-2])
        repeated_verts = np.random.choice(verts_obj.shape[-2], n_sample_verts-verts_obj.shape[-2], replace=False)
        verts_sample_id = np.concatenate((verts_sample_id, repeated_verts))
    else:
        verts_sample_id = np.random.choice(verts_obj.shape[-2], n_sample_verts, replace=False)
    verts_sampled = obj_mesh_verts_rotated[..., verts_sample_id, :]

    obj_mesh_v = obj_mesh_verts_rotated

    obj_mesh_new = [obj_mesh_v, obj_mesh_f]

    return verts_sampled, obj_mesh_new

def get_ground(grnd_size=5, offset=0.0):
    """
    Return the mesh for ground.
    :param grnd_size    (int)
    :param offset       (int)
    :return grnd_mesh   (psbody mesh)
    """
    d = offset
    g_points = np.array([[-.2, -.2, d],
                         [.2, .2, d],
                         [.2, -0.2, d],
                         [-.2, .2, d]])
    g_faces = np.array([[0, 1, 2], [0, 3, 1]])
    grnd_mesh = Mesh(v=[grnd_size, grnd_size, 1] * g_points, f=g_faces)
    return grnd_mesh

def recompose_angle(yaw, pitch, roll, style='rotmat'):
    """
    Given yaw, pitch and roll, get final rotation matrix as a function of them.
    Return it in required 'style', i.e., rotmat / aa

    :param yaw    (torch.Tensor) -- shape (b,)
    :param pitch  (torch.Tensor) -- shape (b,)
    :param roll   (torch.Tensor) -- shape (b,)

    :return angle (torch.Tensor) -- shape (b,3) or (b,3,3)
    """
    bs = yaw.shape[0]
    yaw_rotmat = torch.vstack((torch.cos(yaw), -torch.sin(yaw), torch.zeros(bs).to(yaw.device),
                               torch.sin(yaw), torch.cos(yaw), torch.zeros(bs).to(yaw.device),
                               torch.zeros(bs).to(yaw.device), torch.zeros(bs).to(yaw.device), torch.ones(bs).to(yaw.device)
                             )).transpose(0,1).reshape(-1, 3, 3)
    pitch_rotmat = torch.vstack((torch.cos(pitch), torch.zeros(bs).to(pitch.device),  torch.sin(pitch),
                                 torch.zeros(bs).to(pitch.device), torch.ones(bs).to(pitch.device), torch.zeros(bs).to(pitch.device),
                                 -torch.sin(pitch), torch.zeros(bs).to(pitch.device), torch.cos(pitch)
                               )).transpose(0,1).reshape(-1, 3, 3)
    roll_rotmat = torch.vstack((torch.ones(bs).to(roll.device), torch.zeros(bs).to(roll.device), torch.zeros(bs).to(roll.device),
                                torch.zeros(bs).to(roll.device), torch.cos(roll),  -torch.sin(roll),
                                torch.zeros(bs).to(roll.device), torch.sin(roll), torch.cos(roll)
                              )).transpose(0,1).reshape(-1, 3, 3)
    angle = torch.bmm(torch.bmm(yaw_rotmat, pitch_rotmat), roll_rotmat)
    if style == 'aa':
        angle = rotmat2aa(angle.reshape(bs,1,1,9)).reshape(bs,3)
    return angle

def euler_torch(rots, order='xyz', units='deg'):
    """
    TODO: Confirm that copying does not affect gradient.

    :param rots     (torch.Tensor) -- (b, 3)
    :param order    (str)
    :param units    (str)

    :return r       (torch.Tensor) -- (b, 3, 3)
    """
    bs = rots.shape[0]
    single_val = False if len(rots.shape) > 1 else True
    rots = rots.reshape(-1, 3)

    if units == 'deg':
        rots = torch.deg2rad(rots)

    r = torch.eye(3)[None].repeat(bs, 1, 1).to(rots.device)

    for axis in range(3):
        theta, axis = rots[:, axis], order[axis]

        c = torch.cos(theta)
        s = torch.sin(theta)

        aux_r = torch.eye(3)[None].repeat(bs, 1, 1).to(rots.device)  # repeat, because expand copies memory, which we do not want here

        if axis == 'x':
            aux_r[:, 1, 1] = aux_r[:, 2, 2] = c
            aux_r[:, 1, 2] = s
            aux_r[:, 2, 1] = -s
        if axis == 'y':
            aux_r[:, 0, 0] = aux_r[:, 2, 2] = c
            aux_r[:, 0, 2] = s
            aux_r[:, 2, 0] = -s
        if axis == 'z':
            aux_r[:, 0, 0] = aux_r[:, 1, 1] = c
            aux_r[:, 0, 1] = -s
            aux_r[:, 1, 0] = s

        r = torch.matmul(aux_r, r)

    if single_val:
        return r[0]
    else:
        return r

def replace_topk(curr_res, topk):
    """
    Save curr_res in appropriate position in results based on index determined by loss compared with best_losses.

    :param curr_res       (dict) of keys ['pose_init', 'transl_init', 'global_orient_init', 'pose_final', 'transl_final', 'global_orient_final', 'rh_verts', 'loss_dict']

    :return results       (dict)
    """
    topk_idx = torch.topk(curr_res['loss_dict']['total'] * -1, topk).indices
    new_res = curr_res.copy()
    for k, v in new_res.items():
        if k == 'loss_dict' or k == 'losses': continue
        new_res[k] = v[topk_idx]
    for k, v in new_res['loss_dict'].items():
        new_res['loss_dict'][k] = v[topk_idx]
    for key, val in new_res['losses'].items():
        for idx, losses_stage in enumerate(val):
            new_res['losses'][key][idx] = losses_stage[topk_idx]
    return new_res