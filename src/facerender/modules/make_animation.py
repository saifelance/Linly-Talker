from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = torch.arange(66, dtype=torch.float32, device=device)
    pred = F.softmax(pred, dim=1)
    degree = torch.sum(pred * idx_tensor, dim=1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * np.pi
    pitch = pitch / 180 * np.pi
    roll = roll / 180 * np.pi

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1).view(-1, 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                         torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                         -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1).view(-1, 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1).view(-1, 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)
    return rot_mat

def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical['value']
    yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he: yaw = he['yaw_in']
    if 'pitch_in' in he: pitch = he['pitch_in']
    if 'roll_in' in he: roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)
    t, exp = he['t'], he['exp']
    if wo_exp:
        exp = exp * 0

    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)
    t[:, 0] = t[:, 0] * 0
    t[:, 2] = t[:, 2] * 0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {'value': kp_transformed}

def make_animation(source_image, source_semantics, target_semantics,
                   generator, kp_detector, he_estimator, mapping, 
                   yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                   use_exp=True, use_half=False):

    device = torch.device("cpu")  # ✅ Force CPU
    source_image = source_image.to(device)
    source_semantics = source_semantics.to(device)
    target_semantics = target_semantics.to(device)
    if yaw_c_seq is not None:
        yaw_c_seq = yaw_c_seq.to(device)
    if pitch_c_seq is not None:
        pitch_c_seq = pitch_c_seq.to(device)
    if roll_c_seq is not None:
        roll_c_seq = roll_c_seq.to(device)

    generator.to(device)
    kp_detector.to(device)
    he_estimator.to(device)
    mapping.to(device)

    predictions = []
    with torch.no_grad():
        kp_canonical = kp_detector(source_image)
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source)

        for frame_idx in tqdm(range(target_semantics.shape[1]), desc='Face Renderer:'):
            target_semantics_frame = target_semantics[:, frame_idx]
            he_driving = mapping(target_semantics_frame)

            if yaw_c_seq is not None:
                he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
            if pitch_c_seq is not None:
                he_driving['pitch_in'] = pitch_c_seq[:, frame_idx]
            if roll_c_seq is not None:
                he_driving['roll_in'] = roll_c_seq[:, frame_idx]

            kp_driving = keypoint_transformation(kp_canonical, he_driving)
            out = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)
            predictions.append(out['prediction'])

        predictions_ts = torch.stack(predictions, dim=1)

    return predictions_ts

class AnimateModel(torch.nn.Module):
    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        return make_animation(
            source_image=x['source_image'],
            source_semantics=x['source_semantics'],
            target_semantics=x['target_semantics'],
            generator=self.generator,
            kp_detector=self.kp_extractor,
            he_estimator=self.mapping,
            mapping=self.mapping,
            yaw_c_seq=x.get('yaw_c_seq'),
            pitch_c_seq=x.get('pitch_c_seq'),
            roll_c_seq=x.get('roll_c_seq')
        )
