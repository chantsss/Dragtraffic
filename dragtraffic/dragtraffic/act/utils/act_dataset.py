import copy
import os
import pickle

import numpy as np
import torch
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from tqdm import tqdm
import glob
from dragtraffic.utils.config import load_config_act, get_parsed_args

LANE_SAMPLE = 10
RANGE = 60
MAX_AGENT = 32
MAX_CONDITION_POINTS_NUM = 1
MIN_VALID_FRAMES = 30

# object_type = {
#     0: 'TYPE_UNSET',
#     1: 'TYPE_VEHICLE',
#     2: 'TYPE_PEDESTRIAN',
#     3: 'TYPE_CYCLIST',
#     4: 'TYPE_OTHER'
# }

def cal_rel_dir(dir1, dir2):
    """
    Calculate the relative direction between two directions.

    Parameters:
    dir1 (float): The first direction.
    dir2 (float): The second direction.

    Returns:
    float: The relative direction between dir1 and dir2.
    """
    dist = dir1 - dir2

    while not np.all(dist >= 0):
        dist[dist < 0] += np.pi * 2
    while not np.all(dist < np.pi * 2):
        dist[dist >= np.pi * 2] -= np.pi * 2

    dist[dist > np.pi] -= np.pi * 2
    return dist


def rotate(x, y, angle):
    """
    Rotate the coordinates (x, y) by the given angle.

    Args:
        x (torch.Tensor or np.ndarray): The x-coordinates.
        y (torch.Tensor or np.ndarray): The y-coordinates.
        angle (float): The angle of rotation in radians.

    Returns:
        torch.Tensor or np.ndarray: The rotated coordinates.

    """
    if isinstance(x, torch.Tensor):
        other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
        other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
        output_coords = torch.stack((other_x_trans, other_y_trans), axis=-1)

    else:
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    return output_coords


def wash(batch):
    """
    Convert float64 arrays in the batch to float32.

    Args:
        batch (dict): A dictionary containing arrays.

    Returns:
        None. The function modifies the input dictionary in-place.
    """
    for key in batch.keys():
        if isinstance(batch[key], np.ndarray):
            if batch[key].dtype == np.float64:
                batch[key] = batch[key].astype(np.float32)


def process_map(lane, traf, center_num=384, edge_num=128, lane_range=60, offest=-40):
    """
    Process the lane and traffic data to generate different types of lanes and their corresponding masks.

    Args:
        lane (numpy.ndarray): Array containing lane data.
        traf (numpy.ndarray): Array containing traffic data.
        center_num (int, optional): Number of center lanes to process. Defaults to 384.
        edge_num (int, optional): Number of edge lanes to process. Defaults to 128.
        lane_range (int, optional): Range of the lanes. Defaults to 60.
        offest (int, optional): Offset value for lane processing. Defaults to -40.

    Returns:
        tuple: A tuple containing the processed center lanes, center lane masks, edge lanes, edge lane masks,
               crosswalk lanes, crosswalk lane masks, rest lanes, and rest lane masks.
    """
    lane_with_traf = np.zeros([*lane.shape[:-1], 5])
    lane_with_traf[..., :4] = lane

    lane_id = lane[..., -1]
    b_s = lane_id.shape[0]

    for i in range(b_s):
        traf_t = traf[i]
        lane_id_t = lane_id[i]
        for a_traf in traf_t:
            control_lane_id = a_traf[0]
            state = a_traf[-2]
            lane_idx = np.where(lane_id_t == control_lane_id)
            lane_with_traf[i, lane_idx, -1] = state

    # lane = np.delete(lane_with_traf,-2,axis=-1)
    lane = lane_with_traf
    lane_type = lane[0, :, 2]
    center_1 = lane_type == 1
    center_2 = lane_type == 2
    center_3 = lane_type == 3
    center_ind = center_1 + center_2 + center_3

    boundary_1 = lane_type == 15
    boundary_2 = lane_type == 16
    bound_ind = boundary_1 + boundary_2

    cross_walk = lane_type == 18
    speed_bump = lane_type == 19
    cross_ind = cross_walk + speed_bump

    rest = ~(center_ind + bound_ind + cross_walk + speed_bump + cross_ind)

    cent, cent_mask = process_lane(lane[:, center_ind], center_num, lane_range, offest)
    bound, bound_mask = process_lane(lane[:, bound_ind], edge_num, lane_range, offest)
    cross, cross_mask = process_lane(lane[:, cross_ind], 32, lane_range, offest)
    rest, rest_mask = process_lane(lane[:, rest], 192, lane_range, offest)

    return cent, cent_mask, bound, bound_mask, cross, cross_mask, rest, rest_mask


def process_lane(lane, max_vec, lane_range, offset=-40):
    """
    Process the lane data to generate vectors and masks.

    Args:
        lane (numpy.ndarray): The lane data array of shape (batch_size, num_points, lane_dim).
        max_vec (int): The maximum number of vectors to generate.
        lane_range (float): The range of the lane.
        offset (int, optional): The offset value. Defaults to -40.

    Returns:
        tuple: A tuple containing the generated vectors and masks.
            - all_vec (numpy.ndarray): The generated vectors of shape (batch_size, max_vec, vec_dim).
            - all_mask (numpy.ndarray): The generated masks of shape (batch_size, max_vec).

    """
    vec_dim = 6

    lane_point_mask = (abs(lane[..., 0] + offset) < lane_range) * (abs(lane[..., 1]) < lane_range)

    lane_id = np.unique(lane[..., -2]).astype(int)

    vec_list = []
    vec_mask_list = []
    b_s, _, lane_dim = lane.shape

    for id in lane_id:
        id_set = lane[..., -2] == id
        points = lane[id_set].reshape(b_s, -1, lane_dim)
        masks = lane_point_mask[id_set].reshape(b_s, -1)

        vector = np.zeros([b_s, points.shape[1] - 1, vec_dim])
        vector[..., 0:2] = points[:, :-1, :2]
        vector[..., 2:4] = points[:, 1:, :2]
        # id
        # vector[..., 4] = points[:,1:, 3]
        # type
        vector[..., 4] = points[:, 1:, 2]
        # traffic light
        vector[..., 5] = points[:, 1:, 4]
        vec_mask = masks[:, :-1] * masks[:, 1:]
        vector[vec_mask == 0] = 0
        vec_list.append(vector)
        vec_mask_list.append(vec_mask)

    vector = np.concatenate(vec_list, axis=1) if vec_list else np.zeros([b_s, 0, vec_dim])
    vector_mask = np.concatenate(vec_mask_list, axis=1) if vec_mask_list else np.zeros([b_s, 0], dtype=bool)

    all_vec = np.zeros([b_s, max_vec, vec_dim])
    all_mask = np.zeros([b_s, max_vec])
    for t in range(b_s):
        mask_t = vector_mask[t]
        vector_t = vector[t][mask_t]

        dist = vector_t[..., 0]**2 + vector_t[..., 1]**2
        idx = np.argsort(dist)
        vector_t = vector_t[idx]
        mask_t = np.ones(vector_t.shape[0])

        vector_t = vector_t[:max_vec]
        mask_t = mask_t[:max_vec]

        vector_t = np.pad(vector_t, ([0, max_vec - vector_t.shape[0]], [0, 0]))
        mask_t = np.pad(mask_t, ([0, max_vec - mask_t.shape[0]]))
        all_vec[t] = vector_t
        all_mask[t] = mask_t

    return all_vec, all_mask.astype(bool)


class WaymoAgent:
    """
    Represents an agent in the Waymo dataset.

    Args:
        feature (ndarray): The feature array containing information about the agent.
        vec_based_info (ndarray, optional): Vector-based information about the agent. Defaults to None.
        range (int, optional): The range of the agent. Defaults to 50.
        max_speed (int, optional): The maximum speed of the agent. Defaults to 30.
        from_inp (bool, optional): Indicates whether the feature array is from input. Defaults to False.
    """

    def __init__(self, feature, vec_based_info=None, range=50, max_speed=30, from_inp=False):
        # index of xy,v,lw,yaw,type,valid

        self.RANGE = range
        self.MAX_SPEED = max_speed

        if from_inp:
            self.position = feature[..., :2] * self.RANGE
            self.velocity = feature[..., 2:4] * self.MAX_SPEED
            self.heading = np.arctan2(feature[..., 5], feature[..., 4])
            self.length_width = feature[..., 6:8]

        else:
            self.feature = feature
            self.position = feature[..., :2]
            self.velocity = feature[..., 2:4]
            self.heading = feature[..., [4]]
            self.length_width = feature[..., 5:7]
            self.type = feature[..., [7]]
            self.vec_based_info = vec_based_info

    def get_agent(self, indx):
        return WaymoAgent(self.feature[[indx]], self.vec_based_info[[indx]])

    def get_inp(self, act=False, act_inp=False):
        """
        Get the input representation of the agent.

        Args:
            act (bool, optional): Indicates whether to output the representation directly. Defaults to False.
            act_inp (bool, optional): Indicates whether to normalize the representation. Defaults to False.

        Returns:
            ndarray: The input representation of the agent.
        """

        if act:
            return np.concatenate([self.position, self.velocity, self.heading, self.length_width], axis=-1)

        pos = self.position / self.RANGE
        velo = self.velocity / self.MAX_SPEED
        cos_head = np.cos(self.heading)
        sin_head = np.sin(self.heading)

        if act_inp:
            return np.concatenate([pos, velo, self.heading, self.type, self.length_width], axis=-1)
            # return np.concatenate([pos, velo, cos_head, sin_head, self.type, self.length_width], axis=-1)

        vec_based_rep = copy.deepcopy(self.vec_based_info)
        vec_based_rep[..., 5:9] /= self.RANGE
        vec_based_rep[..., 2] /= self.MAX_SPEED
        agent_feat = np.concatenate([pos, velo, cos_head, sin_head, self.length_width, vec_based_rep], axis=-1)
        return agent_feat

    def get_rect(self, pad=0):
        """
        Get the rectangular representation of the agent.

        Args:
            pad (int, optional): The padding value. Defaults to 0.

        Returns:
            list: A list of rectangular representations of the agent.
        """

        l, w = (self.length_width[..., 0] + pad) / 2, (self.length_width[..., 1] + pad) / 2
        x1, y1 = l, w
        x2, y2 = l, -w

        point1 = rotate(x1, y1, self.heading[..., 0])
        point2 = rotate(x2, y2, self.heading[..., 0])
        center = self.position

        x1, y1 = point1[..., [0]], point1[..., [1]]
        x2, y2 = point2[..., [0]], point2[..., [1]]

        p1 = np.concatenate([center[..., [0]] + x1, center[..., [1]] + y1], axis=-1)
        p2 = np.concatenate([center[..., [0]] + x2, center[..., [1]] + y2], axis=-1)
        p3 = np.concatenate([center[..., [0]] - x1, center[..., [1]] - y1], axis=-1)
        p4 = np.concatenate([center[..., [0]] - x2, center[..., [1]] - y2], axis=-1)

        p1 = p1.reshape(-1, p1.shape[-1])
        p2 = p2.reshape(-1, p1.shape[-1])
        p3 = p3.reshape(-1, p1.shape[-1])
        p4 = p4.reshape(-1, p1.shape[-1])

        agent_num, dim = p1.shape

        rect_list = []
        for i in range(agent_num):
            rect = np.stack([p1[i], p2[i], p3[i], p4[i]])
            rect_list.append(rect)
        return rect_list

    def get_polygon(self):
        """
        Get the polygon representation of the agent.

        Returns:
            list: A list of polygon representations of the agent.
        """

        rect_list = self.get_rect(pad=0.2)

        poly_list = []
        for i in range(len(rect_list)):
            a = rect_list[i][0]
            b = rect_list[i][1]
            c = rect_list[i][2]
            d = rect_list[i][3]
            poly_list.append(Polygon([a, b, c, d]))

        return poly_list


class actDataset(Dataset):
    """
    A custom dataset class for loading and processing data for the act model.

    Args:
        cfg (dict): Configuration parameters for the dataset.
        vis_test_set_indices (list): List of indices for the test set in visualization mode.

    Attributes:
        total_data_usage (str): Data usage information.
        data_path (str): Path to the data.
        pred_len (int): Length of the prediction.
        data_len (int): Length of the loaded data.
        data_loaded (list): List of loaded data.
        cfg (dict): Configuration parameters for the dataset.
        black_list (list): List of blacklisted files.
        agent_type (str): Type of the agent.
        use_cache (bool): Flag indicating whether to use cache.
        vis (bool): Flag indicating whether in visualization mode.
        keep_all_future (bool): Flag indicating whether to keep all future data.
        vis_test_set_indices (list): List of indices for the test set in visualization mode.
        cal_scr (bool): Flag indicating whether to calculate SCR.
        mixture_data (bool): Flag indicating whether to use mixture data.

    Methods:
        load_data(vis): Loads the data from cache or raw data.
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the data at the specified index.
        process_scene(data): Processes the scene data.
        rotate(x, y, angle): Rotates the coordinates.
        process_agent(data): Processes the agent data.
        get_inp_gt(case_info, agent, agent_mask): Gets the input and ground truth data.

    """

    def __init__(self, cfg, vis_test_set_indices=[]):
        super().__init__()
        self.total_data_usage = cfg["data_usage"]
        self.data_path = cfg['data_path']
        self.pred_len = cfg['pred_len']
        self.data_len = None
        self.data_loaded = []
        self.cfg = cfg
        self.black_list = []
        self.agent_type = cfg['agent_type']
        self.use_cache = cfg['use_cache']
        self.vis = cfg['vis']
        self.keep_all_future = cfg['keep_all_future']
        self.vis_test_set_indices = vis_test_set_indices
        self.cal_scr = cfg['cal_scr']
        self.mixture_data = cfg['mixture_data']
        if self.vis:
            self.vis_test_set_indices = vis_test_set_indices
        self.load_data(self.vis)

    def load_data(self, vis):
        """
        Load data from cache or raw data files.

        Args:
            vis (bool): Flag indicating whether to load visualization data.

        Returns:
            None
        """
        if self.cfg['use_cache']:
            if vis:
                cache_name = 'act_cache_vis_' + self.agent_type + '.pkl'
            else:
                cache_name = 'act_cache_' + self.agent_type + '.pkl'
            data_path = os.path.join(self.data_path, cache_name)
            if self.cal_scr:
                data_path = self.data_path
            if self.mixture_data:
                cache_name = 'act_cache_combine.pkl'
                data_path = os.path.join(self.data_path, cache_name)
            print('loading data from cache at', data_path)
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    self.data_loaded = pickle.load(f)
                # self.data_loaded = self.data_loaded[::100]
                self.data_len = len(self.data_loaded)
                print('Length of data:', len(self.data_loaded))

        else:
            print('No cache found, loading data from raw data')
            src_files = glob.glob(os.path.join(self.data_path, '*.pkl'))
            exclude_prefix = 'cache'
            if vis:
                if len(self.vis_test_set_indices) < 60:
                    src_files = [f for i, f in enumerate(glob.glob(os.path.join(self.data_path, '*.pkl'))) if exclude_prefix not in f]
                else:
                    src_files = [f for i, f in enumerate(glob.glob(os.path.join(self.data_path, '*.pkl'))) if exclude_prefix not in f and i % 60 == 0]

                # src_files = [f for i, f in enumerate(glob.glob(os.path.join(self.data_path, '*.pkl'))) if exclude_prefix not in f]
                if len(self.vis_test_set_indices) > 0:
                    # src_files = [os.path.join(self.data_path, str(f)+ '.pkl') for idx, f in enumerate(self.vis_test_set_indices) if idx % 20 == 0]
                    src_files = [os.path.join(self.data_path, str(f)+ '.pkl') for idx, f in enumerate(self.vis_test_set_indices)]

            else:
                src_files = [f for i, f in enumerate(glob.glob(os.path.join(self.data_path, '*.pkl'))) if exclude_prefix not in f and i % 2 == 0] # sample 3*2, for saving time
            src_files.sort()
            length = len(src_files)
            for file_idx in tqdm(range(length)):
                with open(src_files[file_idx], 'rb+') as f:
                    datas = pickle.load(f)
                if datas == {}:
                    self.black_list.append(os.path.splitext(os.path.basename(src_files[file_idx]))[0])
                    print('Adding file to black list: ', self.black_list)
                    continue

                if self.agent_type=='ego':
                    datas = self.process(datas)
                elif self.agent_type=='pedestrian':  
                    datas = self.process_pedestrian(datas)
                elif self.agent_type=='cyclist':  
                    datas = self.process_cyclist(datas)

                if datas == None:
                    continue
                wash(datas)
                self.data_loaded.append(datas)

            self.data_len = length
            if vis:
                cache_name = 'act_cache_vis_' + self.agent_type + '.pkl'
            else:
                cache_name = 'act_cache_' + self.agent_type + '.pkl'
            data_path = os.path.join(self.data_path, cache_name)
            print('Saving cache at', data_path, '...')
            print('Length of data:', len(self.data_loaded))
            with open(data_path, 'wb') as f:
                pickle.dump(self.data_loaded, f)
            

    def __len__(self):
        # debug set length=478
        return self.data_len

    def __getitem__(self, index):
        """
        Calculate for saving spaces
        """
        return self.data_loaded[index]

    def process_scene(self, data):
        """
        Process the scene data and return a dictionary containing relevant information.

        Args:
            data (dict): A dictionary containing scene data.

        Returns:
            dict: A dictionary containing processed scene information.

        """
        case = {}

        sdc_theta = data['sdc_theta']
        pos = data['sdc_pos']
        all_agent = np.concatenate([data['ego_p_c_f'][np.newaxis], data['nbrs_p_c_f']], axis=0)
        coord = self.rotate(all_agent[..., 0], all_agent[..., 1], -sdc_theta) + pos
        vel = self.rotate(all_agent[..., 2], all_agent[..., 3], -sdc_theta)
        yaw = -sdc_theta + np.pi / 2
        all_agent[..., 4] = all_agent[..., 4] + yaw
        all_agent[..., :2] = coord
        all_agent[..., 2:4] = vel
        pred_list = np.append(np.array([1]), data['pred_list']).astype(bool)
        all_agent = all_agent[pred_list][:, 0]

        valid_mask = all_agent[..., -1] == 1.
        type_mask = all_agent[:, -2] == 1.
        mask = valid_mask * type_mask
        all_agent = all_agent[mask]

        case['all_agent'] = all_agent
        case['lane'] = data['lane']
        case['traf'] = data['traf_p_c_f']
        return case

    def rotate(self, x, y, angle):
        """
        Rotate the coordinates (x, y) by the given angle.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
            angle (float): The angle of rotation in radians.

        Returns:
            numpy.ndarray: The rotated coordinates.

        """
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
        return output_coords

    def process_agent(self, data):
        '''
        Transform the agent data to the ego frame.

        Args:
            data (dict): A dictionary containing the agent data.

        Returns:
            tuple: A tuple containing the transformed agent data and a mask indicating missing elements.

        Notes:
            - The agent data should have the shape (190, 36, 9).
            - The returned agent data will be in the ego frame.
            - Elements in the mask with a value of 0 indicate missing data.

        '''
        agent = data['all_agent'] # (190, 36, 9)
        ego = agent[:, 0]

        ego_pos = copy.deepcopy(ego[[0], :2])[:, np.newaxis]
        ego_heading = ego[[0], [4]]

        agent[..., :2] -= ego_pos
        agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
        agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
        agent[..., 4] -= ego_heading

        agent_mask = agent[..., -1]
        agent_range_mask = (abs(agent[..., 0] - 40) < RANGE) * (abs(agent[..., 1]) < RANGE)
        mask = agent_mask * agent_range_mask

        return agent, mask.astype(bool)

    def get_inp_gt(self, case_info, agent, agent_mask):
        
        """
        Input  agent in ego frame and mask whose 0 inside means missing
        Return case_info stored agent ego and gt
        """
        agent_mask_cp = copy.deepcopy(agent_mask)
        ego_context = agent[:, 0]  # Ego is at the first index
        
        #np.concatenate([pos, velo, self.heading, self.type, self.length_width], axis=-1)
        ego_context = WaymoAgent(ego_context).get_inp(act_inp=True) # pos, vel are normalized here
        ego_future = ego_context[:self.pred_len]
        ego_context = ego_context[:self.pred_len]
        # ego_context = np.pad(agent_context, ([0, MAX_AGENT - agent_context.shape[0]], [0, 0]))
        ego_mask = agent_mask[:self.pred_len, 0]

        if np.count_nonzero(ego_mask) <= MIN_VALID_FRAMES:
            print('Valid states of this segment is too less, skip it')
            return None

        agent_context = copy.deepcopy(agent[0])
        agent_mask = copy.deepcopy(agent_mask[0])
        agent_context = agent_context[agent_mask]
        agent_context = agent_context[:MAX_AGENT]
        agent_mask = agent_mask[agent_mask][:MAX_AGENT]
        agent_context = WaymoAgent(agent_context)
        agent_context = agent_context.get_inp(act_inp=True)  # normalized here
        agent_context = np.pad(agent_context, ([0, MAX_AGENT - agent_context.shape[0]], [0, 0]))
        agent_mask = np.pad(agent_mask, ([0, MAX_AGENT - agent_mask.shape[0]]))
        
        agent_context[:, 4] = cal_rel_dir(agent_context[:, 4], 0) 
        case_info['agent'] = agent_context # (32, 8)
        case_info['agent_mask'] = agent_mask

        if not self.vis:    
            case_info['gt_pos'] = ego_future[:, :2]  # -ego_future[:-1,:2]
            case_info['gt_vel'] = ego_future[:, 2:4]
            case_info['gt_heading'] = cal_rel_dir(ego_future[:, 4], 0)
        else:
            if not self.keep_all_future:
                agent_future = WaymoAgent(agent).get_inp(act_inp=True)
                agent_future_mask = agent_mask_cp
                agent_future_mask_0 = agent_future_mask[0]
                agent_future = agent_future[:, agent_future_mask_0][:, :MAX_AGENT]
                agent_future_mask = agent_future_mask[:, agent_future_mask_0][:, :MAX_AGENT]
                
                padded_agent_future = np.pad(agent_future, ((0, 0), (0, MAX_AGENT - agent_future.shape[1]), (0, 0)), mode='constant', constant_values=0)
                padded_agent_future_mask = np.pad(agent_future_mask, ((0, 0), (0, MAX_AGENT - agent_future_mask.shape[1])), 'constant')
                assert padded_agent_future.shape[1] == MAX_AGENT
                assert padded_agent_future_mask.shape[1] == MAX_AGENT
                case_info['gt_pos'] = padded_agent_future[-1, :MAX_AGENT, :2]
                case_info['gt_vel'] = padded_agent_future[-1, :MAX_AGENT, 2:4]
                case_info['gt_heading'] = cal_rel_dir(padded_agent_future[-1, :MAX_AGENT, 4], 0)
                case_info['gt_mask'] = padded_agent_future_mask

            else:
                agent_future = WaymoAgent(agent).get_inp(act_inp=True)
                agent_future_mask = agent_mask_cp
                agent_future_mask_0 = agent_future_mask[0]
                agent_future = agent_future[:, agent_future_mask_0][:, :MAX_AGENT]
                agent_future_mask = agent_future_mask[:, agent_future_mask_0][:, :MAX_AGENT]
                
                padded_agent_future = np.pad(agent_future, ((0, 0), (0, MAX_AGENT - agent_future.shape[1]), (0, 0)), mode='constant', constant_values=0)
                padded_agent_future_mask = np.pad(agent_future_mask, ((0, 0), (0, MAX_AGENT - agent_future_mask.shape[1])), 'constant')
                assert padded_agent_future.shape[1] == MAX_AGENT
                assert padded_agent_future_mask.shape[1] == MAX_AGENT
                case_info['gt_pos'] = padded_agent_future[:, :MAX_AGENT, :2]
                case_info['gt_vel'] = padded_agent_future[:, :MAX_AGENT, 2:4]
                case_info['gt_heading'] = cal_rel_dir(padded_agent_future[:, :MAX_AGENT, 4], 0)
                case_info['gt_mask'] = padded_agent_future_mask

        case_info['ego_mask'] = ego_mask

        return agent_context, agent_mask

    def transform_coordinate_map(self, data):
        """
        Transforms the coordinate map based on the ego vehicle's position and heading.

        Args:
            data (dict): A dictionary containing the data for the coordinate map transformation.

        Returns:
            None

        Notes:
            - Modifies the 'lane' key in the 'data' dictionary in-place.
            - The 'lane' key should contain a numpy array representing the lane coordinates.

        """
        timestep = data['all_agent'].shape[0]

        ego = data['all_agent'][:, 0]
        pos = ego[:, [0, 1]][:, np.newaxis]

        lane = data['lane'][np.newaxis]
        lane = np.repeat(lane, timestep, axis=0)
        lane[..., :2] -= pos

        x = lane[..., 0]
        y = lane[..., 1]
        ego_heading = ego[:, [4]]
        lane[..., :2] = rotate(x, y, -ego_heading)

        data['lane'] = lane

    def process_cyclist(self, data):
        """
        Process the cyclist agent in the given data.

        Args:
            data (dict): The input data dictionary containing the necessary information.

        Returns:
            dict or None: A dictionary containing the processed case information if the cyclist is present in the scene,
            otherwise None.

        Notes:
            - This method checks if the type of the agent is cyclist and if its length is greater than MIN_VALID_FRAMES.
            - If the conditions are met, it swaps the index between the ego and cyclist agents.
            - The method also performs additional processing on the data, such as transforming coordinates and processing the map.

        """
        case_info = {}
        if self.vis or self.cal_scr:
            case_info['traf'] = data['traffic_light']
        data['all_agent'] = data['all_agent_sliced']
        all_agent_0 = data['all_agent'][0, :]
        all_agent_0_type = all_agent_0[:, -2]
        indices = (all_agent_0_type == 3).nonzero()[0] 

        if not len(indices)>0:
            print('No cyclist in this scene')
            return None
        else:
            switched = False
            for index in indices:
                # if the first element of the current column is 0, skip the current loop
                if data["all_agent"][0, index, -1] == 0:
                    continue
                # if the last element of the current column is 0, skip the current loop
                if data["all_agent"][-1, index, -1] == 0:
                    continue
                # if the last element of the current column is less than 30, skip the current loop
                if sum(data["all_agent"][:, index, -1]) <= MIN_VALID_FRAMES:
                    continue
                # if the above conditions are not met, swap the elements of the current column and the ego column
                else:
                    deleted_column = np.delete(data["all_agent"], index, axis=1)
                    data["all_agent"] = np.insert(deleted_column, 0, data["all_agent"][:, index], axis=1)
                    switched = True
            if not switched:
                return None

        self.transform_coordinate_map(data)
        if self.vis or self.cal_scr:
            case_info['lane'] = data['lane']
        case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
        case_info['cross'], case_info['cross_mask'], case_info['rest'], case_info['rest_mask'] = process_map(
            data['lane'][[0]], [data['traffic_light'][0]], center_num=256, edge_num=128, offest=-40, lane_range=60)

        case_info['center'] = case_info['center'][0]
        case_info['center_mask'] = case_info['center_mask'][0]
        case_info['bound'] = case_info['bound'][0]
        case_info['bound_mask'] = case_info['bound_mask'][0]
        case_info['cross'] = case_info['cross'][0]
        case_info['cross_mask'] = case_info['cross_mask'][0]
        case_info['rest'] = case_info['rest'][0]
        case_info['rest_mask'] = case_info['rest_mask'][0]

        agent, agent_mask = self.process_agent(data)
        if self.get_inp_gt(case_info, agent, agent_mask)==None:
            return None

        return case_info

    def process_pedestrian(self, data):
            """
            Process the pedestrian data in the given dataset.

            Args:
                data (dict): The input data dictionary containing the pedestrian information.

            Returns:
                dict or None: The processed case information dictionary if pedestrian data is present, 
                              otherwise None.

            Notes:
                - The method checks if the type of agent is pedestrian and if the valid length of frames 
                  is greater than MIN_VALID_FRAMES.
                - If the above conditions are met, it swaps the index between the ego and the pedestrian.
                - The method also processes the map information and the agent information.

            """
            data['all_agent'] = data['all_agent_sliced']
            case_info = {}
            if self.vis or self.cal_scr:
                case_info['traf'] = data['traffic_light']
            
            all_agent_0 = data['all_agent'][0, :]
            all_agent_0_type = all_agent_0[:, -2]
            indices = (all_agent_0_type == 2).nonzero()[0] 
            if not len(indices)>0:
                print('No pedestrian in this scene')
                return None
            else:
                switched = False
                for index in indices:
                    # if the first element of the current column is 0, skip the current loop
                    if data["all_agent"][0, index, -1] == 0:
                        continue
                    # if the last element of the current column is 0, skip the current loop
                    if data["all_agent"][-1, index, -1] == 0:
                        continue
                    # if the last element of the current column is less than 30, skip the current loop
                    if sum(data["all_agent"][:, index, -1]) <= MIN_VALID_FRAMES:
                        continue
                    # if the above conditions are not met, swap the elements of the current column and the ego column
                    else:
                        deleted_column = np.delete(data["all_agent"], index, axis=1)
                        data["all_agent"] = np.insert(deleted_column, 0, data["all_agent"][:, index], axis=1)
                        switched = True
                if not switched:
                    return None
                
            self.transform_coordinate_map(data)
            if self.vis or self.cal_scr:
                case_info['lane'] = data['lane']
            case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
            case_info['cross'], case_info['cross_mask'], case_info['rest'], case_info['rest_mask'] = process_map(
                data['lane'][[0]], [data['traffic_light'][0]], center_num=256, edge_num=128, offest=-40, lane_range=60)

            case_info['center'] = case_info['center'][0]
            case_info['center_mask'] = case_info['center_mask'][0]
            case_info['bound'] = case_info['bound'][0]
            case_info['bound_mask'] = case_info['bound_mask'][0]
            case_info['cross'] = case_info['cross'][0]
            case_info['cross_mask'] = case_info['cross_mask'][0]
            case_info['rest'] = case_info['rest'][0]
            case_info['rest_mask'] = case_info['rest_mask'][0]
            

            agent, agent_mask = self.process_agent(data)
            if self.get_inp_gt(case_info, agent, agent_mask)==None:
                return None

            return case_info


    def process(self, data):
        """
        Process the data and return the processed case information.

        Args:
            data (dict): The input data dictionary.

        Returns:
            dict: The processed case information.

        """
        case_info = {}
        data['all_agent'] = data['all_agent_sliced']
        if self.vis or self.cal_scr:
            case_info['traf'] = data['traffic_light']

        if data["all_agent"][-1, 0, -1] == 0:
            return None
        self.transform_coordinate_map(data)

        if self.vis or self.cal_scr:
            case_info['lane'] = data['lane']

        case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
        case_info['cross'], case_info['cross_mask'], case_info['rest'], case_info['rest_mask'] = process_map(
            data['lane'][[0]], [data['traffic_light'][0]], center_num=256, edge_num=128, offest=-40, lane_range=60)

        case_info['center'] = case_info['center'][0]
        case_info['center_mask'] = case_info['center_mask'][0]
        case_info['bound'] = case_info['bound'][0]
        case_info['bound_mask'] = case_info['bound_mask'][0]
        case_info['cross'] = case_info['cross'][0]
        case_info['cross_mask'] = case_info['cross_mask'][0]
        case_info['rest'] = case_info['rest'][0]
        case_info['rest_mask'] = case_info['rest_mask'][0]

        agent, agent_mask = self.process_agent(data)
        if self.get_inp_gt(case_info, agent, agent_mask) == None:
            return None

        return case_info


if __name__ == "__main__":
    args = get_parsed_args()
    cfg = load_config_act(args.config, test=True)
    cfg['use_cache'] = False
    act_dataset = actDataset(cfg)
    data_0 = act_dataset[0]
    print('data_loader.keys()', data_0.keys())
