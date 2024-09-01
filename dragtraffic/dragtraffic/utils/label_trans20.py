import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


from waymo_open_dataset.protos import scenario_pb2

from waymo_open_dataset.utils.sim_agents import visualizations
from sklearn.cluster import KMeans


# import pickle
from enum import Enum
import pickle
import datetime
import glob
import fnmatch


SAMPLE_NUM = 10
LANE_DIM = 4
TIME_SAMPLE = 3  # sample 64 time step, 64*3 = 192
BATCH_SIZE = 190  # 64*3 < 192, ok


def read_trajectory_file(filename):
    """
    Reads a file containing multiple trajectories, where each line is a trajectory, and the first line is the traj num.
    :param filename: The name of the file
    :return: A list of trajectories, or None if the file does not exist
    """
    # Read trajectories from the file
    print('Loading file from ', filename)
    with open(filename, 'r') as f:
        # Read the number of trajectories
        line = f.readline()
        cnt = int(line.split(': ')[-1])

        # Read the list of trajectories
        relative_pos_list = []
        for i in range(cnt):
            # Read a trajectory
            traj = []
            while True:
                line = f.readline().strip()
                if not line:
                    break
                row = [float(x) for x in line.split('\t')]
                traj.append(row)
            relative_pos_list.append(np.array(traj))

    return relative_pos_list

def get_representative_trajectories_by_kmeans(trajectories, k_clusters=10, verbose=False):
    """
    Converts trajectories to feature vectors and clusters them using KMeans algorithm.
    :param trajectories: The trajectories to cluster
    :param k_clusters: The number of clusters
    :param verbose: If True, print verbose information
    :return: The representative trajectories of the clusters and the KMeans instance
    """

    # Convert trajectories to feature vectors
    X = np.array([traj.flatten() for traj in trajectories])

    # Cluster using KMeans algorithm
    kmeans = KMeans(n_clusters=k_clusters, verbose=verbose)
    kmeans.fit(X)

    # Get representative trajectories of the clusters
    representative_trajectories = []
    for i in range(k_clusters):
        traj_indices = np.where(kmeans.labels_ == i)[0]
        traj_mean = np.mean(X[traj_indices], axis=0).reshape(-1, 2)
        representative_trajectories.append(traj_mean)

    return representative_trajectories, kmeans

def plot_track_trajectory(track: scenario_pb2.Track, ax, simu_length=None) -> None:
    valids = np.array([state.valid for state in track.states])
    if np.any(valids):
        x = np.array([state.center_x for state in track.states])
        y = np.array([state.center_y for state in track.states])
        if simu_length != None:
            ax.plot(x[valids][:simu_length], y[valids][:simu_length], linewidth=5)
        else:
            ax.plot(x[valids], y[valids], linewidth=5)

def plot_ego_trajectory(track, ax, color='b', linewidth=1, draw_kmeans=False) -> None:
    """
    Plots the ego vehicle's trajectory.
    :param track: The trajectory to plot
    :param ax: The matplotlib axis to plot on
    :param color: The color of the trajectory
    :param linewidth: The width of the trajectory line
    :param draw_kmeans: If True, draw with KMeans
    """
    length = len(track)
    if draw_kmeans == True:
        transparent_max = 1
    else:
        transparent_max = 0.5
    alpha_values = np.exp(np.linspace(np.log(transparent_max), np.log(0.1), length-1))
    for i in range(length-1):
        alpha = alpha_values[i]  # Calculate transparency
        ax.plot([track[i, 0], track[i+1, 0]], [track[i, 1], track[i+1, 1]], linewidth=linewidth, color=color, alpha=alpha)

def yaw_to_y(angles):
    """
    Converts yaw angles to y-axis angles.
    :param angles: The yaw angles to convert
    :return: The converted angles
    """
    ret = []
    for angle in angles:
        angle = trans_angle(angle)
        angle_to_y = angle - np.pi / 2
        angle_to_y = -1 * angle_to_y
        ret.append(angle_to_y)
    return np.array(ret)

class RoadLineType(Enum):
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    @staticmethod
    def is_road_line(line):
        """
        Checks if the given line is a road line.
        :param line: The line to check
        :return: True if it is a road line, False otherwise
        """
        return True if line.__class__ == RoadLineType else False

    @staticmethod
    def is_yellow(line):
        """
        Checks if the given line is yellow.
        :param line: The line to check
        :return: True if it is yellow, False otherwise
        """
        return True if line in [
            RoadLineType.SOLID_DOUBLE_YELLOW, RoadLineType.PASSING_DOUBLE_YELLOW, RoadLineType.SOLID_SINGLE_YELLOW,
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW
        ] else False

    @staticmethod
    def is_broken(line):
        """
        Checks if the given line is broken.
        :param line: The line to check
        :return: True if it is broken, False otherwise
        """
        return True if line in [
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW, RoadLineType.BROKEN_SINGLE_WHITE
        ] else False

def yaw_to_theta(angles, thetas):
    """
    Converts yaw angles to theta angles in the time horizon.
    :param angles: The yaw angles
    :param thetas: The theta angles
    :return: The converted angles
    """
    ret = []
    for theta, angle in zip(thetas, angles):
        theta = trans_angle(theta)
        angle -= theta
        angle = trans_angle(angle)
        if angle >= np.pi:
            angle -= 2 * np.pi
        ret.append(angle)
    return np.array(ret)

def trans_angle(angle):
    """
    Transforms angles into the range 0~2pi.
    :param angle: The angle to transform
    :return: The transformed angle
    """
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle

def transform_coord(coords, angle):
    """
    Transforms coordinates based on the given angle.
    :param coords: The coordinates to transform
    :param angle: The angle to use for transformation
    :return: The transformed coordinates
    """
    x = coords[..., 0]
    y = coords[..., 1]
    x_transform = np.cos(angle) * x - np.sin(angle) * y
    y_transform = np.cos(angle) * y + np.sin(angle) * x
    output_coords = np.stack((x_transform, y_transform), axis=-1)

    if coords.shape[1] == 3:
        output_coords = np.concatenate((output_coords, coords[:, 2:]), axis=-1)
    return output_coords

def extract_tracks(f, sdc_index):
    """
    Extracts tracks from the given frames and separates the ego vehicle.
    :param f: The frames to extract from
    :param sdc_index: The index of the ego vehicle
    :return: The extracted tracks
    """
    agents = np.zeros([len(f), BATCH_SIZE, 9])
    for i in range(len(f)):
        x = np.array([state.center_x for state in f[i].states])
        y = np.array([state.center_y for state in f[i].states])
        l = np.array([[state.length] for state in f[i].states])[:, 0]
        w = np.array([[state.width] for state in f[i].states])[:, 0]
        head = np.array([state.heading for state in f[i].states])
        vx = np.array([state.velocity_x for state in f[i].states])
        vy = np.array([state.velocity_y for state in f[i].states])
        valid = np.array([[state.valid] for state in f[i].states])[:, 0]
        t = np.repeat(f[i].object_type, len(valid))
        agents[i] = np.stack((x, y, vx, vy, head, l, w, t, valid), axis=-1)[:BATCH_SIZE]
    ego = agents[[sdc_index]]
    others = np.delete(agents, sdc_index, axis=0)
    all_agent = np.concatenate([ego, others], axis=0)
    return all_agent.swapaxes(0, 1)

def extract_dynamic(f):
    # dynamics = np.zeros([BATCH_SIZE, 32, 6])
    dynamics = []
    # time_sample = min(int(len(sdc.states)/BATCH_SIZE), TIME_SAMPLE)
    # sdc_x = np.array([state.center_x for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    # sdc_y = np.array([state.center_y for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    # sdc_yaw = np.array([state.heading for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    # sdc_x = np.array([state.center_x for state in sdc.states])[:BATCH_SIZE]
    # sdc_y = np.array([state.center_y for state in sdc.states])[:BATCH_SIZE]
    # sdc_yaw = np.array([state.heading for state in sdc.states])[:BATCH_SIZE]
    # sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)
    for i in range(BATCH_SIZE):
        # states = f[i * time_sample].lane_states
        states = f[i].lane_states
        traf_list = []
        for j in range(len(states)):
            traf = np.zeros(6)
            traf[0] = states[j].lane
            traf[1:4] = np.array([[states[j].stop_point.x, states[j].stop_point.y, 0]])
            if states[j].state in [1, 4, 7]:
                state_ = 1  # stop
            elif states[j].state in [2, 5, 8]:
                state_ = 2  # caution
            elif states[j].state in [3, 6]:
                state_ = 3  # go
            else:
                state_ = 0  # unknown
            traf[4] = state_
            traf[5] = 1 if states[j].state else 0
            traf_list.append(traf)
        dynamics.append(traf_list)
    return dynamics


def extract_poly(message):
    x = [i.x for i in message]
    y = [i.y for i in message]
    z = [i.z for i in message]
    coord = np.stack((x, y, z), axis=1)

    return coord


def down_sampling(line, type=0):
    # if is center lane
    point_num = len(line)

    ret = []

    if point_num < SAMPLE_NUM or type == 1:
        for i in range(0, point_num):
            ret.append(line[i])
    else:
        for i in range(0, point_num, SAMPLE_NUM):
            ret.append(line[i])

    return ret


def extract_boundaries(fb):
    b = []
    # b = np.zeros([len(fb), 4], dtype='int64')
    for k in range(len(fb)):
        c = dict()
        c['index'] = [fb[k].lane_start_index, fb[k].lane_end_index]
        c['type'] = RoadLineType(fb[k].boundary_type)
        c['id'] = fb[k].boundary_feature_id
        b.append(c)

    return b


def extract_neighbors(fb):
    nbs = []
    for k in range(len(fb)):
        nb = dict()
        nb['id'] = fb[k].feature_id
        nb['indexes'] = [
            fb[k].self_start_index, fb[k].self_end_index, fb[k].neighbor_start_index, fb[k].neighbor_end_index
        ]
        nb['indexes'] = [
            fb[k].self_start_index, fb[k].self_end_index, fb[k].neighbor_start_index, fb[k].neighbor_end_index
        ]
        nb['boundaries'] = extract_boundaries(fb[k].boundaries)
        nb['id'] = fb[k].feature_id
        nbs.append(nb)
    return nbs


def extract_center(f):
    center = {}
    f = f.lane

    poly = down_sampling(extract_poly(f.polyline)[:, :2])
    poly = [np.insert(x, 2, f.type) for x in poly]

    center['interpolating'] = f.interpolating

    center['entry'] = [x for x in f.entry_lanes]

    center['exit'] = [x for x in f.exit_lanes]

    center['left_boundaries'] = extract_boundaries(f.left_boundaries)

    center['right_boundaries'] = extract_boundaries(f.right_boundaries)

    center['left_neighbor'] = extract_neighbors(f.left_neighbors)

    center['right_neighbor'] = extract_neighbors(f.right_neighbors)

    return poly, center


def extract_line(f):
    f = f.road_line
    poly = down_sampling(extract_poly(f.polyline)[:, :2])
    type = f.type + 5
    poly = [np.insert(x, 2, type) for x in poly]
    return poly


def extract_edge(f):
    f = f.road_edge
    poly = down_sampling(extract_poly(f.polyline)[:, :2])
    type = 15 if f.type == 1 else 16
    poly = [np.insert(x, 2, type) for x in poly]

    return poly


def extract_stop(f):
    f = f.stop_sign
    ret = np.array([f.position.x, f.position.y, 17])

    return [ret]


def extract_crosswalk(f):
    f = f.crosswalk
    poly = down_sampling(extract_poly(f.polygon)[:, :2], 1)
    poly = [np.insert(x, 2, 18) for x in poly]
    return poly


def extract_bump(f):
    f = f.speed_bump
    poly = down_sampling(extract_poly(f.polygon)[:, :2], 1)
    poly = [np.insert(x, 2, 19) for x in poly]
    return poly


def extract_map(f):
    maps = []
    center_infos = {}
    # nearbys = dict()
    for i in range(len(f)):
        id = f[i].id

        if f[i].HasField('lane'):
            line, center_info = extract_center(f[i])
            center_infos[id] = center_info

        elif f[i].HasField('road_line'):
            line = extract_line(f[i])

        elif f[i].HasField('road_edge'):
            line = extract_edge(f[i])

        elif f[i].HasField('stop_sign'):
            line = extract_stop(f[i])

        elif f[i].HasField('crosswalk'):
            line = extract_crosswalk(f[i])

        elif f[i].HasField('speed_bump'):
            line = extract_bump(f[i])
        else:
            continue

        line = [np.insert(x, 3, id) for x in line]
        maps = maps + line

    return np.array(maps), center_infos


def transform_coordinate_map(map, sdc):
    """
    Every frame is different
    """
    time_sample = min(int(len(sdc.states) / BATCH_SIZE), TIME_SAMPLE)
    sdc_x = np.array([state.center_x for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    sdc_y = np.array([state.center_y for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    sdc_yaw = np.array([state.heading for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)
    pos = np.stack([sdc_x, sdc_y], axis=-1)
    ret = np.zeros(shape=(BATCH_SIZE, *map.shape))
    for i in range(BATCH_SIZE):
        ret[i] = map
        ret[i][..., :2] = transform_coord(ret[i][..., :2] - pos[i], np.expand_dims(sdc_theta[i], -1))

    # ret[abs(ret[:, :, :, 1]) > 80,-1] =0
    # ret[abs(ret[:, :, :, 2]) > 80, -1] = 0
    valid_ret = np.sum(ret[..., -1], -1)
    lane_mask = valid_ret.astype(bool)
    ret[ret[..., -1] == 0, :] = 0

    return ret, lane_mask


def add_traff_to_lane(scene):
    traf = scene['traf_p_c_f']
    lane = scene['lane']
    traf_buff = np.zeros([*lane.shape[:2]])
    for i in range(BATCH_SIZE):
        lane_i_id = lane[i, :, 0, -1]
        for a_traf in traf[i]:
            lane_id = a_traf[0]
            state = a_traf[-2]
            lane_idx = np.where(lane_i_id == lane_id)
            traf_buff[i, lane_idx] = state
    return traf_buff


def nearest_point(point, line):
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return np.argmin(dist)


def extract_width(map, polyline, boundary):
    l_width = np.zeros(polyline.shape[0])
    for b in boundary:
        idx = map[:, -1] == b['id']
        b_polyline = map[idx][:, :2]

        start_p = polyline[b['index'][0]]
        start_index = nearest_point(start_p, b_polyline)
        seg_len = b['index'][1] - b['index'][0]
        end_index = min(start_index + seg_len, b_polyline.shape[0] - 1)
        leng = min(end_index - start_index, b['index'][1] - b['index'][0]) + 1
        self_range = range(b['index'][0], b['index'][0] + leng)
        bound_range = range(start_index, start_index + leng)
        centerLane = polyline[self_range]
        bound = b_polyline[bound_range]
        dist = np.square(centerLane - bound)
        dist = np.sqrt(dist[:, 0] + dist[:, 1])
        l_width[self_range] = dist
    return l_width


def compute_width(scene):
    lane = scene['unsampled_lane']
    lane_id = np.unique(lane[..., -1]).astype(int)
    center_infos = scene['center_info']

    for id in lane_id:
        if not id in center_infos.keys():
            continue
        id_set = lane[..., -1] == id
        points = lane[id_set]

        width = np.zeros((points.shape[0], 2))

        width[:, 0] = extract_width(lane, points[:, :2], center_infos[id]['left_boundaries'])
        width[:, 1] = extract_width(lane, points[:, :2], center_infos[id]['right_boundaries'])

        width[width[:, 0] == 0, 0] = width[width[:, 0] == 0, 1]
        width[width[:, 1] == 0, 1] = width[width[:, 1] == 0, 0]

        center_infos[id]['width'] = width
    return


def get_kmeans_model_from_ego_traj(inut_path, output_path, \
                                   use_kmeans, k_clusters_direciton,\
                                    k_clusters_behavior, draw_ego_only,\
                                    draw_scene_pic, debug_vis, \
                                    pre_fix=None,debug_vis_files_num=None, simu_len=None):
    # This function processes ego vehicle trajectories to generate KMeans models based on direction and behavior.
    # It optionally visualizes the trajectories and saves the models and representative trajectories.
    
    # Get list of all files in the input directory
    file_list_all = os.listdir(inut_path)
    print('len file_list is', len(file_list_all))
    # Set the number of files to visualize if not specified
    debug_vis_files_num = len(file_list_all) if debug_vis_files_num == None else debug_vis_files_num
    
    cnt = 0  # Counter for processed files
    scenario = scenario_pb2.Scenario()  # Protocol buffer for scenario data
    now = datetime.datetime.now().strftime('%m-%d_%H-%M')  # Current timestamp for file naming
    # File names for saving trajectories
    ego_relative_pos_traj_file_name = f'ego_relative_pos_list_filenum_{debug_vis_files_num}.txt'
    ego_centric_relative_pos_traj_file_name = f'ego_centric_relative_pos_list_filenum_{debug_vis_files_num}.txt'
    # Flags to check if cache exists
    has_relative_pos_list_cache = False
    has_ego_centric_relative_pos_list_cache = False
    # Lists to store trajectories
    relative_pos_list = []
    relative_theta_list = []
    ego_centric_relative_pos_list = []
    # Check for existing trajectory files to avoid reprocessing
    if os.path.exists(ego_relative_pos_traj_file_name) and not draw_scene_pic:
        relative_pos_list = read_trajectory_file(ego_relative_pos_traj_file_name)
        has_relative_pos_list_cache = True
    if os.path.exists(ego_centric_relative_pos_traj_file_name) and not draw_scene_pic:
        ego_centric_relative_pos_list = read_trajectory_file(ego_centric_relative_pos_traj_file_name)
        has_ego_centric_relative_pos_list_cache = True
    # Process files if cache is not found or if scene drawing is requested
    if not has_relative_pos_list_cache or not has_ego_centric_relative_pos_list_cache or draw_scene_pic:
        if debug_vis:
            file_list = file_list_all[:debug_vis_files_num]
        for file in tqdm(file_list):
            file_path = os.path.join(inut_path, file)
            if not 'tfrecord' in file_path:
                continue
            dataset = tf.data.TFRecordDataset(file_path, compression_type='')
            for j, data in enumerate(dataset.as_numpy_iterator()):
                try:
                    # Determine file path for saving processed data
                    if pre_fix == 'None':
                        p = os.path.join(output_path, '{}.pkl'.format(cnt))
                    else:
                        p = os.path.join(output_path, '{}_{}.pkl'.format(pre_fix, cnt))
                    scenario.ParseFromString(data)
                    scene = dict()
                    scene['id'] = scenario.scenario_id
                    sdc_index = scenario.sdc_track_index
                    scene['all_agent'] = extract_tracks(scenario.tracks, sdc_index)
                    ego = scenario.tracks[sdc_index]
                    
                    # Process ego vehicle states for position and orientation
                    time_sample = min(int(len(ego.states) / BATCH_SIZE), TIME_SAMPLE)
                    sdc_x = np.array([state.center_x for state in ego.states])[::time_sample, ...][:BATCH_SIZE]
                    sdc_y = np.array([state.center_y for state in ego.states])[::time_sample, ...][:BATCH_SIZE]
                    sdc_yaw = np.array([state.heading for state in ego.states])[::time_sample, ...][:BATCH_SIZE]
                    sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)
                    
                    pos = np.stack([sdc_x, sdc_y], axis=-1)
                    scene['sdc_theta'] = sdc_theta
                    
                    pos_segment = []
                    theta_segment = []
                    slice_times = 3
                    slice_length = 50
                    # Segment the trajectory for analysis
                    for i in range(slice_times):
                        pos_segment = pos[i*slice_length: i*slice_length + simu_len]
                        theta_segment = sdc_theta[i*slice_length: i*slice_length + simu_len]
                        relative_pos = pos_segment - pos_segment[0]
                        relative_theta = theta_segment - theta_segment[0]
                        
                        relative_pos_list.append(relative_pos)
                        relative_theta_list.append(relative_theta)
                        ego_centric_relative_pos = transform_coord(relative_pos, sdc_theta[0])
                        ego_centric_relative_pos_list.append(ego_centric_relative_pos)
                    
                except:
                    print(f'fail to parse {cnt},continue')
                    continue
                if debug_vis:
                    # Visualization logic for trajectories
                    if draw_scene_pic:
                        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                        visualizations.add_map(ax, scenario)
                        
                        if draw_ego_only:
                            plot_track_trajectory(ego, ax)
                        else:
                            for track in scenario.tracks:
                                plot_track_trajectory(track, ax)
                        plt.show()
                cnt += 1
                if cnt > MAX:
                    break
            if cnt > MAX:
                break
    
    # Save processed trajectories if cache was not used
    if not has_relative_pos_list_cache:
        with open(ego_relative_pos_traj_file_name, 'w') as f:
            f.write(f'# ego trajs num: {cnt}\n')
            for traj in relative_pos_list:
                np.savetxt(f, traj, delimiter='\t')
                f.write('\n')
    if not has_ego_centric_relative_pos_list_cache:
        with open(ego_centric_relative_pos_traj_file_name, 'w') as f:
            f.write(f'# ego trajs num: {cnt}\n')
            for traj in ego_centric_relative_pos_list:
                np.savetxt(f, traj, delimiter='\t')
                f.write('\n')
    # Use KMeans to find representative trajectories and save the models
    if use_kmeans:
        representative_trajectories_list, kmeans_direction = get_representative_trajectories_by_kmeans(relative_pos_list, \
                                                                                                       k_clusters=k_clusters_direciton, verbose=True)
        print('Saving direction kmeans...\n')
        with open(f'kmeans_direction_filenum_{debug_vis_files_num}_cluster_num_{k_clusters_direciton}_{now}.pkl', 'wb') as f:
            pickle.dump(kmeans_direction, f)
        
        print('Saving direction trajs...\n')
        with open(f'representative_trajectories_list_filenum_{debug_vis_files_num}_cluster_num_d{k_clusters_direciton}_b{k_clusters_behavior}_{now}.txt', 'w') as f:
            f.write(f'# ego trajs num: {cnt}\n')
            f.write(f'# representatives num: {len(representative_trajectories_list)}\n')
            for traj in representative_trajectories_list:
                np.savetxt(f, traj, delimiter='\t')
                f.write('\n')
        ego_centric_representative_trajectories_list, kmeans_behavoir = get_representative_trajectories_by_kmeans(ego_centric_relative_pos_list, \
                                                                                                                  k_clusters=k_clusters_behavior, verbose=True)
        print('Saving behavior kmeans...\n')
        with open(f'kmeans_behavior_filenum_{debug_vis_files_num}_cluster_num_{k_clusters_behavior}_{now}.pkl', 'wb') as f:
            pickle.dump(kmeans_behavoir, f)
        print('Saving behavior trajs...\n')
        with open(f'ego_centric_representative_trajectories_list_filenum_{debug_vis_files_num}_cluster_num_d{k_clusters_direciton}_b{k_clusters_behavior}_{now}.txt', 'w') as f:
            f.write(f'# ego trajs num: {cnt}\n')
            f.write(f'# representatives num: {len(ego_centric_representative_trajectories_list)}\n')
            for traj in ego_centric_representative_trajectories_list:
                np.savetxt(f, traj, delimiter='\t')
                f.write('\n')


    if debug_vis:
        # Draw ego-centric trajectories from the whole dataset
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        interval = 1
        # Adjust the interval to avoid running out of memory. Requires 80GB for interval 5.
        if debug_vis_files_num == len(file_list_all):
            interval = 10
            
        print(f'Drawing ego trajs with interval {interval}...\n')
        # Iterate through the list of positions at the specified interval
        for idx in tqdm(range(0, len(relative_pos_list), interval)):
            relative_pos = relative_pos_list[idx]
            ego_centric_relative_pos = ego_centric_relative_pos_list[idx]
            # Plot the trajectories on the left subplot
            plot_ego_trajectory(relative_pos, axs[0])
            # Plot the ego-centric trajectories on the right subplot
            plot_ego_trajectory(ego_centric_relative_pos, axs[1])
        if use_kmeans:
            print('Drawing direction trajs...\n')
            # Plot the representative trajectories for direction using k-means clustering
            for idx, representative_trajectory in enumerate(representative_trajectories_list):
                plot_ego_trajectory(representative_trajectory, axs[0], color='yellow', linewidth=2, draw_kmeans=True)
                # Annotate the end point of each trajectory
                axs[0].text(representative_trajectory[-1, 0], representative_trajectory[-1, 1], f'T{idx+1}', color='black')
                axs[0].scatter(representative_trajectory[-1, 0], representative_trajectory[-1, 1], marker='x', color='black')
            axs[0].set_title('Direction kmeans')

            print('Drawing behavior trajs...\n')
            # Plot the representative trajectories for behavior using k-means clustering
            for idx, ego_centric_representative_trajectory in enumerate(ego_centric_representative_trajectories_list):
                plot_ego_trajectory(ego_centric_representative_trajectory, axs[1], color='yellow', linewidth=2, draw_kmeans=True)
                # Annotate the end point of each trajectory
                axs[1].text(ego_centric_representative_trajectory[-1, 0], ego_centric_representative_trajectory[-1, 1], f'T{idx+1}', color='black')
                axs[1].scatter(ego_centric_representative_trajectory[-1, 0], ego_centric_representative_trajectory[-1, 1], marker='x', color='black')
            axs[1].set_title('Behavior kmeans')

        print('Saving figs...\n')
        # Save the figure with a filename that includes the number of trajectories and the k-means cluster counts
        plt.savefig(f'./traj_dis_trajnum_{len(relative_pos_list)}_d{k_clusters_direciton}_b{k_clusters_behavior}_{now}.png', bbox_inches='tight', pad_inches=0, dpi=200)
        plt.show()
        # Example code for saving data using pickle (commented out)
        # with open(p, 'wb') as f:
        #     pickle.dump(scene, f)

    # Return the k-means clustering results for behavior and direction
    return kmeans_behavior, kmeans_direction


def parse_data_with_label(inut_path, output_path,  kmeans_behavior=None, kmeans_direction=None, pre_fix=None):
    # Define a function to parse data with labels.
    # inut_path: The input file path containing the data to be parsed.
    # output_path: The path where the parsed data with labels should be saved.
    # kmeans_behavior: An optional KMeans clustering model for behavior analysis. If provided, it is used to label the data based on behavior.
    # kmeans_direction: An optional KMeans clustering model for direction analysis. If provided, it is used to label the data based on direction.
    # pre_fix: An optional prefix to add to the output files for identification purposes.

    file_list = os.listdir(inut_path)
    print('len file_list is', len(file_list))
    # debug_vis_files_num = len(file_list) if debug_vis_files_num == None else debug_vis_files_num
    cnt = 0
    scenario = scenario_pb2.Scenario()
    now = datetime.datetime.now().strftime('%m-%d_%H-%M')

    kmeans_behavior = kmeans_behavior
    kmeans_direction = kmeans_direction

    relative_pos_list = []
    relative_theta_list = []
    ego_centric_relative_pos_list = []
    
    for file in tqdm(file_list):
        file_path = os.path.join(inut_path, file)
        if not 'tfrecord' in file_path:
            continue
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        for j, data in enumerate(dataset.as_numpy_iterator()):
            try:
                scenario.ParseFromString(data)
                scene = dict()
                scene['id'] = scenario.scenario_id
                sdc_index = scenario.sdc_track_index
                scene['all_agent'] = extract_tracks(scenario.tracks, sdc_index)

                scene['traffic_light'] = extract_dynamic(scenario.dynamic_map_states)
                global SAMPLE_NUM
                SAMPLE_NUM = 10
                scene['lane'], scene['center_info'] = extract_map(scenario.map_features)
                print('scene lane shape is ', scene['lane'].shape)
                SAMPLE_NUM = 10e9
                scene['unsampled_lane'], _ = extract_map(scenario.map_features)

                compute_width(scene)

                # Get ego traj
                ego = scenario.tracks[sdc_index]
                time_sample = min(int(len(ego.states) / BATCH_SIZE), TIME_SAMPLE)
                sdc_x = np.array([state.center_x for state in ego.states])[::time_sample, ...][:BATCH_SIZE]
                sdc_y = np.array([state.center_y for state in ego.states])[::time_sample, ...][:BATCH_SIZE]
                sdc_yaw = np.array([state.heading for state in ego.states])[::time_sample, ...][:BATCH_SIZE]
                sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)
                pos = np.stack([sdc_x, sdc_y], axis=-1)

                pos_segment = []
                theta_segment = []

                slice_times = 3
                slice_length = 50
                # 1,90, 51,140, 101,190
                # Get ego label
                scene['traj_label'] = []
                scene['all_agent_list'] = []

                for i in range(slice_times):

                    if pre_fix == 'None':
                        p = os.path.join(output_path, '{}.pkl'.format(cnt))
                    else:
                        p = os.path.join(output_path, '{}_{}.pkl'.format(pre_fix, cnt))

                    pos_segment = pos[i*slice_length: i*slice_length + simu_len]
                    theta_segment = sdc_theta[i*slice_length: i*slice_length + simu_len]

                    relative_pos = pos_segment - pos_segment[0]
                    relative_theta = theta_segment - theta_segment[0]
                    
                    relative_pos_list.append(relative_pos)
                    relative_theta_list.append(relative_theta)

                    ego_centric_relative_pos = transform_coord(relative_pos, sdc_theta[0])

                    if kmeans_direction != None and kmeans_behavior != None:

                        scene['traj_label_sliced'] = str(kmeans_direction.predict(ego_centric_relative_pos.reshape(1, -1))[0]) + '_' +\
                                                    str(kmeans_behavior.predict(ego_centric_relative_pos.reshape(1, -1))[0])
                    
                    else:
                        scene['traj_label_sliced'] = str('no_labels')

                    scene['all_agent_sliced'] = scene['all_agent'][i*slice_length: i*slice_length + simu_len] 

                    with open(p, 'wb') as f:
                        pickle.dump(scene, f)

                    cnt += 1
                    if cnt > MAX:
                        break

            except:
                print(f'fail to parse {cnt},continue')
                continue

            if cnt > MAX:
                break

            if debug_vis:
                # Draw traj on each scene
                if draw_scene_pic:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    visualizations.add_map(ax, scenario)
                    
                    if draw_ego_only:
                        plot_track_trajectory(ego, ax)

                    else:
                        for track in scenario.tracks:
                            plot_track_trajectory(track, ax)
                    plt.show()

    return


def load_kmeans(pattern):

    # Search for matching files
    files = glob.glob(pattern)

    if len(files) > 0:
        # Get the first matching file
        filename = files[0]

        # Check if the filename matches the specified pattern
        if fnmatch.fnmatch(os.path.basename(filename), pattern):
            # Load the file
            with open(filename, 'rb') as f:
                kmeans = pickle.load(f)
                print('Loading from', filename)
                return kmeans
    
    else:
        print('No matched model found')

    return None

def get_parsed_args():
    parser = argparse.ArgumentParser(description='Process raw data with optional visualization and k-means clustering.')
    parser.add_argument('raw_data_path', type=str, help='Path to the raw data directory.')
    parser.add_argument('processed_data_path', type=str, help='Path to the directory where processed data will be stored.')
    parser.add_argument('pre_fix', type=str, help='Prefix for processed files.')
    parser.add_argument('--debug_vis', action='store_true', help='Enable debug visualization.')
    parser.add_argument('--debug_vis_files_num', type=int, default=1000, help='Number of files to visualize for debugging.')
    parser.add_argument('--draw_scene_pic', action='store_true', help='Enable drawing of scene pictures.')
    parser.add_argument('--draw_ego_only', action='store_true', help='Draw only the ego vehicle in scene pictures.')
    parser.add_argument('--use_kmeans', action='store_true', help='Use k-means clustering.')
    parser.add_argument('--k_clusters_direction', type=int, default=32, help='Number of clusters for direction k-means.')
    parser.add_argument('--k_clusters_behavior', type=int, default=64, help='Number of clusters for behavior k-means.')
    parser.add_argument('--simu_len', type=int, default=90, help='Simulation length.')
    return parser.parse_args()

if __name__ == '__main__':
    """
    Usage: devide the source data to several pieces, like 10 dirs each containing 100 rf record data, out_put to on dir
    for x in 10
        mkdir raw_x
        move 0-100 raw_x
    then you put data to 10 dir
    for x in 10
        nohup python trans20 ./raw_x ./scenario x > x.log 2>&1 &
    NOTE: 3 *x* to change when manually repeat !!!!!
    ```tail -n 100 -f x.log```  to vis progress of process x

    After almost 30 minutes, all cases are stored in dir scenario
    Then run ```nohup python unify.py scenario > unify.log 2>&1 &``` to unify the name      

    ls -l scenario | wc -l 
    output: 70542   

    Some data may be broken
    """
    # raw_data_path = sys.argv[1]
    # processed_data_path = sys.argv[2]
    # pre_fix = sys.argv[3]
    # raw_data_path = ".\\data"
    # processed_data_path = ".\\debug_data"
    # pre_fix = str(uuid.uuid4())
    #  parse raw data from input path to output path,
    #  there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    args = get_parsed_args()
    MAX = 100000

    relative_pos_list = []
    relative_theta_list = []
    ego_centric_relative_pos_list = []

    raw_data_path = args.raw_data_path
    processed_data_path = args.processed_data_path
    pre_fix = args.pre_fix
    debug_vis = args.debug_vis
    print('Enable debug visualization:', debug_vis)
    debug_vis_files_num = args.debug_vis_files_num
    draw_scene_pic = args.draw_scene_pic
    draw_ego_only = args.draw_ego_only
    use_kmeans = args.use_kmeans
    k_clusters_direction = args.k_clusters_direction
    k_clusters_behavior = args.k_clusters_behavior
    simu_len = args.simu_len

    kmeans_behavior_filename = f'kmeans_behavior_filenum_{debug_vis_files_num}_cluster_num_{k_clusters_behavior}*'
    kmeans_direction_filename = f'kmeans_direction_filenum_{debug_vis_files_num}_cluster_num_{k_clusters_direction}*'

    kmeans_behavior = load_kmeans(kmeans_behavior_filename)
    kmeans_direction = load_kmeans(kmeans_direction_filename)

    if use_kmeans:
        print('Using kmeans labels')
        if kmeans_behavior == None or kmeans_direction == None:
            kmeans_behavior, kmeans_direction = get_kmeans_model_from_ego_traj(raw_data_path, processed_data_path, \
                                        use_kmeans, k_clusters_direction,\
                                        k_clusters_behavior, draw_ego_only,  \
                                        draw_scene_pic, debug_vis, \
                                        pre_fix, debug_vis_files_num, simu_len)

    parse_data_with_label(raw_data_path, processed_data_path, kmeans_behavior, kmeans_direction, pre_fix)
