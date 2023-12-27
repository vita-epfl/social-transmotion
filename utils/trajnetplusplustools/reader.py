from collections import defaultdict
import itertools
import json
import random

import numpy as np

from .data import SceneRow, TrackRow_all_jta, TrackRow_jrdb_2dbb_3dbb

class Reader_jrdb_2dbox(object):

    def __init__(self, input_file, scene_type=None, image_file=None):
        if scene_type is not None and scene_type not in {'rows', 'paths', 'tags'}:
            raise Exception('scene_type not supported')
        self.scene_type = scene_type

        self.tracks_by_frame = defaultdict(list)
        self.scenes_by_id = dict()
        self.read_file(input_file)
    def read_file(self, input_file):
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line)

                track = line.get('track')
                if track is not None:
                    row = TrackRow_jrdb_2dbb_3dbb(track['f'], track['p'], track['x'], track['y'],                               
                                    track['h'], track['w'], track['l'], track['rot_z'],
                                    track['bb_left'], track['bb_top'], track['bb_width'], track['bb_height'],
                                   track.get('prediction_number'), track.get('scene_id'))
                    self.tracks_by_frame[row.frame].append(row)
                    continue

                scene = line.get('scene')
                if scene is not None:
                    row = SceneRow(scene['id'], scene['p'], scene['s'], scene['e'], \
                                   scene.get('fps'), scene.get('tag'))
                    self.scenes_by_id[row.scene] = row
    def scenes(self, randomize=False, limit=0, ids=None, sample=None):
        scene_ids = self.scenes_by_id.keys()
        if ids is not None:
            scene_ids = ids
        if randomize:
            scene_ids = list(scene_ids)
            random.shuffle(scene_ids)
        if limit:
            scene_ids = itertools.islice(scene_ids, limit)
        if sample is not None:
            scene_ids = random.sample(scene_ids, int(len(scene_ids) * sample))
        for scene_id in scene_ids:
            yield self.scene(scene_id)
    @staticmethod
    def track_rows_to_paths(primary_pedestrian, track_rows):
        primary_path = []
        other_paths = defaultdict(list)
        for row in track_rows:
            if row.pedestrian == primary_pedestrian:
                primary_path.append(row)
                continue
            other_paths[row.pedestrian].append(row)

        return [primary_path] + list(other_paths.values())
    @staticmethod
    def paths_to_xy(paths):
        """Convert paths to numpy array with nan as blanks."""
        frames = set(r.frame for r in paths[0])
        pedestrians = set(row.pedestrian
                          for path in paths
                          for row in path if row.frame in frames)
        paths = [path for path in paths if path[0].pedestrian in pedestrians]
        frames = sorted(frames)
        pedestrians = list(pedestrians)

        frame_to_index = {frame: i for i, frame in enumerate(frames)}

        xy = np.full((len(frames), len(pedestrians), 8), np.nan)

        for ped_index, path in enumerate(paths):
            for row in path:
                if row.frame not in frame_to_index:
                    continue
                entry = xy[frame_to_index[row.frame]][ped_index]

                ## extract trajectory
                entry[0] = row[2]
                entry[1] = row[3]
                entry[2] = 0.0
                entry[3] = 0.0

                ## extract 2d BB
                for j in range(4):
                    entry[j+4] = row[j+4+4]
             

        return xy
    
    def scene(self, scene_id, total_joints_dim=66):
        scene = self.scenes_by_id.get(scene_id)
        if scene is None:
            raise Exception('scene with that id not found')

        frames = range(scene.start, scene.end + 1)
        track_rows = [r
                      for frame in frames
                      for r in self.tracks_by_frame.get(frame, [])]

        # return as rows
        if self.scene_type == 'rows':
            return scene_id, scene.pedestrian, track_rows

        # return as paths
        paths = self.track_rows_to_paths(scene.pedestrian, track_rows)
        if self.scene_type == 'paths':
            return scene_id, paths

        ## return with scene tag

        if self.scene_type == 'tags':
            return scene_id, scene.tag, self.paths_to_xy(paths)

        # return a numpy array
        return scene_id, self.paths_to_xy(paths)
    



class Reader_jta_all_visual_cues(object):

    def __init__(self, input_file, scene_type=None, image_file=None):
        if scene_type is not None and scene_type not in {'rows', 'paths', 'tags'}:
            raise Exception('scene_type not supported')
        self.scene_type = scene_type

        self.tracks_by_frame = defaultdict(list)
        self.scenes_by_id = dict()
        self.read_file(input_file)
    def read_file(self, input_file):
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line)

                track = line.get('track')
                if track is not None:
                    row = TrackRow_all_jta(track['f'], track['p'], track['x'], track['y'], #0-3
                                    track['h'], track['w'], track['l'], track['rot_z'], #4-7
                                    track['bb_left'], track['bb_top'], track['bb_width'], track['bb_height'], #8-11
                                    #12-77
                                    track['x0'], track['x1'], track['x2'], track['x3'], track['x4'], track['x5'], track['x6'], track['x7'], track['x8'], track['x9'], track['x10'], track['x11'], track['x12'], track['x13'], track['x14'], track['x15'],  track['x16'], track['x17'], track['x18'], track['x19'], track['x20'], track['x21'], 
                                    track['y0'], track['y1'], track['y2'], track['y3'], track['y4'], track['y5'], track['y6'], track['y7'], track['y8'], track['y9'], track['y10'], track['y11'], track['y12'], track['y13'], track['y14'], track['y15'],  track['y16'], track['y17'], track['y18'], track['y19'], track['y20'], track['y21'], 
                                    track['z0'], track['z1'], track['z2'], track['z3'], track['z4'], track['z5'], track['z6'], track['z7'], track['z8'], track['z9'], track['z10'], track['z11'], track['z12'], track['z13'], track['z14'], track['z15'],  track['z16'], track['z17'], track['z18'], track['z19'], track['z20'], track['z21'],
                                    #78-121
                                    track['xx0'], track['xx1'], track['xx2'], track['xx3'], track['xx4'], track['xx5'], track['xx6'], track['xx7'], track['xx8'], track['xx9'], track['xx10'], track['xx11'], track['xx12'], track['xx13'], track['xx14'], track['xx15'],  track['xx16'], track['xx17'], track['xx18'], track['xx19'], track['xx20'], track['xx21'], 
                                    track['yy0'], track['yy1'], track['yy2'], track['yy3'], track['yy4'], track['yy5'], track['yy6'], track['yy7'], track['yy8'], track['yy9'], track['yy10'], track['yy11'], track['yy12'], track['yy13'], track['yy14'], track['yy15'],  track['yy16'], track['yy17'], track['yy18'], track['yy19'], track['yy20'], track['yy21'],                                    
                                    track.get('prediction_number'), track.get('scene_id'))
                    self.tracks_by_frame[row.frame].append(row)
                    continue

                scene = line.get('scene')
                if scene is not None:
                    row = SceneRow(scene['id'], scene['p'], scene['s'], scene['e'], \
                                   scene.get('fps'), scene.get('tag'))
                    self.scenes_by_id[row.scene] = row
    def scenes(self, randomize=False, limit=0, ids=None, sample=None):
        scene_ids = self.scenes_by_id.keys()
        if ids is not None:
            scene_ids = ids
        if randomize:
            scene_ids = list(scene_ids)
            random.shuffle(scene_ids)
        if limit:
            scene_ids = itertools.islice(scene_ids, limit)
        if sample is not None:
            scene_ids = random.sample(scene_ids, int(len(scene_ids) * sample))
        for scene_id in scene_ids:
            yield self.scene(scene_id)
    @staticmethod
    def track_rows_to_paths(primary_pedestrian, track_rows):
        primary_path = []
        other_paths = defaultdict(list)
        for row in track_rows:
            if row.pedestrian == primary_pedestrian:
                primary_path.append(row)
                continue
            other_paths[row.pedestrian].append(row)

        return [primary_path] + list(other_paths.values())
    @staticmethod
    def paths_to_xy(paths):
        """Convert paths to numpy array with nan as blanks."""
        frames = set(r.frame for r in paths[0])
        pedestrians = set(row.pedestrian
                          for path in paths
                          for row in path if row.frame in frames)
        paths = [path for path in paths if path[0].pedestrian in pedestrians]
        frames = sorted(frames)
        pedestrians = list(pedestrians)

        frame_to_index = {frame: i for i, frame in enumerate(frames)}


        input_token_num = 47 # change here to modify different inputs of modalities

        xy = np.full((len(frames), len(pedestrians), 4*input_token_num), np.nan)

        for ped_index, path in enumerate(paths):
            for row in path:
                if row.frame not in frame_to_index:
                    continue
                entry = xy[frame_to_index[row.frame]][ped_index]

                ## extract trajectory
                entry[0] = row[2]
                entry[1] = row[3]
                entry[2] = 0.0
                entry[3] = 0.0

                for b in range(8):
                    entry[b+4] = row[b+4]


                for j in range(22):
                    entry[j*4+12] = row[j+12]
                    entry[j*4+13] = row[j+12+22]
                    entry[j*4+14] = row[j+12+22+22]
                    entry[j*4+15] = 0.0

                    entry[j*4+12+88] = row[j+78]
                    entry[j*4+13+88] = row[j+78+22]
                    entry[j*4+14+88] = 0.0
                    entry[j*4+15+88] = 0.0

        return xy
    
    def scene(self, scene_id, total_joints_dim=66):
        scene = self.scenes_by_id.get(scene_id)
        if scene is None:
            raise Exception('scene with that id not found')

        frames = range(scene.start, scene.end + 1)
        track_rows = [r
                      for frame in frames
                      for r in self.tracks_by_frame.get(frame, [])]

        # return as rows
        if self.scene_type == 'rows':
            return scene_id, scene.pedestrian, track_rows

        # return as paths
        paths = self.track_rows_to_paths(scene.pedestrian, track_rows)
        if self.scene_type == 'paths':
            return scene_id, paths

        ## return with scene tag

        if self.scene_type == 'tags':
            return scene_id, scene.tag, self.paths_to_xy(paths)

        # return a numpy array
        return scene_id, self.paths_to_xy(paths)

