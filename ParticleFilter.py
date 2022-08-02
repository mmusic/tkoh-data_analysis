from collections import deque as dq
#from itertools import takewhile
from math import sqrt
import numpy as np
import pandas as pd

from Visualization import Plotter

PERIOD = 1
BEACON_WINDOW = 15 # seconds
# MAX_BEACON_GAP = 3 # seconds, equal to BEACON_WINDOW for lack of MPU participation
SWARM_SIZE = 500
EFFECTIVE_SIZE = 300

def cross2d(a,b):
    return a[0] * b[1] - a[1] * b[0]

class PF:
    def __init__(self, ground_info):
        np.random.seed(0)

        self.floor, self.polygons, self.beacons = ground_info

        self._initialize_swarm() # init self.pos, self.vel, self.w
        self._beacon_buffer = dq()
        self._beacon_buffer_t = dq()

        self.tracked = False
        self.event_log = '' # as notations at the corner of plots

        self.pos_estimate = None
        self.vel_estimate = None
        self.speed_estimate = None
        self.pos_var = None

        self.polygon_idx = 0
        
        self.plotter = Plotter((self.polygons, self.beacons))

    def _initialize_swarm(self):
        x = np.random.uniform(0,1,SWARM_SIZE); y = np.random.uniform(0,1,SWARM_SIZE)
        z = x + y > 1
        x[z] = 1 - x[z]; y[z] = 1 - y[z]
        triangles = np.concatenate((self.polygons[:,0:3],self.polygons[:,[2,3,0]]), axis=2)
        triangle_areas = np.abs(cross2d(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])) # S = |OA×OB|/2, but no need to half
        triangle_areas /= triangle_areas.sum()
        triangles = triangles[:,:,np.random.choice(triangles.shape[2], SWARM_SIZE, p=triangle_areas)] # select triangle with area as probability
        self.pos = triangles[:,0] + x * (triangles[:,1] - triangles[:,0]) + y * (triangles[:,2] - triangles[:,0]) # map [0,1]²∩{x+y<1} into the triangle
        self.vel = np.zeros((2, SWARM_SIZE))
        self.w = np.full(SWARM_SIZE, 1/SWARM_SIZE)
    # will refactor
    def feed_data(self, t:int, beacon_input:pd.DataFrame):
        if beacon_input.empty:
            print('empty input')
            return
        # Predict prior 
        self.event_log = ''
        if self.tracked:
            self._blind_predict()
            #print('predict')
        # Update posterior
        #print(self._beacon_buffer)
        #print(t, self._beacon_buffer_t)
        if self._beacon_buffer:
            last_t = self._beacon_buffer_t[-1]
            if t - last_t > BEACON_WINDOW:
                self._set_distracked()
                self.event_log += f'No beacon data for more than {BEACON_WINDOW} seconds.'
            for idx in range(len(self._beacon_buffer)):
                if t - self._beacon_buffer_t[idx] > BEACON_WINDOW:
                    self._beacon_buffer.popleft()
                else:
                    break
            for _ in range(len(self._beacon_buffer_t) - len(self._beacon_buffer)):
                self._beacon_buffer_t.popleft()
        else:
            self._initialize_swarm()
            self.event_log += 'Initialize; '
        self.tracked = True
        self._beacon_buffer.append(beacon_input)
        self._beacon_buffer_t.append(t)
        beacon_batch = pd.concat(self._beacon_buffer, ignore_index=True).groupby('bID').max()
        self._update(beacon_batch)
        self._estimate()
        self._idx_polygon()
        self.plotter.draw_posterior(self, beacon_batch, t) # delete when not used
        self._resample()        
        # can delete
        if self.event_log:
            print(self.event_log)
    
    def _idx_polygon(self):
        P = self.polygons
        P01 = ((P[0,1]-P[0,0])*(self.pos[1].reshape(-1,1)-P[1,0]) - (P[1,1]-P[1,0])*(self.pos[0].reshape(-1,1)-P[0,0])) < 0
        P12 = ((P[0,2]-P[0,1])*(self.pos[1].reshape(-1,1)-P[1,1]) - (P[1,2]-P[1,1])*(self.pos[0].reshape(-1,1)-P[0,1])) < 0
        P23 = ((P[0,3]-P[0,2])*(self.pos[1].reshape(-1,1)-P[1,2]) - (P[1,3]-P[1,2])*(self.pos[0].reshape(-1,1)-P[0,2])) < 0
        P30 = ((P[0,0]-P[0,3])*(self.pos[1].reshape(-1,1)-P[1,3]) - (P[1,0]-P[1,3])*(self.pos[0].reshape(-1,1)-P[0,3])) < 0
        self.polygon_idx = np.argmax((self.w.reshape(-1,1) * ((P01==P12) & (P12==P23) & (P23==P30))).sum(axis=0))

    # # Old algorithm
    # def _update(self, beacon_batch:pd.DataFrame):
    #     b_locs = self.beacons.loc[beacon_batch.index].values.T
    #     rssi = beacon_batch.rssi.values
    #     dist = np.sqrt((b_locs[0] - self.pos[0].reshape(-1,1))**2 + (b_locs[1] - self.pos[1].reshape(-1,1))**2) # cross_distance(self.pos, b_locs)
    #     in_range = (dist<40).any(axis=0) # <40 and >70
    #     rssi = rssi[in_range]; dist = dist[:,in_range]
    #     self.w = (np.clip(1-(dist-5)/35, 0, 1) * np.exp(((rssi+76.0656)*53.171 - (dist-2.8871)*5.218) / 255.212856)).prod(axis=1)
    #     self._normalize_w('Beacon')
    #     self._polygon_filt()
    def _update(self, beacon_batch:pd.DataFrame):
        b_locs = self.beacons.loc[beacon_batch.index].values.T
        rssi = beacon_batch.rssi.values
        dist = np.sqrt((b_locs[0] - self.pos[0].reshape(-1,1))**2 + (b_locs[1] - self.pos[1].reshape(-1,1))**2) # cross_distance(self.pos, b_locs)
        self.w = np.exp((((rssi+85)*11 - (dist-5))/10 * np.clip((rssi+85)/40,0,1)).sum(axis=1))
        self._normalize_w('Beacon')
        self._polygon_filt()

    def _blind_predict(self):
        self.vel = np.random.multivariate_normal((0,0),((1.5,0),(0,1.5)), SWARM_SIZE).T
        self.pos += self.vel * PERIOD
        self._polygon_filt() # necessary?
    
    def _estimate(self):
        "Update self.pos_estimate, self.vel_estimate, self.speed_estimate, and self.pos_var; consider only top EFFECTIVE_SIZE in weight"
        sample_idx = np.argpartition(self.w, -EFFECTIVE_SIZE)[-EFFECTIVE_SIZE:]
        sample_pos = self.pos[:,sample_idx]
        #sample_vel = self.vel[:,sample_idx]
        sample_w = self.w[sample_idx]
        sample_w /= sample_w.sum()

        new_estimate = (sample_pos * sample_w).sum(axis=1)
        self.pos_var = ((np.sqrt(((sample_pos - new_estimate.reshape(2,1))**2).sum(axis=0))) * sample_w).sum()

        if self.pos_estimate is not None:
            self.vel_estimate = (new_estimate - self.pos_estimate) / PERIOD
            self.speed_estimate = sqrt(self.vel_estimate[0]**2 + self.vel_estimate[1]**2)
        self.pos_estimate = new_estimate
                
        #return expectation, ((np.sqrt(((sample_pos - expectation.reshape(2,1))**2).sum(axis=0))) * sample_w).sum()#, (sample_vel * sample_w).sum(axis=1)

    def _resample(self):
        if True: # if (self.w**2).sum() * 4 > 1 or self.w.max() > 0.005:
            self.w **= 2 # force an entropy reduction
            self._normalize_w('resample')
            resample_idx = np.random.choice(SWARM_SIZE, SWARM_SIZE, p=self.w) # np.random.choice(from_size, to_size, p=)
            self.pos = self.pos[:,resample_idx]
            self.vel = self.vel[:,resample_idx]
            self.w.fill(1/SWARM_SIZE)
    
    def _polygon_filt(self): # filter out all points that's not in any polygon    
        # point p is inside a convex and simple polygon P with vertices (P1,P2,P3,...) iff pP1 × P1P2, pP2 × P2P3, ... have same signal
        P = self.polygons
        P01 = ((P[0,1]-P[0,0])*(self.pos[1].reshape(-1,1)-P[1,0]) - (P[1,1]-P[1,0])*(self.pos[0].reshape(-1,1)-P[0,0])) < 0
        P12 = ((P[0,2]-P[0,1])*(self.pos[1].reshape(-1,1)-P[1,1]) - (P[1,2]-P[1,1])*(self.pos[0].reshape(-1,1)-P[0,1])) < 0
        P23 = ((P[0,3]-P[0,2])*(self.pos[1].reshape(-1,1)-P[1,2]) - (P[1,3]-P[1,2])*(self.pos[0].reshape(-1,1)-P[0,2])) < 0
        P30 = ((P[0,0]-P[0,3])*(self.pos[1].reshape(-1,1)-P[1,3]) - (P[1,0]-P[1,3])*(self.pos[0].reshape(-1,1)-P[0,3])) < 0
        self.w *= ((P01==P12) & (P12==P23) & (P23==P30)).any(axis=1)

        self._normalize_w('After filter')

    # call everytime when w is modified
    def _normalize_w(self, error_message:str = ''):
        if self.w.sum() != 0:
            self.w /= self.w.sum()
        else:
            self._set_distracked()
            self.event_log += f'{error_message} lose track; '
    
    def _set_distracked(self):
        self.tracked = False
        self._initialize_swarm()
        self.pos_estimate = None
        self.vel_estimate = None
        self.speed_estimate = None
        self.pos_var = None
        # clear buffer?

    def adsorb_polygon(self):
        e = self.pos_estimate
        if not self.polygon_idx:
            return e
        polygon = self.polygons[:, :, self.polygon_idx]
        c = polygon.mean(axis=1)
        for i in range(4):
            p = polygon[:, i]
            s = polygon[:, i-3] - p
            ep = cross2d(e-p, s)
            cp = cross2d(c-p, s)
            if ep * cp < 0: # not in polygon
                r = cross2d(e-p, c-p) / (ep - cp)
                if 0 <= r <= 1: # line segments (s and (e, c)) intersect
                    return p + r*s
        return e