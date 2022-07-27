'''
TODO: 

'''
# %%
from math import radians, degrees, cos, sqrt
import pandas as pd
import numpy as np
import json
from ParticleFilter import PF, PERIOD

def distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_gi(floor_id):
    "floor_id in {'8F', 'G', 'LG'}"
    with open(f'config/TKOH-{floor_id}.txt', 'r') as gi_file:
        gi = gi_file.read().split('\n\n')
    polygons = eval(gi[0].replace('\n','')[7:])
    polygon_id = np.array(list(polygons.keys()))
    polygons = np.array(tuple(((v[0],v[1]) for p in polygons.values() for v in p))).reshape(len(polygons),4,2).transpose(2,1,0)
    beacon_str = gi[1].replace('\n','')[21:-3].split("),'")
    bx = np.empty(len(beacon_str))
    by = np.empty(len(beacon_str))
    for i, b in enumerate(beacon_str):
        idx = b.index('x=')
        bx[i] = float(b[idx+2:b.index(', ',idx)])
        idx = b.index('y=')
        by[i] = float(b[idx+2:b.index(', ',idx)])
        #idx = b.index('z=')
        #bz[i] = int(b[idx+2:b.index(', ',idx)])
    beacon_loc = pd.DataFrame(columns=['bID',0,1])
    beacon_loc['bID'] = np.array(list(map(lambda b:b[:5], beacon_str)))
    beacon_loc[0] = bx; beacon_loc[1] = by
    beacon_loc = beacon_loc.set_index('bID')
    center = (polygons.max(axis=(1,2))+polygons.min(axis=(1,2)))/2
    def geo2meter(coordinate): # 6371000 = radius of earth in meters
        x = radians(coordinate[0] - center[0]) * 6371000 * cos(radians(center[1]))
        y = radians(coordinate[1] - center[1]) * 6371000
        return (x,y)
    def meter2geo(coordinate):
        lon = center[0] + degrees(coordinate[0] / (6371000 * cos(radians(center[1]))))
        lat = center[1] + degrees(coordinate[1] / 6371000)
        return (lon, lat)
    polygons[0] = np.radians(polygons[0]-center[0]) * 6371000 * cos(radians(center[1]))
    polygons[1] = np.radians(polygons[1]-center[1]) * 6371000
    beacon_loc[0] = np.radians(beacon_loc[0]-center[0]) * 6371000 * cos(radians(center[1]))
    beacon_loc[1] = np.radians(beacon_loc[1]-center[1]) * 6371000
    return polygons, polygon_id, beacon_loc, geo2meter, meter2geo

polygons_8F, polygon_id_8F, beacon_loc_8F, geo2meter_8F, meter2geo_8F = get_gi('8F')
polygons_G, polygon_id_G, beacon_loc_G, geo2meter_G, meter2geo_G = get_gi('G')
polygons_LG, polygon_id_LG, beacon_loc_LG, geo2meter_LG, meter2geo_LG = get_gi('LG')

# gt_cp = list()
# gt_ct = list()
# for floor in ('8F','G','LG'):
#     with open(f'survey/0621/gt-{floor}.txt', 'r') as gt_file:
#         gt = json.loads(gt_file.read())
#     gt_cp.extend(json.loads(gt['event_plan'][0]['geojson']))
#     gt_ct.extend(json.loads(gt['event_plan'][0]['ground_truth']))
#     gt_floor=None #unimplemented
# print(gt_cp)
# print(gt_ct)
# print(len(gt_ct)) # change below 0 from 0 to len(gt_ct)

def get_floor(beacon_batch):
    a = beacon_loc_8F.index; b = beacon_loc_G.index; c = beacon_loc_LG.index
    a1 = beacon_batch.bID.isin(a).sum()
    b1 = beacon_batch.bID.isin(b).sum()
    c1 = beacon_batch.bID.isin(c).sum()
    m = max(a1,b1,c1)
    if m == a1: return '8F', polygons_8F, polygon_id_8F, beacon_loc_8F, geo2meter_8F, meter2geo_8F
    elif m == b1: return 'G', polygons_G, polygon_id_G, beacon_loc_G, geo2meter_G, meter2geo_G
    else: return 'LG', polygons_LG, polygon_id_LG, beacon_loc_LG, geo2meter_LG, meter2geo_LG



data_beacon = pd.read_csv('survey/0726_sample/survey.csv', usecols=[0,1,2])
data_beacon.columns = ('bID', 'rssi', 'ts')
#data_beacon = data_beacon[(data_beacon.ts > gt_interval[0]) & (data_beacon.ts < gt_interval[1])]
data_beacon['ts'] //= PERIOD

initpf = True
for t, beacon_batch in data_beacon.groupby('ts'):
    floor, polygons, polygon_id, beacon_loc, geo2meter, meter2geo = get_floor(beacon_batch) #infer floor by some indicator, currently only number of packets
    if initpf: 
        pf = PF((floor,polygons,beacon_loc))
        initpf = False
    if floor != pf.floor:
        del pf; pf = PF((floor,polygons,beacon_loc))    
    beacon_batch = beacon_batch[beacon_batch.bID.isin(beacon_loc.index)]
    pf.feed_data(t, beacon_batch[['bID','rssi']]) # add condition beacon_batch is not None and nonempty to run online
    if pf.tracked:
        #x, y = meter2geo(pf.pos_estimate)
        x, y = meter2geo(pf.adsorb_polygon())
        if pf.polygon_idx:
            print(f'{int(t)} at ({x:.3f}, {y:.3f})±{pf.pos_var:.3f}m in polygon #{polygon_id[pf.polygon_idx]} on floor {floor};') # polygon_idx==0 means the location must not in any polygon
        else:
            print(f'{int(t)} at ({x:.3f}, {y:.3f})±{pf.pos_var:.3f}m not in any polygon on floor {floor};')
# %%
