from functools import partial
from itertools import pairwise
from math import sqrt
import numpy as np
from PIL import Image, ImageFont, ImageDraw as ID
import matplotlib as mpl
from matplotlib import pyplot as plt#, ticker as mtick
from statsmodels.distributions import ECDF
from datetime import datetime as dt

VERSION = 'v0.1'
PERIOD = 1
IMG_BOUND = 10000 # pixel size of longer side (horizonal)
BORDER_WIDTH = 5 # meters, around the four sides of union of all polygons to visualize
TEXT_HEIGHT = 135 # pixels
txtfnt = ImageFont.truetype('config/arial.ttf', 120) # annotation in the plots; How to use system default instead?
def norm(x):
    return sqrt(x[0]**2 + x[1]**2)
def distance(x, y):
    return sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

colors = [[0.113, 0.737, 0.803],
        [0.737, 0.745, 0.196],
        [0.501, 0.501, 0.501],
        [0.847, 0.474, 0.698],
        [0.549, 0.337, 0.29],
        [0.564, 0.403, 0.674],
        [0.843, 0.149, 0.172],
        [0.152, 0.631, 0.278],
        [0.96, 0.498, 0.137],
        [0.09, 0.466, 0.701]] # color: [r, g, b], and each element is in [0., 1.]
line_styles = [(0, ()), (0, (8, 4)), (0, (2, 2)), (0, (3, 4, 1, 4)), (0, (1, 1))]
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
def _plot_each_cdf(ax:plt.Axes, data:dict, x_type:str, title=None):
    plt.sca(ax)
    if x_type == 'x':
        ax.set_xlim(0, 15)
    else:
        ax.set_xlim(-2, 2)
    for idx, (key, val) in enumerate(data.items()):
        ecdf = ECDF(val)
        x = np.linspace(min(val), max(val), 1000)
        y = ecdf(x)
        # plt.plot(x, y, dashes=[6, 2], label=key, color=color, marker=marker, ms=4, markevery=0.1)
        ax.plot(x, y, label=key, color=colors[idx], ls=line_styles[idx])

    ax.grid(True, axis='both', linestyle='--', alpha=0.5, linewidth=1.0)
    plt.xticks(size="xx-large")
    plt.yticks(size="xx-large")
    ax.set_xlabel(f"Error (m{'' if x_type=='x' else '/s'})", size="xx-large")
    ax.set_ylabel("CDF", size="xx-large")
    if title is not None:
        ax.set_title(title, size='xx-large', weight='bold')
    # plt.tight_layout()
    ax.legend(loc="lower right",fontsize="large")
def plot_cdf(params, save_dir, single_plot=True):
    """
    :param param: data: contains the data to plot: {key: value, ...}
                        key: the class name, and also the legend name of this class;
                        value: [val, ...], and these items are of the same class (i.e. key).
                  x_label:
                  subtitle: (iff plotting multiple axes)
    :param save_dir: the name of the figure to save.
    :return: None
    """
    if single_plot:
        fig, ax = plt.subplots(figsize=(4,3))
        _plot_each_cdf(ax, *params)
    else: # assume <=4 plots
        fig, axs = plt.subplots(2, 2, figsize=(8,6))
        for p, ax in zip(params, axs.reshape(-1)):
            _plot_each_cdf(ax, *p)
        plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir, bbox_inches='tight')
        plt.close(fig)


class Plotter():
    def __init__(self, ground_info):
        self.RES_PATH = 'survey/res/v0.1/'
        self.STATION = 'TKOH'
        polygons, beacon_loc = ground_info
        # meter coordinate -> pixel coordinate
        (E,N) = polygons.max(axis=(1,2)) + BORDER_WIDTH
        (W,S) = polygons.min(axis=(1,2)) - BORDER_WIDTH
        (WIDTH, HEIGHT) = (E-W, N-S)
        self.TRANSPOSED = HEIGHT >= WIDTH
        if self.TRANSPOSED:
            self.IMG_SIZE = (IMG_BOUND, round(IMG_BOUND*WIDTH//HEIGHT)+480) # 480=TEXT_HEIGHT*4-STRIDE*3
            self.SCALE = IMG_BOUND / HEIGHT
            self.point_pos_in_img = lambda pos: ((N-pos[1])*self.SCALE, (pos[0]-W)*self.SCALE)
            self.pos_in_img = lambda pos: np.stack(((N-pos[1])*self.SCALE, (pos[0]-W)*self.SCALE))
        else:
            self.IMG_SIZE = (IMG_BOUND, round(IMG_BOUND*HEIGHT//WIDTH)+480)
            self.SCALE = IMG_BOUND / WIDTH
            self.point_pos_in_img = lambda pos: ((pos[0]-W)*self.SCALE, (N-pos[1])*self.SCALE)
            self.pos_in_img = lambda pos: np.stack(((pos[0]-W)*self.SCALE, (N-pos[1])*self.SCALE))
        self.beacon_pos = beacon_loc.copy(True)
        self.beacon_pos[[0,1]] = self.pos_in_img(beacon_loc).T
        self.polygon_pos = self.pos_in_img(polygons)

    @staticmethod
    def _draw_particle(draw:ID.ImageDraw, p, v, w_ind):
        color = 'grey' if w_ind == 0 else f'#{w_ind:02x}0000'
        draw.ellipse((*(p-6), *(p+6)), color)
        if v[0] == 0 and v[1] == 0: # or just small enough
            return
        color = 'lightgrey' if w_ind == 0 else 'grey'
        tx = p[0]+v[0]; ty = p[1]+v[1] # terminal point (tx, ty)
        draw.line((p[0], p[1], tx, ty), fill=color, width=2)
        draw.line((tx, ty, tx-(v[0]*1.732-v[1])*0.125, ty-(v[0]+v[1]*1.732)*0.125), fill=color, width=2)
        draw.line((tx, ty, tx-(v[0]*1.732+v[1])*0.125, ty+(v[0]-v[1]*1.732)*0.125), fill=color, width=2)

# TODO: Represent rssi with better granularity
    @staticmethod
    def _draw_beacon(draw:ID.ImageDraw, b): # b (beacon_x, beacon_y, rssi)
        if b is None or b[2] == 0:
            draw.ellipse((b[0]-12,b[1]-12,b[0]+12,b[1]+12), 'darkcyan')
        elif b[2] < -85:
             draw.ellipse((b[0]-25,b[1]-25,b[0]+25,b[1]+25), 'darkcyan')
             draw.text((b[0],b[1]), str(round(b[2])), 'white', ImageFont.truetype('config/arial.ttf', 36), anchor='mm')    
        else:
            draw.ellipse((b[0]-25,b[1]-25,b[0]+25,b[1]+25), 'blue')
            draw.text((b[0],b[1]), str(round(b[2])), 'white', ImageFont.truetype('config/arial.ttf', 36), anchor='mm')
            # b_ind = int((b[2]+75)/5) # 5=255/((-24)-(-75))
            # draw.ellipse((b[0]-25,b[1]-25,b[0]+25,b[1]+25), f'#00{b_ind:02x}{255-b_ind:02x}') #be cautious of overflow      

    @staticmethod
    def _draw_polygon(draw:ID.ImageDraw, p):
        draw.polygon(tuple(p), fill='lightcyan', outline='#836FFF')

    def _draw_ground(self, draw:ID.ImageDraw, beacon_batch=None):
        # draw polygons
        np.apply_along_axis(partial(Plotter._draw_polygon, draw), 1, self.polygon_pos.transpose(2,1,0).reshape(-1,8))    
        # draw beacons
        self.beacon_pos['rssi'] = 0
        if beacon_batch is not None:
            self.beacon_pos.update(beacon_batch)
            np.apply_along_axis(partial(Plotter._draw_beacon, draw), 0, self.beacon_pos.values.T)
        else:
            np.apply_along_axis(partial(Plotter._draw_beacon, draw), 0, self.beacon_pos.values.T)
        


    def draw_posterior(self, pf, beacon_batch=None, t:int|None=None): # t=0: initialize
        img = Image.new('RGB', self.IMG_SIZE, "white")
        draw = ID.Draw(img)

        c = pf.pos_estimate
        r = pf.pos_var
        if c is None:
            return
        result_log = f"Range: {r:.3f}m; "

        if t is not None:
            label = f'{self.STATION} {dt.fromtimestamp(t).strftime("%m/%d %H:%M:%S")} {t} {VERSION}'
        else: # not used
            label = f'{self.STATION} Initialize {VERSION}'
        
        # Draw polygons and beacons
        self._draw_ground(draw, beacon_batch)
        # Draw ground truth
        t += PERIOD # plot the gt after prediction phase
        # Draw swarm
        pos = self.pos_in_img(pf.pos)
        v = pf.vel * self.SCALE
        v[1] = -v[1]
        if self.TRANSPOSED: v = np.flipud(v)
        w_ind = (((pf.w/pf.w.max()) ** 3)*255).astype('uint8') # or other indicator
        #rgb = 
        for i in range(w_ind.shape[0]): # or swarm_size
            Plotter._draw_particle(draw, pos[:,i], v[:,i], w_ind[i])
        # Draw estimation
        c = self.point_pos_in_img(c)
        r *= self.SCALE    
        draw.ellipse((c[0]-r,c[1]-r,c[0]+r,c[1]+r), fill=None, outline='#b2ff66', width=10)
        
        e = self.point_pos_in_img(pf.adsorb_polygon())
        draw.ellipse((e[0]-40, e[1]-40, e[0]+40, e[1]+40), fill='yellow', outline='blue', width=8)
        # if pf.displacement is not None:
        #     c_ = self.point_pos_in_img(c-pf.displacement) # last_pos, scaled
        #     c = self.point_pos_in_img(c)
        #     r *= self.SCALE
        #     draw.ellipse((c[0]-r,c[1]-r,c[0]+r,c[1]+r), fill=None, outline='#b2ff66', width=10)
        #     if c_ is not None:
        #         vx = c[0]-c_[0]; vy = c[1]-c_[1]
        #         draw.line((c_[0], c_[1], c[0], c[1]), fill='#b2ff66', width=8)
        #         draw.line((c[0], c[1], c[0]-(vx*1.732-vy)*0.125, c[1]-(vx+vy*1.732)*0.125), fill='#b2ff66', width=8)
        #         draw.line((c[0], c[1], c[0]-(vx*1.732+vy)*0.125, c[1]+(vx-vy*1.732)*0.125), fill='#b2ff66', width=8)
        
        # Draw annotation in bottom-left corner
        draw.text((20, self.IMG_SIZE[1]-TEXT_HEIGHT), label, font=txtfnt, align='left', fill='midnightblue')
        draw.text((20, self.IMG_SIZE[1]-TEXT_HEIGHT*2), result_log, font=txtfnt, align='left', fill='black')
        draw.text((20, self.IMG_SIZE[1]-TEXT_HEIGHT*4), pf.event_log, font=txtfnt, align='left', fill='maroon')
        img.save(f'{self.RES_PATH}{str(t)}.png')


    def _draw_traject(self, draw:ID.ImageDraw, pos, line_color, point_color):
        points = self.pos_in_img(pos)
        for p1, p2 in pairwise(points.T):
            draw.line((*p1,*p2),fill=line_color,width=8)
        for p in points.T:
            draw.ellipse((*(p-16),*(p+16)),point_color)        
    # TODO: align with gt
    def draw_posterior_trace(self, ex_trace, gt_trace):
        img = Image.new('RGB', self.IMG_SIZE, "white")
        draw = ID.Draw(img)
        self._draw_ground(draw)
        self._draw_traject(draw, gt_trace, '#b2ff66', 'green')
        self._draw_traject(draw, ex_trace, 'pink', 'red')               
        img.save(f'{self.RES_PATH[:-1]}_trace.png')
        

    # # assume: no two distances are equal
    # @staticmethod
    # def plot_cdf(x, ax:plt.Axes, title, xlabel, ylabel):
    #     x.sort()
    #     xm = x.sum() / len(x)
    #     y = np.arange(1,len(x)+1) / len(x) * 100
    #     ax.set_ylim(0,100)
    #     ax.set_xlim(0,20)
    #     ax.axvline(x=xm, color='purple')
    #     ax.axvline(x=x[-1], color='lightgrey')
    #     ax.text(xm+0.2, 2, f'{xm:.3f}', color='purple', fontsize=18)
    #     ax.plot(np.append(0, x), np.append(0, y))
    #     ax.set_title(title, fontsize=36)
    #     ax.set_xlabel(xlabel, fontsize=22)
    #     ax.set_ylabel(ylabel, fontsize=22)
    #     ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # def plot_performance(self, res_path, estimates):
    #     estimates[3] += PERIOD
    #     t_gt = (estimates[3] > self.gt_interval[0]) & (estimates[3] < self.gt_interval[1]) # and other conditions
    #     t = estimates[3, t_gt]
    #     ex_path = estimates[:2, t_gt]
    #     #r = estimates[2, t_gt]
    #     gt = np.vectorize(self.gt_interp)(t)
    #     err = np.sqrt((gt[0]-ex_path[0])**2 + (gt[1]-ex_path[1])**2)

    # # may plot other like cdf(r), cdf(err(v)), cdf(var_t(v)) ...
    #     fig, ax = plt.subplots(1, 1, figsize=(15,8))
    #     Plotter.plot_cdf(err, ax, 'Cumulative Distribution of Error', 'Distance (/m)', 'Percentage')
    #     #Plotter.plot_cdf(r, ax2, 'Cumulative Distribution of Range', 'Radius (/m)', 'Percentage')
    #     #fig.subplots_adjust(hspace=0.4)
    #     fig.savefig(res_path)
    #     # plt.close(fig)
    #     # return err_m
