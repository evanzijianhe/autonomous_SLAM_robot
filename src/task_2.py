#!/usr/bin/env python3

import sys
import os
import numpy as np
import math
import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from graphviz import Graph
from copy import copy, deepcopy
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
import pandas as pd
from copy import copy, deepcopy
import time

class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array, extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""

    def __open_map(self, map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        im = Image.open('/home/zijian/catkin_ws/src/final_project/maps/map.pgm')
        size = 200, 200
        im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin, xmax, ymin, ymax]

    def __get_obstacle_map(self, map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()), (self.map_im.size[1], self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0] * 255
        low_thresh = self.map_df.free_thresh[0] * 255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i, j] > up_thresh:
                    img_array[i, j] = 255
                else:
                    img_array[i, j] = 0
        return img_array

class Queue():
    def __init__(self, init_queue=[]):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue) - 1

    def __len__(self):
        numel = len(self.queue)
        return numel

    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if (i == self.start):
                tmpstr += "<"
                flag = True
            if (i == self.end):
                tmpstr += ">"
                flag = True

            if (flag):
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'

        return tmpstr

    def __call__(self):
        return self.queue

    def initialize_queue(self, init_queue=[]):
        self.queue = copy(init_queue)

    def sort(self, key=str.lower):
        self.queue = sorted(self.queue, key=key)

    def push(self, data):
        self.queue.append(data)
        self.end += 1

    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue) - 1
        return p

class Node():
    def __init__(self, name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self, node, w=None):
        if w == None:
            w = [1] * len(node)
        self.children.extend(node)
        self.weight.extend(w)

class Tree():
    def __init__(self, name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        self.g_visual = Graph('G')

    def __call__(self):
        for name, node in self.g.items():
            if (self.root == name):
                self.g_visual.node(name, name, color='red')
            elif (self.end == name):
                self.g_visual.node(name, name, color='blue')
            else:
                self.g_visual.node(name, name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                # print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name, c.name)
                else:
                    self.g_visual.edge(name, c.name, label=str(w))
        return self.g_visual

    def add_node(self, node, start=False, end=False):
        self.g[node.name] = node
        if (start):
            self.root = node.name
        elif (end):
            self.end = node.name

    def set_as_root(self, node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self, node):
        # These are exclusive conditions
        self.root = False
        self.end = True

g = Graph('G')
g.node('a', 'a', style='filled')
g.node('b', 'b')
g.node('c', 'c')
g.node('d', 'd')
g.node('e', 'e')

g.edge('a', 'b', shape='none')
g.edge('a', 'c')
g.edge('c', 'd')
g.edge('c', 'e')

a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('f')

a.add_children([c], [1])
b.add_children([c, e], [1, 1])
c.add_children([b, e, d], [1, 3, 1])
e.add_children([b, c], [1, 3])
d.add_children([c], [1])

tree = Tree('tree')
tree.add_node(a, start=True)
tree.add_node(b)
tree.add_node(c)
tree.add_node(d)
tree.add_node(e, end=True)
tree.add_node(f)

class AStar():
    def __init__(self, in_tree):
        self.in_tree = in_tree
        self.q = Queue()
        self.dist = {name: np.Inf for name, node in in_tree.g.items()}
        self.h = {name: 0 for name, node in in_tree.g.items()}
        self.visited = {name: False for name, node in in_tree.g.items()}

        for name, node in in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            end = tuple(map(int, self.in_tree.end.split(',')))
            self.h[name] = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

        self.via = {name: 0 for name, node in in_tree.g.items()}
        # for __,node in in_tree.g.items():
        # self.q.push(node)

    def __get_f_score(self, node):
        # pass
        # Place code here (remove the pass
        # statement once you start coding)
        return self.dist[node.name] + self.h[node.name]

    def solve(self, sn, en):
        # pass
        # Place code here (remove the pass
        # statement once you start coding)

        self.q.push(sn)
        self.dist[sn.name] = 0
        while len(self.q) > 0:
            node = self.q.pop()
            if node.name == en.name:
                break
            node.children.sort(key=self.__get_f_score)
            for i in range(len(node.children)):

                c = node.children[i]
                w = node.weight[i]
                if self.dist[c.name] == np.Inf:
                    self.q.push(c)
                    self.via[c.name] = node.name
                    self.dist[c.name] = self.dist[node.name] + w

            # print(node.name,self.q.queue)
            # print(self.dist)
        return self.via

    def reconstruct_path(self, sn, en):
        path = []
        node = en.name
        path.append(node)
        dist = self.dist[en.name]
        while True:
            node = self.via[node]
            if node == 0:
                break
            else:
                path.append(node)
        path.reverse()
        if path[0] != sn.name:
            path = []
        return path, dist

class MapProcessor():
    def __init__(self, name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self, map_array, i, j, value, absolute):
        if ((i >= 0) and
                (i < map_array.shape[0]) and
                (j >= 0) and
                (j < map_array.shape[1])):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self, kernel, map_array, i, j, absolute):
        dx = int(kernel.shape[0] // 2)
        dy = int(kernel.shape[1] // 2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array, i, j, kernel[0][0], absolute)
        else:
            for k in range(i - dx, i + dx):
                for l in range(j - dy, j + dy):
                    self.__modify_map_pixel(map_array, k, l, kernel[k - i + dx][l - j + dy], absolute)

    def inflate_map(self, kernel, absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel, self.inf_map_img_array, i, j, absolute)
        r = np.max(self.inf_map_img_array) - np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array)) / r

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Node('%d,%d' % (i, j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i - 1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d' % (i - 1, j)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_up], [1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i + 1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d' % (i + 1, j)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_dw], [1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j - 1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d' % (i, j - 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_lf], [1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j + 1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d' % (i, j + 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_rg], [1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i - 1][j - 1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d' % (i - 1, j - 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_up_lf], [np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i - 1][j + 1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d' % (i - 1, j + 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_up_rg], [np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i + 1][j - 1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d' % (i + 1, j - 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_dw_lf], [np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i + 1][j + 1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d' % (i + 1, j + 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_dw_rg], [np.sqrt(2)])

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
        r = np.max(g) - np.min(g)
        sm = (g - np.min(g)) * 1 / r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size, size))
        return m

    def draw_path(self, path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array

# Import map and read map, map is manually scanned
map_path = '/home/zijian/catkin_ws/src/final_project/maps/map'
mp = MapProcessor(map_path)
kr = mp.rect_kernel(7, 7) # map inflate rate is increased to avoid collision
mp.inflate_map(kr, True)
mp.get_graph_from_map()


# There are some parameters in amcl.launch file have been modified
# "initial_pose_x" = "-13.0"
# "initial_pose_x" = "-12.0"
# "update_min_d" = "0.001"
# "update_min_a" = "0.001"
class Navigation:
    """! Navigation node class.
    This class should server as a template to implement the path planning and 
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name='Navigation'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """

        # ROS related variables
        self.node_name = node_name
        self.rate = 0
        # Path planner/follower related variables
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()

    def init_app(self):
        """! Node intialization.
        @param  None
        @return None.
        """
        # ROS node initilization

        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(10)
        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.__goal_pose_cbk, queue_size=1)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.__ttbot_pose_cbk, queue_size=1)
        # Publishers
        self.path_pub = rospy.Publisher('global_plan', Path, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """

        # TODO: MAKE SURE YOUR POSITION ESTIMATE IS GOOD ENOUGH.
        self.ttbot_pose = data.pose
        cov = data.pose.covariance

    def a_star_path_planner(self, start_pose, end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        # rospy.loginfo('A* planner.\n> start:{},\n> end:{}'.format(start_pose.pose.position, end_pose.pose.position))
        # TODO: IMPLEMENTATION OF THE ASTAR ALGORITHM

        path = Path()
        goal_pos_x = end_pose.pose.position.x   #read end point from the map
        goal_pos_y = end_pose.pose.position.y   #read end point from the map

        if len(path.poses) == 0 and goal_pos_x != 0:    # check if we have generated path
            start_x = start_pose.pose.position.x        # read car's initial x position, in meter
            start_y = start_pose.pose.position.y        # read car's initial y position, in meter
            end_x = end_pose.pose.position.x            # read car's end x position, in meter
            end_y = end_pose.pose.position.y            # read car's end y position, in meter

            # convert car's start and end position from meter in gazebo coordinate
            # to pixel in A* coordinate.
            start_x_pixel = int(np.round(100 + start_x/0.2))
            start_y_pixel = int(np.round(100 - start_y/0.2))
            end_x_pixel = int(np.round(100 + end_x/0.2))
            end_y_pixel = int(np.round(100 - end_y/0.2))
            # rospy.loginfo(start_y_pixel)
            # rospy.loginfo(start_x_pixel)
            # rospy.loginfo(end_y_pixel)
            # rospy.loginfo(end_x_pixel)

            # assemble the start point and end point by following A*'s format
            mp.map_graph.root = ("%d,%d" % (start_y_pixel, start_x_pixel))
            mp.map_graph.end = ("%d,%d" % (end_y_pixel, end_x_pixel))
            # solve for path
            as_maze = AStar(mp.map_graph)
            as_maze.solve(mp.map_graph.g[mp.map_graph.root],
                          mp.map_graph.g[mp.map_graph.end])
            path.poses, dist_as = as_maze.reconstruct_path(mp.map_graph.g[mp.map_graph.root],
                                                           mp.map_graph.g[mp.map_graph.end])
        # if we do not select an end point in the map, path is empty
        elif len(path.poses) != 0 and start_pose.pose.position == end_pose.pose.position:
            path.poses = []
        elif len(path.poses) != 0 and start_pose.pose.position != end_pose.pose.position:
            path.poses = path.poses
        self.path_pub.publish(path)
        return path

    def get_path_idx(self, idx, path, vehicle_pose):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  current_goal_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position int the path pointing to the next goal pose to follow.
        """
        # determine the desired location based on current idx number
        desire_position = tuple(map(int, path.poses[idx].split(',')))
        # rospy.loginfo(desire_position)

        # measure the distance between car's current position and path waypoint
        distance = math.dist([vehicle_pose.pose.position.x, vehicle_pose.pose.position.y],
                             [(desire_position[1]*0.2-100*0.2), 100*0.2-desire_position[0]*0.2])
        if distance <= 0.05:    # check if robot is close enough to the desired point
            idx += 1            # if it is, move to next point in the path
        else:                   # if it is not, keep moving
            idx = idx

        # TODO: IMPLEMENT A MECHANISM TO DECIDE WHICH POINT IN THE PATH TO FOLLOW idx<=len(path)

        return idx

    def path_follower(self, vehicle_pose, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
        next_position = tuple(map(int, current_goal_pose.split(',')))   #next position determined by index module, in pixel
        # convert from pixel back to meter
        next_pos_x = (next_position[1] * 0.2) - 100*0.2
        next_pos_y = 100*0.2 - next_position[0] * 0.2

        # calculate the heading angle between current car's location and desired location, range from -pi to pi
        next_heading = np.arctan2((next_pos_y - vehicle_pose.pose.position.y),
                                  (next_pos_x - vehicle_pose.pose.position.x))
        # rospy.loginfo('next_heading {}'.format(next_heading))

        # calculate the heading angle of the car, range from -pi to pi
        current_heading = tf.transformations.euler_from_quaternion([vehicle_pose.pose.orientation.x,
                                                                    vehicle_pose.pose.orientation.y,
                                                                    vehicle_pose.pose.orientation.z,
                                                                    vehicle_pose.pose.orientation.w])[2]
        # rospy.loginfo('current_heading {}'.format(current_heading))
        # determine the sign of the two heading angles
        sign_current_heading = np.sign(current_heading)
        sign_next_heading = np.sign(next_heading)

        # The following if-elseif structure is used to decide the rotation angle of the car. The basic idea is
        # to make sure the robot is straightly facing to the next point of the path, then move forward to that
        # point.
        # The heading angle is from -pi to pi.
        # The heading angle of the inital point of the robot is 0.
        # CW is from 0 to pi
        # CCW if from 0 to -pi
        if sign_current_heading != sign_next_heading and abs(current_heading-next_heading) > math.pi:
            direction = sign_current_heading
        elif sign_current_heading != sign_next_heading and abs(current_heading-next_heading) < math.pi:
            direction = sign_next_heading
        elif sign_current_heading == sign_next_heading and current_heading > next_heading:
            direction = -1
        elif sign_current_heading == sign_next_heading and current_heading < next_heading:
            direction = 1
        else:
            direction = 0

        # calculate angle difference
        heading_e = abs(current_heading - next_heading)
        speed = 0.0
        heading = 0.0
        # robot's moving and twisting
        if heading_e > 2:
            heading = direction * 0.8
            speed = 0.0
        elif 1 < heading_e < 2:
            heading = direction * 0.6
            speed = 0.0
        elif 0.7 < heading_e < 1:
            heading = direction * 0.5
            speed = 0.08
        elif 0.3 < heading_e < 0.7:
            heading = direction * 0.4
            speed = 0.1
        elif 0.1 < heading_e < 0.3:
            heading = direction * 0.3
            speed = 0.2
        elif heading_e <= 0.1:
            heading = 0
            speed = 0.4

        # TODO: IMPLEMENT PATH FOLLOWER

        return speed, heading, next_heading, current_heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired yaw angle.
        @param  heading   Desired speed.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()
        # TODO: IMPLEMENT YOUR CONTROLLER LOW LEVEL CONTROLLER
        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading

        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """

        '''
            Main loop
        '''
        path_complete = False
        timeout = False
        idx = 0
        while not rospy.is_shutdown():
            # 1. Create the path to follow
            path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            # 2. Loop through the path and move the robot
            # if len(path.poses) != 0 and not path_complete:
            if len(path.poses) != 0 and not path_complete:
                while idx < len(path.poses)-1:
                    idx = self.get_path_idx(idx, path, self.ttbot_pose)
                    current_goal = path.poses[idx]
                    speed, heading, next_heading, current_heading = self.path_follower(self.ttbot_pose, current_goal)
                    self.move_ttbot(speed, heading)
                # self.move_ttbot(speed, heading)
                # here we check TTbot's heading angle with the goal heading orientation
                goal_heading = self.goal_pose.pose.orientation.z
                ttbot_heading = self.ttbot_pose.pose.orientation.z
                # rospy.loginfo(global_heading)
                # rospy.loginfo(ttbot_heading)
                goal_heading_sign = np.sign(goal_heading)
                ttbot_heading_sign = np.sign(ttbot_heading)
                heading_diff = abs(goal_heading - ttbot_heading)
                heading = 0
                speed = 0
                self.move_ttbot(speed, heading)
                path.poses = []     # retrieve new goal position
                idx = 0
                # adjust its heading angle difference
                while heading_diff > 0.01:
                    heading = 0.2 * -1
                    goal_heading = self.goal_pose.pose.orientation.z
                    ttbot_heading = self.ttbot_pose.pose.orientation.z
                    heading_diff = abs(goal_heading - ttbot_heading)
                    rospy.loginfo(goal_heading)
                    rospy.loginfo(ttbot_heading)
                    rospy.loginfo(heading_diff)
                    self.move_ttbot(speed, heading)
                heading = 0
                speed = 0
                self.move_ttbot(speed, heading)
                # path_complete = True
                    # ----------------------------------------
                    # TODO: YOU NEED TO ADD YOUR SOLUTION HERE
                    # ----------------------------------------
                    # This code can allow the car to move from start point to
                    # almost every point on the map, except to the points closed to the wall
                    # or boundaries. Thus, please choose the point at the middle of the
                    # hallway.
            else:
                speed = 0       # stop the robot
                heading = 0
                self.move_ttbot(speed, heading)
                # rospy.loginfo('Path is completed')
                # path_complete = True
            self.rate.sleep()
        rospy.signal_shutdown("[{}] Finished Cleanly".format(self.name))

if __name__ == "__main__":
    nav = Navigation(node_name='Navigation')
    nav.init_app()
    try:
        nav.run()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)
