#!/usr/bin/env python3
import rospy
import rospkg
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose


def spawn_model(model_name):
    initial_pose = Pose()
    initial_pose.position.x = 0
    initial_pose.position.y = 0
    initial_pose.position.z = 0.25

    # Spawn the new model #
    model_path = rospkg.RosPack().get_path('final_project')+'/models/'
    model_xml = ''

    with open (model_path + "obstacle" + '/model.sdf', 'r') as xml_file:
        model_xml = xml_file.read().replace('\n', '')

    spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
    spawn_model_prox(model_name, model_xml, '', initial_pose, 'world')

def position_node():
    # Create a publisher object with Twist
    pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
    # Declare the node, and register it with a unique name
    rospy.init_node('model_service_node', anonymous=True)
    # Define the execution rate object (10Hz)
    rate = rospy.Rate(100)
    spawn_model('obstacle_0')
    spawn_model('obstacle_1')
    spawn_model('obstacle_2')
    '''
        This is the main node loop
    '''
    state_msg_0 = ModelState()
    state_msg_0.model_name = 'obstacle_0'
    state_msg_1 = ModelState()
    state_msg_1.model_name = 'obstacle_1'

    state_msg_2 = ModelState()
    state_msg_2.model_name = 'obstacle_2'

    t = 1000
    while not rospy.is_shutdown():
        for i in range(t):
            state_msg_0.pose.position.y = -4+6.0*i/t
            state_msg_1.pose.position.y = 14-6.0*i/t
            state_msg_1.pose.position.x = -12+6.0*i/t
            state_msg_2.pose.position.x = 6 + 6.0 * i / t
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                resp = set_state(state_msg_0)
                resp = set_state(state_msg_1)
                resp = set_state(state_msg_2)
            except rospy.ServiceException as e:
                print("Service call failed: ", e)
            rate.sleep()
        for i in range(t):
            state_msg_0.pose.position.y = 2-6.0*i/t

            state_msg_1.pose.position.y = 8+6.0*i/t
            state_msg_1.pose.position.x = -6-6.0*i/t
            state_msg_2.pose.position.x = 12 - 6.0 * i / t
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                resp = set_state( state_msg_0 )
                resp = set_state(state_msg_1)
                resp = set_state(state_msg_2)
            except rospy.ServiceException as e:
                print("Service call failed: ", e)
            rate.sleep()

if __name__ == '__main__':
    try:
        position_node()
    except rospy.ROSInterruptException:
        pass

