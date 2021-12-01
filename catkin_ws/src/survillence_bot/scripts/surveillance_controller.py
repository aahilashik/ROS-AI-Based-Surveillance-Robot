#!/usr/bin/env python

import numpy as np
import actionlib
import pickle
import rospy
import time
import cv2
import os

from sensor_msgs.msg import Image
from actionlib_msgs.msg import GoalStatus
from object_detection import detectObjects
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

rospy.loginfo("Hello ROS!")

# Initialize the CvBridge class
bridge = CvBridge()

# Define a function to show the image in an OpenCV Window
def showImage(img):
	cv2.imshow("Surveillance", img)
	cv2.waitKey(3)

scale = 0.6
stopFlag = False
# Define a callback for the Image message
def imagesCallback(img_msg):
	global client, stopFlag, image, subImage
	start = time.time()

	try: image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
	except CvBridgeError as e: print(e)

	image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
	
	try: 
		data, image = detectObjects(image, ["person"])
		if "person" in data.keys():
			client.cancel_goal()
			x, y, w, h = data['person']
			image = cv2.putText(image, "Intruder", (int(x+w/2), int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, [20, 50, 250], 3)
			print("#"*50)
			print("#"*50)
			print("#"*17 + " Intruder Alert " + "#"*17)	
			print("#"*50)
			print("#"*50)
			stopFlag = True
			cv2.imshow("Surveillance", image)
			subImage.unregister()
			cv2.waitKey(-1)
	except: pass; print("Skipped")
	
	stop = time.time()

	print("FPS : " + str(1.0/(stop - start)))


	showImage(image)



waypoints = []
cachePath = "./cache.p"
markerArray = MarkerArray()
def pointsCallback(data):
	global waypoints, cachePath, markerArray, pointsMarker
	waypoints.append([data.point.x, data.point.y])

	marker = Marker()
	marker.id = len(waypoints)
	marker.header.frame_id = "map"
	marker.type = marker.SPHERE
	marker.action = marker.ADD
	marker.scale.x = 0.2
	marker.scale.y = 0.2
	marker.scale.z = 0.4
	marker.color.a = 1.0
	marker.color.r = 1.0
	marker.color.g = 1.0
	marker.color.b = 0.0
	marker.pose.orientation.w = 1.0
	marker.pose.position.x = data.point.x
	marker.pose.position.y = data.point.y 
	marker.pose.position.z = data.point.z

	markerArray.markers.append(marker)
	with open(cachePath, "wb") as file:
		pickle.dump([waypoints, markerArray], file)
	
	pointsMarker.publish(markerArray)
	
	print("Received - X : {} | Y : {}".format(data.point.x, data.point.y))


def main():
	global client, waypoints, cachePath, markerArray, pointsMarker, stopFlag, image, subImage

	rospy.init_node('surveillance_controller', anonymous=True)

	client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
	client.wait_for_server()

	publishPoint = rospy.Subscriber("/clicked_point", PointStamped, pointsCallback)
	pointsMarker = rospy.Publisher("/markers", MarkerArray, queue_size=100)
	subImage = rospy.Subscriber("/camera/rgb/image_raw", Image, imagesCallback)

	if os.path.exists(cachePath): 
		if "y" in raw_input("Found Cache Data!!\nDo you want to use it? (Y/N): "): 
			with open(cachePath, "rb") as file:
				waypoints, markerArray = pickle.load(file)
				pointsMarker.publish(markerArray)

	idx, UPDN = 0, 0
	while not rospy.is_shutdown():
		if stopFlag: 
			print("Intruder Detected!\nProgram Stopped\nPress Ctrl + C to Close")
						
			break

		if len(waypoints) <= 1: continue
		if idx >= len(waypoints)-1 and UPDN == 0: 	UPDN = 1
		if idx <= 0 and UPDN == 1: 			UPDN = 0

		x, y = waypoints[idx]
		goal = MoveBaseGoal()
		goal.target_pose.header.frame_id 	= "map"
		goal.target_pose.header.stamp 		= rospy.Time.now()
		goal.target_pose.pose.position.x 	= x
		goal.target_pose.pose.position.y 	= y
		goal.target_pose.pose.orientation.w 	= 1.0

		client.send_goal(goal)

		if client.wait_for_result():
			if client.get_result() == GoalStatus.SUCCEEDED: print("Reached - X : {} | Y : {}".format(x, y))
		else:
			print("Action server not available!")
			

		if UPDN: 	idx = idx - 1
		else: 		idx = idx + 1
	rospy.spin()

if __name__ == "__main__":
	main()

