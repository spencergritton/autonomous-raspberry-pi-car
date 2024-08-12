from direction import Direction
from drive_instruction import DriveInstruction
from node import Node
from Motor import *
from Ultrasonic import *
from servo import *
import threading
import sys
import utils
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import queue
from time import sleep, time
from io import StringIO
from bisect import insort
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

class Car:
	'''
	Represents the vehicle, its map of the environment, position, target, and a variety of other things.
	In a refactor this class would be broken up into multiple classes.

	Attributes:
	cmScale (int): How many centimeters should be represented by each (x,y) position in the environment map (envMap).
	envMapBuffer (int): Number of tiles of padding to encase the envMap in to allow the vehicle more navigation room.
	motor (Motor): Freenove class allowing the vehicle to drive.
	servo (Servo): Freenove class allowing the camera/ultrasonic sensor to move left and right.
	ultrasonic (Ultrasonic): Freenove class allowing the capture of ultrasonic distances from the ultrasonic sensor.
	target (Node): Target node to navigate to from position node.
	position (Node): Vehicles current position.
	direction (Direction): Which way the vehicle is facing in real life in relation to its envMap.
	envMapSizeCm (int): How large to make the envMap is in the x and y directions (i.e 4 -> 4x4 map).
	objectDetectionThread (Thread): Thread to run TensorFlow object detection inside.
	exitObjectDetectionEvent (Event): Event that occurs when object detection should stop.
	personDetectedEvent (Event): Event that occurs when object detection recognizes a person.
	stopSignDetectedEvent (Event): Event that occurs when object detection recognizes a stop sign.
	threadMessageQueue (Queue): Queue of messages sent to the main Car thread from the Object detection thread.
	'''
	PERSON_DETECTION_CATEGORY = 'person'
	STOP_SIGN_DETECTION_CATEGORY = 'stop sign'
	
	def __init__(self, targetXCm, targetYCm):
		'''
		Initialize a new Car instance with a target given in centimeters away from the vehicle in both x and y distances.
		Assumes car will always be pointing in 90 degree intervals. Target to the left of car is -x, target behind car is -y.

		Parameters:
		targetXCm (int): Number of cm from the car to target - Negative means left, positive means right
		targetYCm (int): Number of cm from the car to the target - Negative means behind the vehicle, positive means in front 
		'''
		self.cmScale = 6
		self.envMapBuffer = 5 # have buffer area around each side of envMap so target and position is not on edge of x or y
		self.motor = Motor()
		self.servo = Servo()
		self.ultrasonic = Ultrasonic()
		self.target = Node(targetXCm // self.cmScale, targetYCm // self.cmScale)
		self.position = Node(0, 0, None)
		self.direction = Direction.UP
		self.envMapSizeCm = max(abs(self.target.x), abs(self.target.y)) + self.envMapBuffer * 2
		
		# Object detection
		self.objectDetectionThread = None
		self.exitObjectDetectionEvent = threading.Event()
		self.personDetectedEvent = threading.Event()
		self.stopSignDetectedEvent = threading.Event()
		self.threadMessageQueue = queue.Queue()

		# Convert initial position to accommodate buffer and negative values
		if self.target.x >= 0:
			self.target.x += self.envMapBuffer
			self.position.x += self.envMapBuffer
			
		if self.target.y >= 0:
			self.target.y += self.envMapBuffer
			self.position.y += self.envMapBuffer

		# Deal with possibility of target being in negative coords
		if self.target.x < 0:
			self.position.x = abs(self.target.x) + self.envMapBuffer
			self.target.x = self.envMapBuffer
			
		if self.target.y < 0:
			self.position.y = abs(self.target.y) + self.envMapBuffer
			self.target.y = self.envMapBuffer
					
		# Set ultrasonic sensor angle to expectation
		self.ultrasonic_angle = 90
		
		# Create environment map
		self.resetMap()
		
		# Move servos to meet expectations
		self.servo.setServoPwm('0', self.ultrasonic_angle)
		
	def resetMap(self):
		'''
		Resets internal representation of the environment for the car to be blank. Does not erase cars knowledge of its position.
		Used to remove past assumptions of the environment and allow the Car to scan again to build a map of obstacles.
		'''
		self.envMap = np.zeros((self.envMapSizeCm, self.envMapSizeCm))
		self.lastScan = (self.position.x, self.position.y, -1) # records last scan x, y, and object id (-1 car, 0 nothing, 1 obstacle)
		self.position.envMap = self.envMap
		self.target.envMap = self.envMap
		self.position.updateVal()
		self.target.updateVal()
		
	def showMap(self, envMap=None):
		'''
		Displays the current or given envMap to the user via a 2D graph in matplotlib.

		Parameters:
		envMap (2D NP Array): Array indicating car, obstacle, and target positions.
		'''
		mapToShow = np.copy(self.envMap) if envMap is None else envMap
		mapToShow[self.position.y, self.position.x] = -1
		mapToShow[self.target.y, self.target.x] = 2
		plt.imshow(np.flipud(mapToShow), cmap='viridis', interpolation='nearest')
		plt.colorbar()
		plt.axis('off')
		plt.show()
		
	def updateMapInterpolateBetweenLastScanAndCurrent(self, x2, y2, val):
		'''
		Interpolates between lastScan and the current scan given with x2 and y2 to fill in the envMap with obstacles
		between the points where the car has sampled data from the environment.

		If val != lastScans val, don't interpolate.

		Parameters:
		x2 (int): X value of current scan in the envMap.
		y2 (int): Y value of current scan in the envMap.
		val (int): Value of current scan in the envMap (obstacle - 1, air - 0, etc.)
		'''
		if self.lastScan[2] != 1 or val != 1:
			return
		
		x1 = self.lastScan[0]
		y1 = self.lastScan[1]
		
		# Skip interpolation if distance between last and current scan is too large
		if math.sqrt((x1 - x2)**2 + (y1 - y2)**2) > 3:
			return
			
		dx = abs(x2 - x1)
		dy = abs(y2 - y1)
		sx = 1 if x1 < x2 else -1
		sy = 1 if y1 < y2 else -1
		err = dx - dy

		while x1 != x2 or y1 != y2:
			# Set the value at the current coordinates
			self.envMap[y1, x1] = val

			e2 = 2 * err
			if e2 > -dy:
				err -= dy
				x1 += sx
			if e2 < dx:
				err += dx
				y1 += sy
		
	def updateMapAtCoords(self, x, y, val):
		'''
		Sets the envMap at given coordinates to a given value.
		Also attempts to interpolate from this scan to the last scan to fill in data between.

		Parameters:
		x (int): X value in envMap
		y (int): Y value in envMap
		val (int): Value to insert into envMap at (x,y)
		'''
		# Update map value
		self.envMap[y, x] = val
		
		# Update interpolations between current value and last scan
		self.updateMapInterpolateBetweenLastScanAndCurrent(x, y, val)
			
		# Update last scan to this scan
		self.lastScan = (x, y, val)
		
	def setSensorAngle(self, angle):
		'''
		Sets the angle of the ultrasonic sensor. 90 is perpendicular to the front of the Car.

		Parameters (float): Angle to turn the ultrasonic sensor. Min 60, Max 120.
		'''
		self.ultrasonic_angle = angle
		self.servo.setServoPwm('0', angle)
		
	def getUltrasonicDistance(self):
		'''
		Retrieves an ultrasonic distance from the sensor

		Returns:
		float: Distance in cm from the nearest obstacle to the car at the angle of the ultrasonic sensor. 0 if no obstacle. 
		'''
		return ultrasonic.get_distance()
		
	def updateEnvMap(self, angle, distance):
		'''
		Takes an angle and distance reading from the car and updates envMap to potentially add an obstacle to the map.

		Parameters:
		angle (float): Angle of ultrasonic sensor (90 is perpendicular to Car).
		distance (float): Cm from vehicle to nearest obstacle at the given angle. 0 if no obstacle.
		'''
		if distance == 0:
			return
		
		# Convert angle to be zero based
		angle_zero = angle - 90
		
		# Convert angle to be plus or minus given our direction
		if self.direction == Direction.RIGHT:
			angle_zero += 90
		elif self.direction == Direction.LEFT:
			angle_zero -= 90
		if self.direction == Direction.DOWN:
			angle_zero += 180
		
		angle_radians = math.radians(angle_zero)
		
		# Find x and y
		x = int(self.position.x + distance // self.cmScale * math.sin(angle_radians))
		y = int(self.position.y + distance // self.cmScale * math.cos(angle_radians))
		
		print(f'pos-x: {self.position.x}, pos-y: {self.position.y}, Distance {distance}, Angle: {angle}, x: {x}, y: {y}')
		
		# if object too far away return and set last scan to nothing
		if x >= self.envMapSizeCm or x <= 0 or y >= self.envMapSizeCm or y <= 0:
			self.lastScan = (x, y, 0)
			return
			
		self.updateMapAtCoords(x, y, 1)
		
	def scanEnvironment(self):
		'''
		Erases past knowledge of obstacles in the environment and re-scans with the ultrasonic sensor to find obstacles.
		'''
		self.lastScanFoundObjectNumber = 0
		self.resetMap()
		firstScan = True
		
		for angle in range(60, 122, 2):
			self.setSensorAngle(angle)
			sleep(.05) # sleep unfortunately needed due to freenove libs taking a bit to move servos and providing no waiting mechanism
			if firstScan:
				sleep(.5)
				firstScan = False
				
			distance = self.getUltrasonicDistance()
			self.updateEnvMap(angle, distance)
			
		sleep(.05)
		self.setSensorAngle(90)
		
	def a(self):
		'''
		Finds route from Car position to target using a* algorithm and envMap to determine where obstacles are.

		Returns:
		Node: target node which has path back to start, if a path from position to target was found.
		None: If no path was found from position to target.
		'''
		open = [self.position]
		closed = []

		while open:
			current = open.pop()
			closed.append(current)

			# print(f'Curr: {current}\nTar: {self.target}\n')
			if current == self.target:
				return current
			
			u = Node(current.x, current.y + 1, self.envMap, parent=current)
			l = Node(current.x - 1, current.y, self.envMap, parent=current)
			r = Node(current.x + 1, current.y, self.envMap, parent=current)
			d = Node(current.x, current.y - 1, self.envMap, parent=current)
			neighbors = [u, d, l, r]

			for node in neighbors: node.updatePenalty(self.target)
			neighbors = list(filter(lambda x: x.isInbounds() and not x.isObstacle() and x not in closed, [u, l, r, d]))

			for neighbor in neighbors:
				neighborInOpenIndex = None
				try:
					neighborInOpenIndex = open.index(neighbor)
				except Exception:
					pass
				
				if neighborInOpenIndex and open[neighborInOpenIndex].f > neighbor.f:
					open[neighborInOpenIndex] = neighbor
				if not neighborInOpenIndex:
					insort(open, neighbor)
					
		return None
		
	def getRouteAsDirectionsList(self, steps):
		'''
		Gets steps number of DriveInstructions from a* algorithm.
		Used for generating a series of instructions to drive the car towards the target.

		Parameters:
		steps (int): Number of steps to take towards the target along the path discovered to target.

		Returns:
		List[DriveInstruction]: List of instructions telling the car where to drive.

		Raises:
		Exception: If no route is findable from Car position to the target.
		'''
		node = self.a() # Gets a* route starting at target going towards position
		if node is None:
			raise Exception('No valid route from position to target found!')
		
		# Navigate from end of route to start setting node children on the way
		while True:
			if node == self.target:
				node.parent.child = node
				node = node.parent
			elif node == self.position:
				break
			else:
				node.parent.child = node
				node = node.parent
				
		# Have the starting node, follow it x steps and get drive instructions
		driveInstructions = []
		step = 0
		while node.child is not None and step < steps:
			# Get direction of current node to its child
			direction = None
			child = node.child
			
			if node.x == child.x - 1:
				direction = Direction.RIGHT
			elif node.x == child.x + 1:
				direction = Direction.LEFT
			elif node.y == child.y - 1:
				direction = Direction.UP
			else:
				direction = Direction.DOWN
			
			# Add to previous direction if its in the same direction as this one
			previousInstruction = driveInstructions[-1] if len(driveInstructions) > 0 else None
			
			if previousInstruction is not None and previousInstruction.direction == direction: # make instructions do more than one unit at a time
				previousInstruction.incrementDistance()
			else:
				driveInstructions.append(DriveInstruction(1, direction))
			
			node = node.child
			step +=1
			
		return driveInstructions
				
	def drawRoute(self):
		'''
		Draw the cars proposed a* route on the map and display it to the user with matplotlib.	
		'''
		envMap = np.copy(self.envMap)
		node = self.a()
		
		if node is None:
			raise Exception('Unable to find a route to destination!')
		
		while True:
			if node == self.target:
				node = node.parent
			elif node == self.position:
				tempEnvMap = self.envMap
				self.envMap = envMap
				self.showMap(envMap)
				self.envMap = tempEnvMap
				return
			else:
				envMap[node.y, node.x] = 5
				node = node.parent
			
	def runObjectDetection(self, num_threads=4, model='efficientdet_lite0.tflite', width=640, height=480):
		'''
		Start detecting objects like stop signs and people using the camera and TensorFlow.
		Meant to run in a separate thread and alert the main thread when objects are detected.

		Continuously attempts to detect people and stop signs, when they are detected send relevant events
		to the main thread to take action on these detections.

		Parameters:
		num_threads (int): Number of threads to use for the object detection model.
		model (string): Name of TensorFlow model to use for object detection (should be stored locally).
		width (int): Width of video feed from camera.
		height (int): Height of video feed from camera.
		'''	
		# Reroute output (due to raspberry pi dependency issues)
		old_stderr = sys.stderr
		sys.stderr = StringIO()
		
		# Variables to calculate FPS
		counter, fps = 0, 0
		start_time = time()

		# Start capturing video input from the camera
		connstr = 'libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! videoscale ! appsink'
		cap = cv2.VideoCapture(connstr, cv2.CAP_GSTREAMER)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

		# Visualization parameters
		row_size = 20  # pixels
		left_margin = 24  # pixels
		text_color = (0, 0, 255)  # red
		font_size = 1
		font_thickness = 1
		fps_avg_frame_count = 10

		# Initialize the object detection model
		base_options = core.BaseOptions(
		file_name=model, use_coral=False, num_threads=num_threads)
		detection_options = processor.DetectionOptions(
		max_results=2, score_threshold=0.3)
		options = vision.ObjectDetectorOptions(
		base_options=base_options, detection_options=detection_options)
		detector = vision.ObjectDetector.create_from_options(options)

		# Continuously capture images from the camera and run inference
		while cap.isOpened() and not self.exitObjectDetectionEvent.is_set():
			success, image = cap.read()
			if not success:
				sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

			counter += 1
			image = cv2.flip(image, 1)

			# Convert the image from BGR to RGB as required by the TFLite model.
			rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			# Create a TensorImage object from the RGB image.
			input_tensor = vision.TensorImage.create_from_array(rgb_image)

			# Run object detection estimation using the model.
			detection_result = detector.detect(input_tensor)
			
			# Send messages if we detect a categories of objects we are interested in
			if len(detection_result.detections) > 0:
				categories = set(map(lambda detection: detection.categories[0].category_name, detection_result.detections))
				
				if Car.PERSON_DETECTION_CATEGORY in categories:
					self.personDetectedEvent.set()
				else:
					self.personDetectedEvent.clear()
					
				if Car.STOP_SIGN_DETECTION_CATEGORY in categories:
					self.stopSignDetectedEvent.set()
				else:
					self.stopSignDetectedEvent.clear()
				
			# Draw keypoints and edges on input image
			image = utils.visualize(image, detection_result)

			# Calculate the FPS
			if counter % fps_avg_frame_count == 0:
				end_time = time()
				fps = fps_avg_frame_count / (end_time - start_time)
				start_time = time()

			# Show the FPS
			fps_text = 'FPS = {:.1f}'.format(fps)
			text_location = (left_margin, row_size)
			cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
			font_size, text_color, font_thickness)
			
			self.threadMessageQueue.put(f'FPS: {fps}')
			
			#cv2.imshow('object_detector', image)

		cap.release()
		cv2.destroyAllWindows()
		

	def runQueueMessageReader(self, detectionQueue):
		'''
		Prints messages in the detection queue out to console.

		Not used in final project, but was used for debugging threading issues.

		Parameters:
		detectionQueue (Queue): Queue of threaded messages to read.
		'''
		while not self.exitObjectDetectionEvent.is_set():
			print(detectionQueue.get(block=False))
		
	def startObjectDetection(self):
		'''
		Starts object detection thread and begins attempting to detect objects on that thread.
		'''
		print('Starting object detection...')
		if self.objectDetectionThread is None or not self.objectDetectionThread.is_alive():
			self.objectDetectionThread = threading.Thread(target=self.runObjectDetection)
			self.objectDetectionThread.start()
			sleep(1)
			
			#queueReaderThread = threading.Thread(target=self.runQueueMessage, args=(self.threadMessageQueue))
			#queueReaderThread.start()
			# print('Started queue!')
			
	def stopObjectDetection(self):
		'''
		Stops the object detection thread.
		'''
		self.exitObjectDetectionEvent.set()
		self.objectDetectionThread.join()
		self.objectDetectionThread = None
		
	def turnLeft90Degrees(self):
		'''
		Turns the vehicle left by 90 degrees in real space.
		Motor values found through experimentation (slightly inaccurate).
		'''
		self.motor.setMotorModel(-4000, -4000, 4000, 4000)
		sleep(.4) # not ideal but sleep necessary due to provided libraries from Freenove for motor movement
		self.motor.setMotorModel(0, 0, 0, 0)
		sleep(.1)
		
	def turnRight90Degrees(self):
		'''
		Turns the vehicle right by 90 degrees in real space.
		Motor values found through experimentation (slightly inaccurate).
		'''
		self.motor.setMotorModel(4000, 4000, -4000, -4000)
		sleep(.475)
		self.motor.setMotorModel(0, 0, 0, 0)
		sleep(.1)
		
	def driveForward(self, direction):
		'''
		Drives the vehicle forward in the direction it is facing by the number of units specified by the given direction.

		Example:
		If the vehicle faces left, it will drive left by direction.distance units in the envMap.
		It will drive forward by direction.distance * cmScale in real life.

		Parameters:
		direction (DriveInstruction): Instruction with distance telling the vehicle how far to drive in the
			direction it is facing.
		'''
		cm = direction.distance * self.cmScale
		self.motor.setMotorModel(1600, 1400, 1000, 1000)
		
		# Since car has no sensors for speed or distance, made up a weighing function to attempt to map motor movement to distance
		# This is not the most accurate method obviously but all I had time for.
		sleepModifierBig = 0.026 # for 30 cm or higher. Found via testing motors.
		sleepModifierSmall = 0.035 # for 6 cm or lower. Found via testing motors.
		
		sleepTime = 0
		if cm >= 26: sleepTime = cm*sleepModifierBig
		elif cm <= 6: sleepTime = cm*sleepModifierSmall
		else: sleepTime = cm*(cm/26 * sleepModifierBig + (1-cm/26) * sleepModifierSmall)
		
		# Attempt to move for sleepTime, but if object detected react
		startMovementTime = time()
		while time() - startMovementTime < sleepTime:
			if self.personDetectedEvent.is_set():
				self.motor.setMotorModel(0, 0, 0, 0)
				sleep(2)
				startMovementTime = time() - (time() - startMovementTime())
				
		
		self.updatePositionGivenDirection(direction)
		self.motor.setMotorModel(0, 0, 0, 0)
		sleep(.1)
		
	def driveBackward(self, direction):
		'''
		Drives the vehicle backward in the opposite of the direction it is facing by the number of units
		specified by the given direction.

		Parameters:
		direction (DriveInstruction): Instruction with distance telling the vehicle how far to drive opposite
			the direction it is facing.
		'''
		cm = direction.distance * self.cmScale
		self.motor.setMotorModel(-1600, -1400, -1000, -1000)
		
		sleepModifierBig = 0.026 # for 30 cm or higher. Found via testing motors.
		sleepModifierSmall = 0.035 # for 6 cm or lower. Found via testing motors.
		
		if cm >= 26: sleep(cm*sleepModifierBig)
		elif cm <= 6: sleep(cm*sleepModifierSmall)
		else: sleep(cm*(cm/26 * sleepModifierBig + (1-cm/26) * sleepModifierSmall))
		
		self.updatePositionGivenDirection(direction)
		self.motor.setMotorModel(0, 0, 0, 0)
		sleep(.1)
		
	def updatePositionGivenDirection(self, direction):
		'''
		Updates cars internal representation of its position given a direction.

		Parameters:
		direction (DriveInstruction): Instruction specifying where a car should move relative to its current position.
		'''
		if direction.direction == Direction.UP:
			self.position.y += direction.distance
			
		elif direction.direction == Direction.DOWN:
			self.position.y -= direction.distance
			
		elif direction.direction == Direction.RIGHT:
			self.position.x += direction.distance
			
		elif direction.direction == Direction.LEFT:
			self.position.x -= direction.distance
		
	def drive(self, directions):
		'''
		Takes a list of DriveInstruction's and drives following them.

		Parameters:
		directions (List[DriveInstruction]): List of directions to drive relative to the cars current position.
			Ordered from 0->N, where 0 is the first instruction to drive.
		'''
		for direction in directions:
			print(f'Driving from: {self.position}.\n Currently pointing at {self.direction}.\n Towards: {direction}\n\n')
			
			# Person is detected elsewhere as it is more important to stop immediately for
			if self.stopSignDetectedEvent.is_set():
				print('Detected a stop sign!')
				sleep(2)
				self.stopSignDetectedEvent.clear()
			
			# Direction matches ours
			if self.direction == direction.direction:
				self.driveForward(direction)
			# Direction is opposite of ours
			elif (self.direction == Direction.DOWN and direction.direction == Direction.UP) or (self.direction == Direction.UP and direction.direction == Direction.DOWN) or (self.direction == Direction.RIGHT and direction.direction == Direction.LEFT) or (self.direction == Direction.LEFT and direction.direction == Direction.RIGHT):
				self.driveBackward(direction)
			# Direction is to the right of ours
			elif (self.direction == Direction.UP and direction.direction == Direction.RIGHT) or (self.direction == Direction.RIGHT and direction.direction == Direction.DOWN) or (self.direction == Direction.DOWN and direction.direction == Direction.LEFT) or (self.direction == Direction.LEFT and direction.direction == Direction.UP):
				self.turnRight90Degrees()
				self.direction = direction.direction
				self.driveForward(direction)
			# Direction is to the left of ours
			else:
				self.turnLeft90Degrees()
				self.direction = direction.direction
				self.driveForward(direction)
		
	def driveToTarget(self):
		'''
		Drives to target node, scanning environment, routing, driving following the route, and stopping for people/signs.
		'''
		self.startObjectDetection()
		
		while self.position != self.target:
			self.scanEnvironment()
			self.drawRoute()
			directions = self.getRouteAsDirectionsList(10)
			self.drive(directions)
			
		self.stopObjectDetection()
			
		print('Got to target!')