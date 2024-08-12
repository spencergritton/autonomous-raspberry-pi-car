import math
import numpy as np

class Node():
	'''
	Represents a location in 2D space within the Cars internal map of the environment.

	Attributes:
	x (int): X position in space.
	y (int): Y position in space.
	envMap (2D NP array): Array of Nodes in internal map of the car.
	val (int): Value of node (-1 vehicle, 0 air, 1 obstacle, 5 destination).
	f (float): Cost to arrive at this node from the vehicle towards the destination.
	parent (Node): Node that came before this one in-route to the destination.
	child (Node): Node that will come after this one in-route to the destination.
	'''
	def __init__(self, x, y, envMap=None, f=0, parent=None):
		'''
		Initialize a new Node instance

		Parameters:
		x (int): X position in space.
		y (int): Y position in space.
		envMap (2D NP array): Array of Nodes in internal map of the car.
		f (float): Cost to arrive at this node from the vehicle towards the destination.
		parent (Node): Node that came before this one in-route to the destination.
		'''
		self.x = x
		self.y = y
		self.envMap = envMap
		self.val = envMap[y, x] if self.isInbounds() else None
		self.f = f
		self.parent = parent
		self.child = None # Child used for tracing route from start to end

	def __eq__(self, other):
		if isinstance(other, Node):
			return self.x == other.x and self.y == other.y and self.val == other.val

		return False

	def __hash__(self):
		return hash(f'{self.x}-{self.y}-{self.val}')

	def __lt__(self, other):
		return self.f > other.f

	def __str__(self):
		return f'({self.x}, {self.y}) val: {self.val}, f: {self.f}, hasParent: {self.parent is not None}'
    
	def isInbounds(self):
		'''
		Checks if the given instance of the Node is inside the environment map (which is shared between Nodes).
		'''
		if self.envMap is None:
			return False

		rows, cols = self.envMap.shape
		return self.x >= 0 and self.x < cols and self.y >= 0 and self.y < rows
		
	def isObstacle(self):
		'''
		Checks if the given instance of the Node is an obstacle the car should avoid.
		'''
		return self.envMap[self.y, self.x] == 1
		
	def updateVal(self):
		'''
		Updates the value of the node to the given value it has in the Cars environment map (which is shared between Nodes).
		'''
		if self.envMap is not None:
			self.val = self.envMap[self.y, self.x]
        
	def updatePenalty(self, target):
		'''
		Given a target Node, updates this instances f (a*) penalty to allow a* to follow the path of least penalty.

		Utilizes a weighted combination of:
		- the count of nearby obstacles (heavily weighted)
		- the euclidean distance from this node to the target
		- +1 (for having to travel)
		in order to calculate the distance penalty from hitting this node in the path traversal.

		i.e. self.f = parent.f + 1 + dist(this, target) * 0.1 + nearbyObstacleCount * 30

		Parameters:
		target (Node): Node to travel to from this node.
		'''
		if not self.isInbounds():
			return
			
		# Get obstacles nearby and add weight depending on number of them
		localZone = self.envMap[max(0, self.y-2):self.y+3, max(0, self.x-5):self.x+6]
		localizedOnesZone = np.count_nonzero(localZone == 1)
		localizedPenalty = localizedOnesZone * 30

		# Get euclidean distance from position to target and add portion of it as weight
		euclideanDistanceToTarget = math.sqrt((target.x - self.x)**2 + (target.y - self.y)**2) * .1

		self.f = self.parent.f + 1 + euclideanDistanceToTarget + localizedPenalty