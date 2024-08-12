from direction import Direction

class DriveInstruction:
	'''
	Represents an instruction to drive a given distance in a given direction.

	Attributes:
	distance (int): Distance to drive in the given direction.
	direction (Direction): Direction in 2D space to drive.
	'''
	def __init__(self, distance, direction):
		'''
		Initialize a new DriveInstruction instance.

		Parameters:
		distance (int): Distance to drive in the given direction.
		direction (Direction): Direction in 2D space to drive.
		'''
		self.distance = distance
		self.direction = direction
		
	def __str__(self):
		return f'Distance: {self.distance}, Direction: {self.direction}'
		
	def incrementDistance(self):
		'''
		Increases drive distance by one
		'''
		self.distance += 1