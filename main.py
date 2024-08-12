from car import Car

if __name__ == '__main__':
	'''
	Defines a Car object with a target destination of 75 cm to the right and 150 cm in front of its current
	position.

	Tells the car to drive to its target destination, if a route from where it is to that destination can be determined.
	'''
	car = Car(75, 150)
	car.driveToTarget()