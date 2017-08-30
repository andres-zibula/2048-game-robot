"""
Author:	Andres Zibula
Source:	https://github.com/andres-zibula/2048-game-robot
"""

from vision import Vision
from game_2048_ai import Game2048AI #https://github.com/andres-zibula/2048-game-ai
import serial
import time

class RobotPlayer(object):

	"""
	This class integrates the robot's vision with the actuator
	"""
	
	def __init__(self):
		"""
		Initialize the connection with the arduino
		"""

		self.conn = serial.Serial('/dev/ttyACM0',9600)
		time.sleep(2)

	def closeConnection(self):
		"""
		Close the connection with the arduino
		"""

		self.conn.close()

	def play2048(self):
		"""
		Start playing!
		"""

		vision = Vision()
		matrix = vision.getMatrix()
		gameAI = Game2048AI()

		while not gameAI.isGameOver(matrix):
			nextMove = gameAI.getNextMove(matrix)
			self.sendCommand(str(nextMove))
			matrix = vision.getMatrix()

	def sendCommand(self, command):
		"""
		Sends the command and waits a reply indicating that the execution finished
		
		Args:
		    command (str): Sends the 1 length string (8 bit character) to the arduino
		"""
		self.conn.write(command)
		stillWaiting = True

		while stillWaiting:
			res = self.conn.read(1)
			if res == '4':
				stillWaiting = False

if __name__ == "__main__":
	robot = RobotPlayer()
	robot.play2048()
	robot.closeConnection()