"""
Author:	Andres Zibula
Source:	https://github.com/andres-zibula/2048-game-robot
"""

import time
import cv2
import pytesseract
import multiprocessing
import numpy as np
from PIL import Image
from imutils.perspective import four_point_transform

def cellToText(data):
	"""
	This function is called by multiprocessing.Pool() for parallelism
	
	Args:
	    data (Set): A set containing info about a cell: {'x': x, 'y': y, 'empty': False, 'board': board, 'cellHeight': cellHeight, 'cellWidth': cellWidth, 'cell': None, 'value': 0, 'valid': False}
	
	Returns:
	    Set: The processed cell, with its text value
	"""

	if(data['empty'] or data['valid']): #if cell is empty or we already know it is a valid number we just return
		return data

	x = data['x']
	y = data['y']
	cellHeight, cellWidth = data['cellHeight'], data['cellWidth']
	trimPixels = 3 #erase the borders of the cell by 3 pixels

	#check if the cell contains a number, if no we mark the cell as empty and return
	cellToCheck = data['board'][y*cellHeight+cellHeight/3:y*cellHeight+cellHeight-cellHeight/3, x*cellWidth+cellWidth/3:x*cellWidth+cellWidth-cellWidth/3]
	mean = cv2.mean(cellToCheck)
	if(mean[0] > 10.0):
		#this cell is not empty
		data['empty'] = False
		cell = data['board'][y*cellHeight+trimPixels:y*cellHeight+cellHeight-trimPixels, x*cellWidth+trimPixels:x*cellWidth+cellWidth-trimPixels]
		
		(contours, hierarchy) = cv2.findContours(cell.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#after obtaining the contours, filter them by area
		contoursGt16 = [contour for contour in contours if cv2.contourArea(contour) >= 16]
		contoursLt16 = [contour for contour in contours if cv2.contourArea(contour) < 16]
		
		#fill the contours
		for c in contoursGt16:
			cv2.fillPoly(cell, pts =[c], color=(255,255,255))
		for c in contoursLt16:
			cv2.fillPoly(cell, pts =[c], color=(0,0,0))

		#apply erode for better visibility
		kernel = np.ones((3,3),np.uint8)
		cell = cv2.erode(cell,kernel,iterations = 1)

		#change black to white and vice-versa
		cell = cv2.bitwise_not(cell)
		data['cell'] = cell

		#run tesseract to get the string
		data['value'] = pytesseract.image_to_string(Image.fromarray(cell), config='-psm 8 -l eng -c tessedit_char_whitelist=0123456789')

		#if the string is empty it means tesseract failed to get the correct text,
		#we return and then comeback later with a slighty different image
		if data['value'] == '':
			return data
		
		n = int(data['value'].split()[0])

		#a cell is valid if it is a power of 2 (except for 1)
		if(n != 0 and n != 1 and ((n & (n - 1)) == 0)):
			data['valid'] = True

		data['value'] = n
	
		return data

	#cell does not contain valid number, mark it as empty
	data['empty'] = True
	data['valid'] = True
	return data

class Vision(object):
	"""
	This class provides the functionality for capturing the image
	from a camera and processing it to get the game board matrix of numbers
	"""
	
	def __init__(self):
		"""
		Initialize the camera and start a daemon thread for capturing the images,
		this is done in order to reduce the "lag" between frames
		"""

		self.cap = cv2.VideoCapture(0)
		self.__debug = True

		self.frameQueue = multiprocessing.Queue(1)
		self.captureDaemon = multiprocessing.Process(name='captureDaemon', target=self.updateFrame, args=(self.frameQueue,))
		self.captureDaemon.daemon = True
		self.captureDaemon.start()

	def close(self):
		"""
		Stop the thread and free the camera
		"""

		self.captureDaemon.terminate()
		self.captureDaemon.join()
		self.cap.release()
		cv2.destroyAllWindows()

	def updateFrame(self, frameQueue):
		"""
		This function is called by the daemon thread to get the last frame from the camera
		Note: this Queue is a workaround since multiprocessing does not accept a Stack :(
		
		Args:
		    frameQueue (Queue): This Queue of size 1 stores the last frame from the camera
		"""

		while(True):
			#capture frame
			frameRet, actualFrame = self.cap.read()
			#while the frame is not valid we try again
			while (frameRet == False): 
				frameRet, actualFrame = self.cap.read()
				time.sleep(0.1)

			#if the queue is full we erase the element, if the main thread gets the frame first,
			#the queue will be empty and this raises a exception, so we try first :P
			if(frameQueue.full()):
				try:
					frameQueue.get(False)
				except:
					pass

			#put the last frame
			frameQueue.put(actualFrame)
	
	def getGameBoardImg(self):
		"""
		Locate the board and return an image of it
		
		Returns:
		    Numpy array: The image of the board
		"""

		#get the image of the the camera
		frame = self.frameQueue.get()
		boardRotated = None

		#color to gray
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#reduce noise
		gray2 = cv2.bilateralFilter(gray, 11, 17, 17)
		#detect edges
		edges = cv2.Canny(gray, 24, 140)

		#get the contours
		(contours, hierarchy) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contoursFiltered = []
	
		for c in contours:
			#approximate the contour to a polygon
			peri = cv2.arcLength(c, True)
			approxContour = cv2.approxPolyDP(c, 0.02 * peri, True)
		 	
		 	#we are looking for a trapezoid, so we check if the polygon has 4 points
			if len(approxContour) == 4:
				contoursFiltered.append([c, approxContour])
		
		#we ensure that it has an area greater than 20000, and then sort them by area, so then we get the smallest
		contoursFiltered = [contour for contour in contoursFiltered if cv2.contourArea(contour[0]) > 20000]
		contoursFiltered = sorted(contoursFiltered, key = lambda contour: cv2.contourArea(contour[0]))

		#if the contour meets the requirements
		if contoursFiltered:
			contour = contoursFiltered[0][1] #the "approxContour" polygon
			contour = contour.reshape(4, 2) #we change the array shape for the following function to work

			#transform the trapezoid into a rectangle
			board = four_point_transform(frame, contour)
	
			#rotate it
			boardRotated = cv2.transpose(board)
			boardRotated = cv2.flip(boardRotated,flipCode=0)

			if self.__debug:
				cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
				cv2.imshow('board',boardRotated)
				cv2.moveWindow('board', 100, 640)

		if self.__debug:
			cv2.imshow('frame',frame)
			cv2.moveWindow('frame', 30, 100)
			cv2.imshow('gray without noise',gray2)
			cv2.moveWindow('gray without noise', 600, 100)
			cv2.imshow('edges',edges)
			cv2.moveWindow('edges', 1200, 100)

		return boardRotated

	def getGameBoardMatrix(self, gameBoardImg, boardMatrix):
		"""
		We divide the board in 16 cells and we run tesseract in each cell to get the number
		
		Args:
		    gameBoardImg (Numpy array): The image of the board
		    boardMatrix (List): A list of sets with info of every cell in the matrix
		
		Returns:
		    List: Returns a list of sets with info of every cell in the matrix
		"""

		#convert color to grayscale
		grayBoard = cv2.cvtColor(gameBoardImg, cv2.COLOR_BGR2GRAY)
		#detect the edges
		board = cv2.Canny(grayBoard, 20, 80)

		boardHeight, boardWidth = board.shape[:2]
		cellHeight, cellWidth = boardHeight/4, boardWidth/4

		if boardMatrix:
			for i in range(4*4):
				boardMatrix[i]['board'] = board
		else:
			for x in range(4):
				for y in range(4):
					boardMatrix.append({'x': x, 'y': y, 'empty': False, 'board': board, 'cellHeight': cellHeight, 'cellWidth': cellWidth, 'cell': None, 'value': 0, 'valid': False})

		#run our function cellToText for every cell in the matrix in parallel, this speeds up 4 times faster in my quadcore CPU
		pool = multiprocessing.Pool(4)
		boardMatrix = pool.map(cellToText, boardMatrix)
		pool.close()
		pool.join()

		if self.__debug:
			#create a white image of the size of the board
			processedBoard = np.zeros((boardHeight,boardWidth), np.uint8)
			processedBoard[:] = (255)
			
			#unify the processed cells images on this image and show it
			for i in range(4*4):
				if not boardMatrix[i]['empty']:
					processedBoard[boardMatrix[i]['y']*cellHeight:boardMatrix[i]['y']*cellHeight+boardMatrix[i]['cell'].shape[0], boardMatrix[i]['x']*cellWidth:boardMatrix[i]['x']*cellWidth+boardMatrix[i]['cell'].shape[1]] = boardMatrix[i]['cell']

			cv2.imshow('board gray',grayBoard)
			cv2.moveWindow('board gray', 400, 640)
			cv2.imshow('board edges',board)
			cv2.moveWindow('board edges', 700, 640)
			cv2.imshow('processed', processedBoard)
			cv2.moveWindow('processed', 1000, 640)

		return boardMatrix

	def isGameBoardMatrixValid(self, boardMatrix):
		"""
		Check that all elements of the matrix are valid
		
		Args:
		    boardMatrix (List): A list containing info about the cells
		
		Returns:
		    Bool: True if all are valid, else False
		"""
		for i in range(4*4):
			if not boardMatrix[i]['valid']:
				return False
		return True

	def getNumbersMatrix(self):
		"""
		Returns the numbers of the matrix
		
		Returns:
		    List: List of numbers in the matrix
		"""
		gameBoardMatrix = []

		#we keep looping until we get a valid matrix
		while True:
			gameBoardImg = self.getGameBoardImg()
			if gameBoardImg is not None:
				gameBoardMatrix = self.getGameBoardMatrix(gameBoardImg, gameBoardMatrix)
				if self.isGameBoardMatrixValid(gameBoardMatrix):
					numbersMatrix = []
					for i in range(4*4):
						numbersMatrix.append(gameBoardMatrix[i]['value'])
					return numbersMatrix

	def printMatrix(self, matrix):
		"""
		Prints the matrix
		
		Args:
		    matrix (List): The matrix of numbers
		"""
		print("")
		print("+" + "-"*4 + "+" + "-"*4 + "+" + "-"*4 + "+" + "-"*4 + "+")
		print('|{0: ^4}|{1: ^4}|{2: ^4}|{3: ^4}|'.format(matrix[0], matrix[4], matrix[8], matrix[12]))
		print("|" + "-"*19 + "|")
		print('|{0: ^4}|{1: ^4}|{2: ^4}|{3: ^4}|'.format(matrix[1], matrix[5], matrix[9], matrix[13]))
		print("|" + "-"*19 + "|")
		print('|{0: ^4}|{1: ^4}|{2: ^4}|{3: ^4}|'.format(matrix[2], matrix[6], matrix[10], matrix[14]))
		print("|" + "-"*19 + "|")
		print('|{0: ^4}|{1: ^4}|{2: ^4}|{3: ^4}|'.format(matrix[3], matrix[7], matrix[11], matrix[15]))
		print("+" + "-"*4 + "+" + "-"*4 + "+" + "-"*4 + "+" + "-"*4 + "+")

	def getMatrix(self):
		"""
		Return the numpy array matrix of numbers
		
		Returns:
		    Numpy array: Matrix of numbers
		"""

		numbersMatrix = self.getNumbersMatrix()

		if self.__debug:
			self.printMatrix(numbersMatrix)
			cv2.waitKey(10)

		return np.reshape(np.array(numbersMatrix), (4,4))

	def test(self):
		"""
		testing stuff
		"""
		while(True):
			numbersMatrix = self.getNumbersMatrix()
			self.printMatrix(numbersMatrix)

			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
