# 2048 Game Robot

![](https://github.com/andres-zibula/project-images/blob/master/2048_game_robot/2048_game_robot.gif)

It is a robot that plays the game 2048 in a smartphone, it uses the built-in camera of the notebook to capture the images, then it processes them to get the matrix of numbers of the game board, after that through a Neural Network it decides what action to take. Finally the program sends signals to the Arduino to slide the stylus pen in the correct direction.

[Watch video](https://youtu.be/B5Zup7bcReA)

## Requirements

The AI is separated in another repo: [2048 Game AI](https://github.com/andres-zibula/2048-game-ai), and the arduino code: [Parallel Scara Stylus](https://github.com/andres-zibula/parallel-scara-stylus)

The following libraries are required:

- [Numpy](http://www.numpy.org/)
- [OpenCV](http://opencv.org/)
- [Imutils](https://github.com/jrosebr1/imutils)
- [Pillow](https://github.com/python-pillow/Pillow)
- [PySerial](https://github.com/pyserial/pyserial)
- [PyTesseract](https://pypi.python.org/pypi/pytesseract)
