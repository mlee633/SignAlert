# ---Sign Alert---

Made by group 16 for COMPSYS302 Project 1


Sign Alert made during a unviersity course, is a tool used to assist hard of hearing individuals to communicate with individuals who do not understand American Sign Language (ASL). 
Our tool is aiming to help raise awareness and promote sign language to the world, as we believe that not enough people in the world understand sign language. The tool provides a way to train models for predicting ASL signs as well as a tool to interpret signs and returns the letter the sign represents.

## Installation
What the user must have:
- PyQT5
- numpy
- Pytorch
- Pillow (PIL)
- CV2

The user MUST start the tool by opening up the MainMenuGUI.py file.

## Usage / Instructions
The user should first have a dataset they wish to use installed on their computer.
We recommend the Sign Language MNIST Dataset (https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

Once the user has a dataset they wish to train their model with, they can click on the Training button on the Main menu, which will lead them to the training window.

The user can import their dataset on this training window, then they can choose which of the three current CNN's to use (LeNet5, AlexNet and BriaNet). Once they click on which of the three to use, they can then alter the parameters (Number of epochs, learning rate and batch size).
After selecting the desired parameters, the user should then choose how to name the model and they can now train the model using the selected dataset. They can also view the images in the selected dataset along with filtering by letter to see images of that letter in the dataset.

Once they have trained their model, they can then do and test the saved model by pressing the Testing button on the main menu. There they choose which saved model they would like to use and what images they would like to test with. There is also an option to take their own picture, where a square will pop up showing where to put your hand signal.

Once they pick which images they would like to test, they can click the Test Images button and a screen will pop up, showing the current image and the predictions on what letter that image contains.

## Version Number

Current version is Sign Alert v1.1

## Contributors:

Brian Wei: Developing backend AI of Sign Alert, as well as a bit of frontend (Progress bars etc.) Made BriaNet and parts of AlexNet

Isaac Lee: Developing mostly frontend and parts of backend AI, created the camera module. Made AlexNet with help from Brian

Shaaran Elango: Devloping front end design, worked on some backend. Made LeNet5

