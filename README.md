# project-1-python-team_16

--- SIGN ALERT ---
Sign Alert is a tool used to help hard of hearing individuals communicate with individuals who do not understand American Sign Language (ASL).
What the user must have:
- PyQT5
- numpy
- Pytorch
- Pillow (PIL)
- CV2

The MUST start the tool by opening up the MainMenuGUI.py file
The user should first have a dataset they wish to use installed on their computer.
We recommend the Sign Language MNIST Dataset (https://www.kaggle.com/datasets/datamunge/sign-language-mnist).
Once the user has a dataset they wish to train their model with, they can click on the Training button on the Main menu, which will lead them to the training window.
The user can import their dataset on this training window, then they can choose which of the three current CNN's to use (LeNet5, AlexNet and BriaNet). Once they click on which of the three to use, they can then alter the parameters (Number of epochs, learning rate and batch size).
After selecting the desired parameters, the user should then choose how to name the model and they can now train the model using the selected dataset. They can also view the images in the selected dataset along with filtering by letter to see images of that letter in the dataset.

Once they have trained their model, they can then do and test the saved model by pressing the Testing button on the main menu. There they choose which saved model they would like to use and what images they would like to test with. There is also an option to take their own picture, where a square will pop up showing where to put your hand signal.
Once they pick which images they would like to test, they can click the Test Images button and a screen will pop up, showing the current image and the predictions on what letter that image contains.

Current version is version 1.0
