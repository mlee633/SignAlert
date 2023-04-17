import tkinter as tk
from PIL import ImageTk, Image
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('C:\\Users\\healt\\OneDrive\\문서\\GitHub\\project-1-python-team_16\\dataset\\sign_mnist_train.csv')

# Convert the pixel values to numpy arrays
images = df.iloc[:, 1:].values.astype(np.uint8)

# Create the main window
window = tk.Tk()
window.title("Train Images")

# Create the File and View menus
menu_bar = tk.Menu(window)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open")
file_menu.add_command(label="Save")
file_menu.add_separator()
file_menu.add_command(label="Exit", command=window.quit)
view_menu = tk.Menu(menu_bar, tearoff=0)
view_menu.add_command(label="View as List")
view_menu.add_command(label="View as Grid")
menu_bar.add_cascade(label="File", menu=file_menu)
menu_bar.add_cascade(label="View", menu=view_menu)
window.config(menu=menu_bar)

# Create the Alphabet label and entry box
alphabet_label = tk.Label(window, text="Alphabet:")
alphabet_label.pack(side=tk.LEFT)
alphabet_entry = tk.Entry(window)
alphabet_entry.pack(side=tk.LEFT)

# Create the OK button
ok_button = tk.Button(window, text="OK", command=lambda: print("Okay"))
ok_button.pack(side=tk.BOTTOM, padx=10, pady=10)

# Create the canvas to display the images
canvas = tk.Canvas(window, width=400, height=400)
canvas.pack()

# Load the first image and display it
img = Image.fromarray(images[0].reshape((28, 28)))
img_tk = ImageTk.PhotoImage(img)
canvas_image = canvas.create_image(200, 200, image=img_tk)

# Create a function to update the displayed image and alphabet entry
def update_image(index):
    img = Image.fromarray(images[index].reshape((28, 28)))
    img_tk = ImageTk.PhotoImage(img)
    canvas.itemconfig(canvas_image, image=img_tk)
    alphabet_entry.delete(0, tk.END)
    alphabet_entry.insert(0, chr(df.iloc[index, 0] + 65))

# Create a function to handle key presses
def handle_keypress(event):
    global index
    if event.keysym == "Left":
        if index > 0:
            index -= 1
            update_image(index)
    elif event.keysym == "Right":
        if index < len(df) - 1:
            index += 1
            update_image(index)

# Bind the canvas to the key press event
canvas.bind("<KeyPress>", handle_keypress)
canvas.focus_set()

# Keep track of the current image index
index = 0

# Update the canvas image and alphabet entry
update_image(index)

# Start the main loop
window.mainloop()
