import customtkinter
from tkinter import *
from tkinter.ttk import *
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class trainWindow(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("SignAlert - Train")
        self.geometry(f"{650}x{450}")
        self.main_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text="Ok", text_color=("gray10", "#DCE4EE"))

if __name__ == "__main__":
    app = trainWindow()
    app.mainloop()
        