
import customtkinter
import trainingWindow
import testWindow
from tkinter import *
from tkinter.ttk import *
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("SignAlert")
        self.geometry(f"{650}x{450}")
        
        self.makeSidebar()
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))

        self.appearance_mode_label2 = customtkinter.CTkLabel(self.sidebar_frame, text="Language Type:", anchor="w")
        self.appearance_mode_label2.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu2 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["ASL", "NZL"])
        self.appearance_mode_optionemenu2.grid(row=8, column=0, padx=20, pady=(10, 10))


        self.main_button_1 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text="Ok", text_color=("gray10", "#DCE4EE"))
        self.main_button_1.grid(row=1, column=2, padx=(30, 30), pady=(30, 30), sticky="nsew")

        self.main_button_2 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text="Exit", text_color=("gray10", "#DCE4EE"), command=self.destroy)
        self.main_button_2.grid(row=1, column=3, padx=(30, 30), pady=(30, 30), sticky="nsew")

        self.makeTabview()

    def makeSidebar(self):
        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="SignAlertTool", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Training", command=trainingWindow.trainWindow)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=testWindow.app, text="Testing")
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=4, column=0, padx=20, pady=(10, 10))

    
    def makeTabview(self):
        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Tab1")
        self.tabview.add("Tab2")
        self.tabview.tab("Tab1").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Tab2").grid_columnconfigure(0, weight=1)
        self.tabviewButtons()

    def tabviewButtons(self):
        self.optionmenu_1 = customtkinter.CTkOptionMenu(self.tabview.tab("Tab1"), dynamic_resizing=False,
                                                        values=["Whatever 1", "Whatever 2"])    # think about what use we can make
        self.optionmenu_1.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.combobox_1 = customtkinter.CTkComboBox(self.tabview.tab("Tab1"),
                                                    values=["Whatever 1", "Whatever 2"])
        self.combobox_1.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Tab1"), text="Open ASL Dialog",
                                                           command=self.open_input_dialog_event_ASL)
        self.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        
        self.optionmenu_2 = customtkinter.CTkOptionMenu(self.tabview.tab("Tab2"), dynamic_resizing=False,
                                                        values=["Whatever 1", "Whatever 2"])    # think about what use we can make
        self.optionmenu_2.grid(row=0, column=0, padx=20, pady=(20, 10)) 
        self.combobox_2 = customtkinter.CTkComboBox(self.tabview.tab("Tab2"),
                                                    values=["Whatever 1", "Whatever 2"])
        self.combobox_2.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Tab2"), text="Open NZL Dialog",
                                                           command=self.open_input_dialog_event_NZL)
        self.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))
    def open_input_dialog_event_ASL(self):  # how to make nzl and asl seperate?
        asl_dialog = customtkinter.CTkInputDialog(text="Type in Alphabet:", title="ASL Dialog")
        print("ASL Language:", asl_dialog.get_input())

    def open_input_dialog_event_NZL(self):  # how to make nzl and asl seperate?
        nzl_dialog = customtkinter.CTkInputDialog(text="Type in Alphabet:", title="NZL Dialog")
        print("NZL Langauge:", nzl_dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("Attempting to load: ", self.sidebar_button_event)



if __name__ == "__main__":
    app = App()
    app.mainloop()
