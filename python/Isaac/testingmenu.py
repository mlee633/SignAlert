import customtkinter
import testWindow
def makeSidebar(menu):
    # create sidebar frame with widgets
    menu.CTkFrame(menu, width=140, corner_radius=0)
    menu.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
    menu.sidebar_frame.grid_rowconfigure(4, weight=1)
    menu.logo_label = customtkinter.CTkLabel(menu.sidebar_frame, text="SignAlert", font=customtkinter.CTkFont(size=20, weight="bold"))
    menu.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
    menu.sidebar_button_1 = customtkinter.CTkButton(menu.sidebar_frame, text="Training", command=testWindow)
    menu.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
    menu.sidebar_button_2 = customtkinter.CTkButton(menu.sidebar_frame, command=testWindow.app, text="Testing")
    menu.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
    menu.appearance_mode_label = customtkinter.CTkLabel(menu.sidebar_frame, text="Appearance Mode:", anchor="w")
    menu.appearance_mode_label.grid(row=3, column=0, padx=20, pady=(10, 0))
    menu.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(menu.sidebar_frame, values=["System", "Light", "Dark"],
                                                                    command=menu.change_appearance_mode_event)
    menu.appearance_mode_optionemenu.grid(row=4, column=0, padx=20, pady=(10, 10))

def centre(menu):
    screen_width = menu.winfo_screenwidth()
    screen_height = menu.winfo_screenheight()
    x_cordinate = int((screen_width/2) - (650/2))
    y_cordinate = int((screen_height/2) - (450/2))
    menu.geometry("{}x{}+{}+{}".format(650, 450, x_cordinate, y_cordinate))
def makeTabview(menu):
    # create tabview
    menu.tabview = customtkinter.CTkTabview(menu, width=250)
    menu.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
    menu.tabview.add("Tab1")
    menu.tabview.add("Tab2")
    menu.tabview.tab("Tab1").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
    menu.tabview.tab("Tab2").grid_columnconfigure(0, weight=1)
    menu.tabviewButtons()

def tabviewButtons(menu):
    menu.optionmenu_1 = customtkinter.CTkOptionMenu(menu.tabview.tab("Tab1"), dynamic_resizing=False,
                                                    values=["Whatever 1", "Whatever 2"])    # think about what use we can make
    menu.optionmenu_1.grid(row=0, column=0, padx=20, pady=(20, 10))
    menu.combobox_1 = customtkinter.CTkComboBox(menu.tabview.tab("Tab1"),
                                                values=["Whatever 1", "Whatever 2"])
    menu.combobox_1.grid(row=1, column=0, padx=20, pady=(10, 10))
    menu.string_input_button = customtkinter.CTkButton(menu.tabview.tab("Tab1"), text="Open ASL Dialog",
                                                        command=menu.open_input_dialog_event_ASL)
    menu.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

    
    menu.optionmenu_2 = customtkinter.CTkOptionMenu(menu.tabview.tab("Tab2"), dynamic_resizing=False,
                                                    values=["Whatever 1", "Whatever 2"])    # think about what use we can make
    menu.optionmenu_2.grid(row=0, column=0, padx=20, pady=(20, 10)) 
    menu.combobox_2 = customtkinter.CTkComboBox(menu.tabview.tab("Tab2"),
                                                values=["Whatever 1", "Whatever 2"])
    menu.combobox_2.grid(row=1, column=0, padx=20, pady=(10, 10))
    menu.string_input_button = customtkinter.CTkButton(menu.tabview.tab("Tab2"), text="Open NZL Dialog",
                                                        command=menu.open_input_dialog_event_NZL)
    menu.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))
def open_input_dialog_event_ASL(menu):  # how to make nzl and asl seperate?
    asl_dialog = customtkinter.CTkInputDialog(text="Type in Alphabet:", title="ASL Dialog")
    print("ASL Language:", asl_dialog.get_input())

def open_input_dialog_event_NZL(menu):  # how to make nzl and asl seperate?
    nzl_dialog = customtkinter.CTkInputDialog(text="Type in Alphabet:", title="NZL Dialog")
    print("NZL Langauge:", nzl_dialog.get_input())

def change_appearance_mode_event(menu, new_appearance_mode: str):
    customtkinter.set_appearance_mode(new_appearance_mode)

def change_scaling_event(menu, new_scaling: str):
    new_scaling_float = int(new_scaling.replace("%", "")) / 100
    customtkinter.set_widget_scaling(new_scaling_float)

# menu 
menu = customtkinter.CTk()
menu.title('SignAlert')
menu.geometry('600x400')

# widgets
label = customtkinter.CTkLabel(
    menu,
    text = 'A ctk label',
    fg_color = 'red',
    text_color = 'white',
    corner_radius = 10)
label.pack()

button = customtkinter.CTkButton(menu, text = 'A ctk button', fg_color = '#FF0', text_color = '#000')
button.pack()

menu.title("SignAlert")
    # Making the screen centred #
centre(menu)

makeSidebar(menu)
menu.scaling_label = customtkinter.CTkLabel(menu.sidebar_frame, text="UI Scaling:", anchor="w")
menu.scaling_label.grid(row=5, column=0, padx=20, pady=(10, 0))
menu.scaling_optionemenu = customtkinter.CTkOptionMenu(menu.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                        command=menu.change_scaling_event)
menu.scaling_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))

menu.appearance_mode_label2 = customtkinter.CTkLabel(menu.sidebar_frame, text="Language Type:", anchor="w")
menu.appearance_mode_label2.grid(row=7, column=0, padx=20, pady=(10, 0))
menu.appearance_mode_optionemenu2 = customtkinter.CTkOptionMenu(menu.sidebar_frame, values=["ASL", "NZL"])
menu.appearance_mode_optionemenu2.grid(row=8, column=0, padx=20, pady=(10, 10))


menu.main_button_1 = customtkinter.CTkButton(master=menu, fg_color="transparent", border_width=2, text="Ok", text_color=("gray10", "#DCE4EE"))
menu.main_button_1.grid(row=1, column=2, padx=(30, 30), pady=(30, 30), sticky="nsew")

menu.main_button_2 = customtkinter.CTkButton(master=menu, fg_color="transparent", border_width=2, text="Exit", text_color=("gray10", "#DCE4EE"), command=menu.destroy)
menu.main_button_2.grid(row=1, column=3, padx=(30, 30), pady=(30, 30), sticky="nsew")

makeTabview(menu)
# run
menu.mainloop()
