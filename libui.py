import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk 

import customtkinter
import CTkTable 

import pandas as pd 
from sklearn.model_selection import train_test_split
import classification as cls_
import regression

file_path = ''
def upload_button_click():
    global file_path

    file_path = filedialog.askopenfilename(title='Select a CSV file !!!')
    if file_path:
        if file_path.split('.')[-1] != 'csv':
            messagebox.showinfo(title='File type error', message="Please select a CSV file")
    else:
        messagebox.showinfo(title='Not found error', message="No file selected")


models_output_list = []
def train_button_click(selected_model):
    global models_output_list
    
    if selected_model == 'Classification':
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1]

        y = y.map({'Onaylandı': 1, 'Reddedildi': 0})

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=42
        )       

        models_output_list = cls_.classification_models(X_train, X_test, y_train, y_test)[0]

        print(models_output_list)

        show_model_output(models_output_list)

    elif selected_model == 'Regression':
        print(selected_model)

    else:
        messagebox.showinfo(title='Not found error', message="Model type not selected")

    return None

def show_model_output(output_df):
    value = [list(models_output_list.columns)]

    for row in models_output_list.values:
        value.append(list(row))

    table_container = customtkinter.CTkFrame(root, corner_radius=15)
    table_container.place(relx=0.3, rely=0.05, relwidth=0.68, relheight=0.9)

    table = CTkTable.CTkTable(
        master=table_container, row=len(output_df) + 1, column=len(output_df.columns), 
        values=value, header_color="#204261")
    table.pack(expand=True, fill="both", padx=20, pady=20)

# def change_theme():
#     customtkinter.set_appearance_mode("light")

def on_close_click():
    root.destroy()

def library_ui():
    global label
    global root 
    global model_type
    global colour1

    # colour1 = '#292929' # Background color
    colour1 = '#161616'
    colour2 = '#05d7ff' # Default button background color
    colour3 = '#246C71' # Button background color when clicked
    colour4 = 'black' # Text color

    root = customtkinter.CTk()
    
    root.title("Machine Learnings Library")
    root.resizable(True, True)
    root.geometry("1400x600")  # Width x Height
    root.configure(background=colour1)
    
    left_container = customtkinter.CTkFrame(root, corner_radius=15)
    left_container.place(relx=0.025, rely=0.05, relwidth=0.25, relheight=0.9)

    # label = tk.Label(left_container, text="Add a CSV file for models")
    # label.pack(pady=10)

    upload_button = tk.Button(
        left_container,
        text='Upload .csv',
        command=upload_button_click,
        background=colour2, # Background color when not hovered
        foreground=colour4, # Text color
        activebackground=colour3, # Background color when hovered/clicked
        activeforeground=colour4, # Text color when hovered/clicked
        highlightthickness=2, # Thickness of the highlight border
        highlightbackground=colour2, # Highlight border color2
        highlightcolor='WHITE', # Highlight color
        width=9,
        height=1,
        border=0,
        cursor='hand1',
        font=('Arial', 16, 'bold'),
    )
    upload_button.pack(pady=10)

    combobox_var = customtkinter.StringVar(value="Machine Learning Type")
    combobox = customtkinter.CTkComboBox(
        master=left_container,
        width=170,     
        values=['Classification', 'Regression'],
        variable=combobox_var
    )
    combobox.pack(padx=20, pady=10)

    train_button = tk.Button(
        left_container,
        text='Train Models',
        command=lambda : train_button_click(combobox.get()),
        background=colour2, # Background color when not hovered
        foreground=colour4, # Text color
        activebackground=colour3, # Background color when hovered/clicked
        activeforeground=colour4, # Text color when hovered/clicked
        highlightthickness=2, # Thickness of the highlight border
        highlightbackground=colour2, # Highlight border color
        highlightcolor='WHITE', # Highlight color
        width=10,
        height=1,
        border=0,
        cursor='hand1',
        font=('Arial', 16, 'bold')
    )
    train_button.pack(pady=10)

    close_button = tk.Button(
        left_container,
        text='Close',
        command=on_close_click,
        background=colour3,
        foreground=colour2,
        activebackground=colour3,
        activeforeground=colour4,
        highlightthickness=2,
        highlightbackground=colour2,
        highlightcolor='WHITE',
        width=8,
        height=1,
        border=0,
        cursor='hand1',
        font=('Arial', 16, 'bold')
    )
    close_button.pack(side='bottom', pady=50)


    root.mainloop()
    
    return file_path

library_ui()