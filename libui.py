import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk 

import customtkinter
import CTkTable 

import classification
import regression

file_path = ''

def upload_button_click():
    global file_path

    file_path = filedialog.askopenfilename(title='Select a CSV file !!!')
    label.config(text=file_path.split('/')[-1])

    if file_path:
        if file_path.split('.')[-1] != 'csv':
            messagebox.showinfo(title='File type error', message="Please select a CSV file")
    else:
        messagebox.showinfo(title='Not found error', message="No file selected")

def train_button_click(selected_model):
    if selected_model == 'Classification':
        print(selected_model)
    
    elif selected_model == 'Regression':
        pass

    else:
        messagebox.showinfo(title='Not found error', message="Model type not selected")
    return None

# def change_theme():
#     customtkinter.set_appearance_mode("light")

def on_close_click():
    root.destroy()

def library_ui():
    global label
    global root 
    global model_type

    # colour1 = '#292929' # Background color
    colour1 = '#161616'
    colour2 = '#05d7ff' # Default button background color
    colour3 = '#246C71' # Button background color when clicked
    colour4 = 'black' # Text color

    root = customtkinter.CTk()
    
    root.title("Machine Learnings Library")
    root.resizable(True, True)
    root.geometry("1000x600")  # Width x Height
    root.configure(background=colour1)
    
    left_container = customtkinter.CTkFrame(root, corner_radius=15)
    left_container.place(relx=0.03, rely=0.05, relwidth=0.3, relheight=0.9)

    label = tk.Label(left_container, text="Add a CSV file for models")
    label.pack(pady=10)

    upload_button = tk.Button(
        left_container,
        text='Upload',
        command=upload_button_click,
        background=colour2, # Background color when not hovered
        foreground=colour4, # Text color
        activebackground=colour3, # Background color when hovered/clicked
        activeforeground=colour4, # Text color when hovered/clicked
        highlightthickness=2, # Thickness of the highlight border
        highlightbackground=colour2, # Highlight border color
        highlightcolor='WHITE', # Highlight color
        width=8,
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

    value = [[1,2,3,4,5],
            [1,2,3,4,5],
            [1,2,3,4,5],
            [1,2,3,4,5],
            [1,2,3,4,5]]

    table_container = customtkinter.CTkFrame(root, corner_radius=15)
    table_container.place(relx=0.35, rely=0.05, relwidth=0.6, relheight=0.9)

    table = CTkTable.CTkTable(master=table_container, row=5, column=5, values=value, header_color=colour1)
    table.pack(expand=True, fill="both", padx=20, pady=20)

    root.mainloop()
    
    return file_path

library_ui()