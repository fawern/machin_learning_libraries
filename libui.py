import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk 

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

def train_button_click():
    return None

def on_close_click():
    root.destroy()
      
def library_ui():
    global label
    global root 
    global model_type

    colour1 = '#020f12' # Background color
    colour2 = '#05d7ff' # Default button background color
    colour3 = '#246C71' # Button background color when clicked
    colour4 = 'black' # Text color

    root = tk.Tk()
    root.title("Machine Learnings Library")
    root.resizable(True, True)
    root.geometry("800x600")  # Width x Height
    root.configure(background=colour1)

    label = tk.Label(root, text="Add a CSV file for models")
    label.pack(pady=10)

    upload_button = tk.Button(
        root,
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
        font=('Arial', 16, 'bold')
    )
    upload_button.pack(pady=10)

    n = tk.StringVar()
    model_type = ttk.Combobox(root, width=12, textvariable=n)
    model_type['values'] = ('Classification', 'Regression')
    model_type.pack(pady=10) 

    train_button = tk.Button(
        root,
        text='Train Models',
        command=train_button_click,
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
        root,
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