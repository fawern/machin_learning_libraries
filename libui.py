import tkinter as tk
from tkinter import filedialog, messagebox

def library_ui():

    file_path = ''

    def on_button_click():
        
        nonlocal file_path

        label.config(text="Hello " + entry.get())

        file_path = filedialog.askopenfilename(title='Select a CSV file !!!')

        if file_path:
            if file_path.split('.')[-1] != 'csv':
                messagebox.showinfo(title='File type error', message="Please select a CSV file")
        else:
            messagebox.showinfo(title='Not found error', message="No file selected")

    colour1 = '#020f12'
    colour2 = '#05d7ff'
    colour3 = '#246C71' # click
    colour4 = 'black'

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
        command=on_button_click,
        background=colour2, # Background color when not hovered
        foreground=colour4, # Text color
        activebackground=colour3, # Background color when hovered/clicked
        activeforeground=colour4, # Text color when hovered/clicked
        highlightthickness=2, # Thickness of the highlight border
        highlightbackground=colour2, # Highlight border color
        highlightcolor='WHITE', # Highlight color
        width=13,                      
        height=2,                         
        border=0,                       
        cursor='hand1',               
        font=('Arial', 16, 'bold')      
    )
    upload_button.pack(pady=10)
    
    root.mainloop()

    return file_path
library_ui()