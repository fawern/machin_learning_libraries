import tkinter as tk
import customtkinter
from tkinter import filedialog, messagebox
import CTkTable

import pandas as pd 
from sklearn.model_selection import train_test_split
import classification as cls_
import regression

class Library_UI:
    def __init__(self):
        self.colour1 = '#161616'
        self.colour2 = '#05d7ff' # Default button background color
        self.colour3 = '#246C71' # Button background color when clicked
        self.colour4 = 'black' # Text color

        self.root = None
        self.left_container = None
        self.target_combobox = None

        # Upload file button
        self.file_path = None
        self.df_columns = None
        self.target_combobox = None

        # Train model button
        self.models_output_list = None

    def upload_button_click(self):
        self.file_path = filedialog.askopenfilename(title='Select a CSV file !!!')

        if self.file_path:
            if self.file_path.split('.')[-1] != 'csv':
                messagebox.showinfo(title='File type error', message="Please select a CSV file")
            else:
                self.df_columns = list(pd.read_csv(self.file_path).columns)
                self.target_combobox.configure(values=self.df_columns)
        else:
            messagebox.showinfo(title='Not found error', message="No file selected")

    def train_button_click(self, selected_model):
        target_col = self.target_combobox.get()

        if selected_model == 'Classification':
            df = pd.read_csv(self.file_path)
            X = df.drop(columns=[target_col]).values
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=.2, random_state=42
            )       

            self.models_output_list = cls_.classification_models(X_train, X_test, y_train, y_test)[0]

            self.show_model_output()

        elif selected_model == 'Regression':
            df = pd.read_csv(self.file_path)
            X = df.drop(columns=[target_col]).values
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=.2, random_state=42
            )       

            models_output_list = regression.regression_models(X_train, X_test, y_train, y_test)[0]

            self.show_model_output()

        else:
            messagebox.showinfo(title='Not found error', message="Model type not selected")

        return None

    def truncate_number(number):
        number *= 1000
        number = int(number)
        return number / 1000

    def show_model_output(self):
        value = [list(self.models_output_list.columns)]
        
        for row in self.models_output_list.values:
            row[1:] = list(map(truncate_number, row[1:]))
            value.append(row)

        table_container = customtkinter.CTkFrame(self.root, corner_radius=15)
        table_container.place(relx=0.3, rely=0.05, relwidth=0.68, relheight=0.9)

        table = CTkTable.CTkTable(
            master=table_container, row=len(self.models_output_list) + 1, column=len(self.models_output_list.columns), 
            values=value, header_color="#204261")
        table.pack(expand=True, fill="both", padx=20, pady=20)

    def on_close_click(self):
        self.root.destroy()

    def library_ui(self):    
        self.root = customtkinter.CTk()
        self.root.title("Machine Learnings Library")
        self.root.resizable(True, True) 
        self.root.geometry("1400x600")  # Width x Height
        self.root.configure(background=self.colour1)

        self.left_container = customtkinter.CTkFrame(self.root, corner_radius=15)
        self.left_container.place(relx=0.025, rely=0.05, relwidth=0.25, relheight=0.9)

        upload_button = tk.Button(
            self.left_container,
            text = 'Upload .csv',
            command = self.upload_button_click,
            background = self.colour2, # Background color when not hovered
            foreground = self.colour4, # Text color
            activebackground = self.colour3, # Background color when hovered/clicked
            activeforeground = self.colour4, # Text color when hovered/clicked
            highlightthickness = 2, # Thickness of the highlight border
            highlightbackground = self.colour2, # Highlight border color2
            highlightcolor = 'WHITE', # Highlight color
            width = 9,
            height = 1,
            border = 0,
            cursor = 'hand1',
            font = ('Arial', 16, 'bold')
        )
        upload_button.pack(pady=10)

        output_col_var = customtkinter.StringVar(value="Target Column")
        self.target_combobox = customtkinter.CTkComboBox(
            master = self.left_container,
            width = 170,
            variable = output_col_var,
            values = ['']
        )
        self.target_combobox.pack(padx=20, pady=10)

        model_type_var = customtkinter.StringVar(value="Machine Learning Type")

        combobox = customtkinter.CTkComboBox(
            master = self.left_container,
            width = 170,     
            values = ['Classification', 'Regression'],
            variable  =model_type_var
        )
        combobox.pack(padx=20, pady=10)

        train_button = tk.Button(
            self.left_container,
            text = 'Train Models',
            command = lambda : self.train_button_click(combobox.get()),
            background = self.colour2, # Background color when not hovered
            foreground = self.colour4, # Text color
            activebackground = self.colour3, # Background color when hovered/clicked
            activeforeground = self.colour4, # Text color when hovered/clicked
            highlightthickness = 2, # Thickness of the highlight border
            highlightbackground = self.colour2, # Highlight border color
            highlightcolor = 'WHITE', # Highlight color
            width = 10,
            height = 1,
            border = 0,
            cursor = 'hand1',
            font = ('Arial', 16, 'bold')
        )
        train_button.pack(pady=10)

        close_button = tk.Button(
            self.left_container,
            text = 'Close',
            command = self.on_close_click,
            background = '#a4072c', # gozuken renk
            foreground = 'white',
            activebackground = '#94042b', # tiklayinca 
            activeforeground = self.colour4,
            highlightthickness = 2,
            highlightbackground = self.colour2,
            highlightcolor = 'WHITE',
            width = 8,
            height = 1,
            border = 0,
            cursor = 'hand1',
            font = ('Arial', 16, 'bold')
        )
        close_button.pack(side='bottom', pady=50)

        self.root.mainloop()

Library_UI().library_ui()