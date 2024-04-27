import tkinter as tk
import customtkinter
from tkinter import filedialog, messagebox
import CTkTable

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import classification as cls_
import regression

class MlUI:

    def __init__(self):
        self.colour1 = '#161616'
        self.colour2 = '#05d7ff' # Default button background color
        self.colour3 = '#246C71' # Button background color when clicked
        self.colour4 = 'black' # Text color

        self.root = None
        self.left_container = None
        self.scale_data_combobox = None

        # Upload file button
        self.file_path = None
        self.df_columns = None
        self.target_combobox = None

        # Train model button
        self.models_output_df = None
        self.checkboxes = None
        # self.combobox = None

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

    def train_button_click(self):
        selected_models = [option.strip() for option, checkbox in self.checkboxes.items() if checkbox.get()]
        target_col = self.target_combobox.get()

        if self.combobox.get() == 'Classification':
            df = pd.read_csv(self.file_path)
            X = df.drop(columns=[target_col]).values
            y = df[target_col]

            if self.scale_data_combobox.get() == 'StandardScaler':
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                
            elif self.scale_data_combobox.get() == 'MinMaxScaler':
                scaler = MinMaxScaler()
                X = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=.2, random_state=42
            )       
            self.models_output_df = cls_.classification_models(X_train, X_test, y_train, y_test, selected_models)[0]
            
            self.show_model_output()

        elif self.combobox.get() == 'Regression':
            df = pd.read_csv(self.file_path)
            X = df.drop(columns=[target_col]).values
            y = df[target_col]

            if self.scale_data_combobox.get() == 'StandardScaler':
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            elif self.scale_data_combobox.get() == 'MinMaxScaler':
                scaler = MinMaxScaler()
                X = scaler.fit_transform(X)
                
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=.2, random_state=42
            )       

            self.models_output_df = regression.regression_models(X_train, X_test, y_train, y_test)[0]

            self.show_model_output()

        else:
            messagebox.showinfo(title='Not found error', message="Model type not selected")

        return None

    @staticmethod
    def truncate_number(number):
        number *= 10000
        number = int(number)
        return number / 10000

    def show_model_output(self):
        values = []

        values.append(self.models_output_df.columns)
        truncated_numbers = np.zeros((self.models_output_df.values.shape[0], self.models_output_df.values.shape[1]), dtype=object)
        truncated_numbers = np.zeros((self.models_output_df.values.shape[0], self.models_output_df.values.shape[1]), dtype=object)
        
        for i in range(self.models_output_df.values.shape[0]):
            truncated_numbers[i][0] = self.models_output_df.values[i][0]
            for j in range(1, self.models_output_df.values.shape[1]):
                truncated_numbers[i][j] = self.truncate_number(self.models_output_df.values[i][j])

        values.extend(truncated_numbers)
        num_rows = len(self.models_output_df) + 1
        num_columns = len(self.models_output_df.columns)

        table_container = customtkinter.CTkFrame(self.root, corner_radius=15)
        table_container.place(relx=0.3, rely=0.05, relwidth=0.68, relheight=0.9)

        table = CTkTable.CTkTable(
            master=table_container, row=num_rows, column=num_columns, 
            values=values, header_color="#204261")
        table.pack(expand=True, fill="both", padx=20, pady=20)

    # def clean_button_click(self):
    #     self.on_close_click()
    #     self.library_ui()

    def on_close_click(self):
        self.root.destroy()

    def get_selected_ml_type(self):
        return self.combobox.get()
        
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
        self.combobox = customtkinter.CTkComboBox(
            master = self.left_container,
            width = 170,
            values = ['Classification', 'Regression'],
            variable  =model_type_var
        )
        self.combobox.pack(padx=20, pady=10)

        # Scale data combo box
        scale_data_var = customtkinter.StringVar(value="Scale Data")
        self.scale_data_combobox = customtkinter.CTkComboBox(
            master = self.left_container,
            width = 170,
            values = ["StandardScaler", 'MinMaxScaler'],
            variable = scale_data_var
        )
        self.scale_data_combobox.pack(padx=20, pady=10)

        #! ---------------- 
        def toggle_dropdown():
            dropdown_frame.pack_forget() if dropdown_frame.winfo_ismapped() else dropdown_frame.pack()

        dropdown_frame = customtkinter.CTkFrame(self.left_container, width=50, height=50)
        dropdown_frame.pack(padx=10, pady=10)
        dropdown_frame.pack_forget()
        
        selected_label = customtkinter.CTkLabel(self.left_container, text="")
        selected_label.pack()

        self.checkboxes = {}
        ml_models = []

        if self.get_selected_ml_type() == 'Classification':
            ml_models = [
                'LogisticRegression',
                'KNeighborsClassifier',
                'DecisionTreeClassifier',
                'SVC',
                'RandomForestClassifier',
                'GradientBoostingClassifier',
                'XGBClassifier',
                'XGBRFClassifier',
                'LGBMClassifier',
                'CatBoostClassifier',
                'GaussianNB',
                'MLPClassifier',
            ]
        
        elif self.get_selected_ml_type() == 'Regression':
            ml_models = [
                'LinearRegression',
                'Ridge',
                'Lasso',
                'ElasticNet',
                'DecisionTreeRegressor',
                'RandomForestRegressor',
                'GradientBoostingRegressor',
                'XGBRegressor',
                'XGBRFRegressor',
                'LGBMRegressor',
                'CatBoostRegressor',
                'MLPRegressor',
            ]

        max_len_model = max([len(model) for model in ml_models])

        for i in range(len(ml_models)):
            ml_models[i] = ml_models[i] + " " * (max_len_model - len(ml_models[i]))

        for option in ml_models:
            checkbox = customtkinter.CTkCheckBox(dropdown_frame, text=option)
            checkbox.pack()
            self.checkboxes[option] = checkbox

        dropdown_button = customtkinter.CTkButton(self.left_container, text="▼", command=toggle_dropdown)
        dropdown_button.pack()

        def update_selected_label():
            selected = [option for option, checkbox in self.checkboxes.items() if checkbox.get()]
            selected_text = ", ".join(selected) if selected else "No selection"
            selected_label.configure(text=selected_text)
        
        for checkbox in self.checkboxes.values():
            checkbox.configure(command=update_selected_label)
        
        update_selected_label()

        #! ----------------

        train_button = tk.Button(
            self.left_container,
            text = 'Train Models',
            command = self.train_button_click,
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

if __name__ == '__main__':
    MlUI().library_ui()