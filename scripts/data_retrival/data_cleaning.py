import pandas as pd
from dotenv import load_dotenv
from encoder import Encoder
import os

load_dotenv()

directory_path = os.getenv("DATA_PATH") 

class DataCleaner:
    def __init__(self, dataframe):
        self.df = dataframe

    def drop_duplicates(self):
        """
        Drop duplicate rows in the DataFrame.
        Convert any list-type columns to tuples to handle unhashable types.
        """
        # Identify columns with unhashable types
        for column in self.df.columns:
            if self.df[column].apply(lambda x: isinstance(x, list)).any():
                self.df[column] = self.df[column].apply(tuple)  # Convert lists to tuples
        
        self.df.drop_duplicates(inplace=True)
        #print("Duplicates dropped.")

    def check_missing_values(self):
        missing_values = self.df.isnull().sum()
        #print("Missing values per column:")
        #print(missing_values)
        return missing_values

    def drop_rows_with_missing_values(self, columns):
        self.df.dropna(subset=columns, inplace=True)
        print(f"Rows with missing values in {columns} have been dropped.")

    def fill_missing_values(self, columns, value):
        for column in columns:
            self.df[column] = self.df[column].fillna(value)
        #print(f"Missing values in {columns} have been filled with '{value}'.")

    def fixDates(self):
        # Ensure 'created_at' and 'closed_at' are in datetime format
        # Calculate the time taken to close issues (in days)
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
        self.df['closed_at'] = pd.to_datetime(self.df['closed_at'], errors='coerce')
        self.df['time_to_close'] = (self.df['closed_at'] - self.df['created_at']).dt.days   
        self.df = self.df.drop(columns=['created_at'])
        self.df = self.df.drop(columns=['closed_at'])

    def toLowerCase(self, columns):
        self.df["title"] = self.df["title"].str.lower()
        self.df["body"] = self.df["body"].str.lower()
    
    def clean_data(self):
        #print("Starting data cleaning...")
        self.drop_duplicates()
        self.check_missing_values()
        self.drop_rows_with_missing_values(['title', 'assignees'])
        self.fixDates()
        self.fill_missing_values(['labels', 'body'], '')
        self.df = self.df[self.df['assignees'].apply(lambda x: x != [] and x!= "[]")]
        encoder = Encoder()
        encoder.encode(df, "title")
        encoder.encode(df, "body")
        print("Data cleaning complete.")
        return self.df
    


for filename in os.listdir(str(directory_path+"raw/")):
    if filename.endswith(".csv"):
        file_path = os.path.join(str(directory_path+"raw/"), filename)
        df = pd.read_csv(file_path)
        cleaner = DataCleaner(df)
        df = cleaner.clean_data()
        df.to_csv(str(directory_path + "processed/"+ filename), index=False)
        print(f"File cleaned: {filename}")
