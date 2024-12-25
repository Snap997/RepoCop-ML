import pandas as pd

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
        #print(f"Rows with missing values in {columns} have been dropped.")

    def fill_missing_values(self, columns, value):
        for column in columns:
            self.df[column].fillna(value, inplace=True)
        #print(f"Missing values in {columns} have been filled with '{value}'.")

    def clean_data(self):
        #print("Starting data cleaning...")
        self.drop_duplicates()
        self.check_missing_values()
        self.drop_rows_with_missing_values(['title', 'assignees'])
        self.fill_missing_values(['labels', 'body'], '')
        print("Data cleaning complete.")
        return self.df
