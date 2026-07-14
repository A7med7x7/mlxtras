import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np

class DataValidationUtils:

    @staticmethod
    def format_data_types(
        data: pd.DataFrame,
        floats: list = None,
        integers: list = None,
        timestamps: list = None,
    ) -> pd.DataFrame:
        floats = [] if floats is None else floats
        floats = [feature for feature in data.select_dtypes(include=['float'])]
        integers = [] if integers is None else integers
        integers = [feature for feature in data.select_dtypes(include=['int'])] 
        timestamps = [] if timestamps is None else timestamps

        data[floats] = data[floats].apply(pd.to_numeric, errors="coerce")
        data[timestamps] = data[timestamps].apply(pd.to_datetime, errors="coerce")

        # formatting integers
        if integers:
            for col in integers:
                if data[col].dtype != "str":
                    data[col] = data[col].astype(str)
                data[col] = data[col].str.replace("[^\d]", "", regex=True)
                data[col] = data[col].str.strip()
                data[col] = data[col].replace("", -1)
                data[col] = data[col].astype(np.int64)

        return data

def enviroment_setup(project_name:str):
    
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

    list_of_files = [
        f"{project_name}/data",
        f"{project_name}/notebooks",
        f"{project_name}/src/data",  
        f"{project_name}/src/models",
        f"{project_name}/src/evaluation",
        f"{project_name}/src/inference",
        f"{project_name}/src/utils",
        f"{project_name}/src/tests",
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]

    for filepath in list_of_files: 
        filepath = Path(filepath) #it solves the / slash and \ slash problem internally
        filedir, filename = os.path.split(filepath)
        if filedir != "":
            os.makedirs(filedir,exist_ok=True)
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0): 
            with open (filepath, "w") as f:
                pass
        else:
            print(f"file path is already present at :{filepath}")
            