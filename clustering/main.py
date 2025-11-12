import pandas as pd

def extract_ratings(filename):
    try:
        if filename == "":
            raise ValueError
        datafile = pd.read_csv(filename, header=None)
        data_array = datafile.to_dict("records")
        return data_array
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return []
    except pd.errors.EmptyDataError:
        print(f"The file {filename} is empty.")
        return []
    except ValueError:
        print(f"Filename was not specified")
        return []

if __name__ == "__main__":
    array = extract_ratings("ratings.csv")
    for i in array:
        print(f"{i}")