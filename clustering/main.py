import pandas as pd

def extract_ratings(filename):
    try:
        if filename == "":
            raise ValueError
        datafile = pd.read_csv(filename, header=None)
        records_list = datafile.to_dict("records")
        final_dict = {}
        for record in records_list:
            person_id = record.get(0) 
            
            if person_id:
                ratings_data = {}
                for key, value in record.items():
                    if key != 0:
                        try:
                            ratings_data[key] = int(value)
                        except (ValueError, TypeError):
                            ratings_data[key] = value
                
                final_dict[person_id] = ratings_data
        return final_dict
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
        print(f"{array[i]}")