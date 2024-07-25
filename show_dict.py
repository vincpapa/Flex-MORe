import pickle
from glob import glob

# Path to the pickle file
data = 'facebook_books'
pickle_file_path = f'results/{data}/performance/'
for file in glob.glob(f'{pickle_file_path}*'):

    with open(pickle_file_path, 'rb') as file:
        loaded_dict = pickle.load(file)
    print(loaded_dict)
