import pickle
from glob import glob

# Path to the pickle file
data = 'facebook_books'
pickle_file_path = f'results/{data}/performance/'
for file in glob(f'{pickle_file_path}*'):
    with open(file, 'rb') as file_1:
        loaded_dict = pickle.load(file_1)
    print(loaded_dict)
