#### save into json
import json
def save(path, data):    
    # Save the dictionary to a JSON file
    with open(path, 'w') as json_file:
        json.dump(data, json_file)

#### load from json
import json
def load(path):
    # Load the JSON file
    with open(path, 'r') as json_file:
        return json.load(json_file)