import re
from langchain_community.document_loaders import PyPDFLoader
import pymupdf
import pickle

file_path = ('data.pdf')
loader = PyPDFLoader(file_path)
pages = loader.load_and_split() #text_splitter=text_splitter)

species_name_pattern = re.compile(r'species name', re.IGNORECASE)
species_list = []
for i in pages:
    if re.search(species_name_pattern, i.page_content):
        species_list.append(i.metadata['page'])

species_list = sorted(list(set(species_list)))

name_output_list = []

doc = pymupdf.open(file_path)

for i in species_list:
    page = doc[i]
    tabs = page.find_tables()
    num = len(tabs.tables)
    print(f"{num} table(s) on {page}")
    
    for j in range(0, num):
        tab = tabs[j]
        header = tab.header #list
        header_list = [str(item) for item in header.names]
        header_string = ' '.join(header_list)
        
        if re.search(species_name_pattern, header_string):
            df = tab.to_pandas()
            # Normalize the column names to lowercase
            df.columns = df.columns.str.lower()
            names = df['species name'].values.tolist()
            tmp = [i for i in names if i is not None]
            tmp1 = [string.replace('\n', ' ') for string in tmp]
            name_output_list.extend(tmp1)


with open('name.pkl', 'wb') as f:
    pickle.dump(name_output_list, f)