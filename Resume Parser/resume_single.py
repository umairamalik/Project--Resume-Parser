from os import write
from pandas.core.indexes.base import Index
import spacy
import pickle
import random
import re
import streamlit as st
import pandas as pd
from pdfminer.high_level import extract_text


st.set_page_config(layout="centered")
st.title('RESUME PARSER')

train_data = pickle.load(open('train_data.pkl', 'rb'))


### Traning the NLP model
nlp = spacy.blank('en')

def train_model(train_data):
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last = True)
    
    for _, annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])
          
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes): # train only NER
        optimizer = nlp.begin_training()
        for itn in range(1):
            print('Starting iteration ' + str(itn))
            random.shuffle(train_data)
            losses = {}
            index = 0
            for text, annotations in train_data:
                try:
                    nlp.update([text], [annotations], drop =0.2, sgd = optimizer, losses = losses)
                except Exception as e:
                    pass  
        

# train_model(train_data)

## Saving the model
# nlp.to_disk('nlp_model')


## Loading the model
nlp_model = spacy.load('nlp_model')



def test_data():
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "json",'pdf' , 'docx'], accept_multiple_files=False)

    st.write(uploaded_file)
    ## For PDF
    text = extract_text(uploaded_file)
    # st.write(text)  
    if text == None:
        st.write('Please select the file for parsing : ')
    else :
          return text  
   
uploaded_file = st.file_uploader("Upload your file :", type=['pdf' , 'docx'], accept_multiple_files=False)
# st.write(uploaded_file)

# text = test_data()

# For Doc file

import docx2txt
def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None



if uploaded_file is not None:

    st.subheader('Parsed Data : ')

    st.text('')
    st.text('')
    # for x in uploaded_file:
    file_type = uploaded_file.name.split('.')
    print(file_type)

    if file_type[1] == 'pdf':
        text = extract_text(uploaded_file)
    elif file_type[1] == 'docx':
        text = (extract_text_from_docx(uploaded_file))


    text = text.replace('\n',' ') # remove \n in text
    doc = nlp_model(text)
    # st.write(text)

    details = {}

    PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
    EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
    LINKED_REG = re.compile('linkedin\.com/in/\w+[a-z0-9-]+\w+')
    GITHUB_LINK = re.compile('github\.com/\w+')
    EXP = re.compile(r'\d+[\.]\d+ years')

    phone_number = re.findall(PHONE_REG, text)
    email = re.findall(EMAIL_REG, text)
    linked_in = re.findall(LINKED_REG , text)
    github = match = re.findall(GITHUB_LINK,  text)
    exp = re.findall(EXP, text)

    try :
        if len(phone_number) != 0:
            details['PHONE NUMBER'] = phone_number[0]
            # print(f'PHONE NUMBER : ' , phone_number[0]) ## number 
        else:
            phone_number = "Not available"
            details['PHONE NUMBER'] = phone_number
            # print(f'PHONE NUMBER : ' , phone_number) ## number
            
        if len(email) !=0:
            details['EMAIL-ID'] = email[0]
            # print(f'EMAIL-ID     : ', email[0] ) ## Email
        else:
            email = "Not available"
            details['EMAIL-ID'] = email
            # print(f'EMAIL-ID     : ', email ) ## Email
            
            
        if linked_in !=0:
            details['LINKEDIN-LINK'] = 'https://www.'+linked_in[0]
            # print(f'LINKEDIN-LINK     : ', 'https://www.'+linked_in[0]) ## Linkedin
        else:
            linked_in = "Not available"
            details['LINKEDIN-LINK'] = linked_in
            # print(f'LINKEDIN-LINK     : ', linked_in) ## Linkedin
            
        if len(github) !=0 :
            details['GITHUB-LINK'] = 'https://'+github[0]
            # print(f'GITHUB-LINK : ' , github[0]) ## number 
        else:
            github = "Not available"
            details['GITHUB-LINK'] = github
            # print(f'GITHUB-LINK     : ', github)
        if len(exp) !=0 :
            details['YEARS OF EXPERIENCE'] = exp[0]
            print(f'YEARS OF EXPERIENCE : ' ,  exp[0] ) ## number 
        else:
            exp = None
            details['YEARS OF EXPERIENCE'] = exp
            print(f'YEARS OF EXPERIENCE    : ', exp)
    except IndexError:
        pass

    for ent in doc.ents:
        if ent.label_.upper() == 'SKILLS':
            details['SKILLS'] = ent.text
            # print(f'{ent.label_.upper():{20}}- {ent.text}')
        elif ent.label_.upper() == 'NAME':
            details['NAME'] = ent.text
            # print(f'{ent.label_.upper():{20}}- {ent.text}')
        elif ent.label_.upper() == 'DEGREE':
            details['DEGREE'] = ent.text
            # print(f'{ent.label_.upper():{20}}- {ent.text}')
        elif ent.label_.upper() == 'COLLEGE NAME':
            details['COLLEGE NAME'] = ent.text
            # print(f'{ent.label_.upper():{20}}- {ent.text}')
        elif ent.label_.upper() == 'GRADUATION YEAR':
            details['GRADUATION YEAR'] = ent.text
            # print(f'{ent.label_.upper():{20}}- {ent.text}') 
        elif ent.label_.upper() == 'DESIGNATION':
            details['DESIGNATION'] = ent.text
            # print(f'{ent.label_.upper():{20}}- {ent.text}')
        elif ent.label_.upper() == 'COMPANIES WORKED AT':
            details['COMPANIES WORKED AT'] = ent.text
            # print(f'{ent.label_.upper():{20}}- {ent.text}')
        elif ent.label_.upper() == 'LOCATION':
            details['LOCATION'] = ent.text
            # print(f'{ent.label_.upper():{20}}- {ent.text}')
        elif ent.label_.upper() == 'YEARS OF EXPERIENCE':
            details['YEARS OF EXPERIENCE'] = ent.text
            # print(f'{ent.label_.upper():{20}}- {ent.text}')

    for x , y in details.items():
        st.write(f'{x:{20}} : {y}')

    col = ['NAME','PHONE NUMBER','EMAIL-ID','LINKEDIN-LINK','GITHUB-LINK','YEARS OF EXPERIENCE',
       'LOCATION','DEGREE','COLLEGE NAME','GRADUATION YEAR','COMPANIES WORKED AT','DESIGNATION','SKILLS']
    new_dict = {}
    for x in col:
        if x in details.keys():
            new_dict[x] = details[x]
        else:
            new_dict[x]= None

    # df = pd.DataFrame([details.values()], columns = [list(details.keys())])

    df = pd.DataFrame([new_dict], columns =col )

    # @st.cache
    def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv(index= False).encode('utf-8')

    st.text('')
    st.text('')
    csv = convert_df(df)
    st.download_button(label="Download data as CSV", data = csv ,file_name='Parsed_data.csv', mime='text/csv',)


else:
    st.write('Please select the file for parsing : ')

st.text('')
st.text('')
st.text('')
st.text('')
# -- Notes on whitening
# st.write('About The App :')
with st.expander("About The App :"):
    st.markdown("""
 * This Resume Parser can extract Name, Phone,Email, Designation, Degree, Skills and University, Location, 
companies worked with and duration details. 
 * We are working on adding other entities and to increase the accuracy of the model.

 * Author : Rahul Ghuge 
 * Mail : itengineer.rghuge@gmail.com
 
 See Source Code:
 * [See Source Code](https://github.com/RahulGhu)
""")
  







