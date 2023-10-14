

import numpy as np

import pandas as pd
import streamlit as st 
import json
import string
import pickle
import os


clf = pickle.load(open('nb_model.sav', 'rb'))
le={}
dir_list = os.listdir("Label_Encoders")
for i in dir_list:
     key=i.split('.')[0]
     temp="Label_Encoders/"+str(i)
     pkl_file = open(temp, 'rb')
     le[key] = pickle.load(pkl_file) 
     pkl_file.close()



def welcome():
    return "Welcome All"


def preproessing(place):
    print(place)  
    place_df=pd.DataFrame(columns=le.keys())
    # Python3 code to demonstrate working of
# Get all substrings of string
# Using list comprehension + string slicing
    first_letter=[]
    last_letter=[]
    first_two_letter=[]
    first_three_letter=[]
    first_four_letter=[]
    last_two_letter=[]
    last_three_letter=[]
    last_four_letter=[]
    last_letter_type=[]
    first_letter_type=[]
    second_letter=[]
    third_letter=[]
    fourth_letter=[]
    second_last_letter=[]
    third_last_letter=[]
    fourth_last_letter=[]
    Length=[]
    second_letter_type=[]
    third_letter_type=[]
    fourth_letter_type=[]
    second_last_letter_type=[]
    third_last_letter_type=[]
    fourth_last_letter_type=[]
    vowels=['a','e','i','o','u','A','E','I','O','U']

    test_str = place
    test_str =test_str.strip()
    length=len(test_str)
    first_letter.append(test_str[0])
    second_letter.append(test_str[1])
    second_last_letter.append(test_str[-2])
    try:
        third_letter.append(test_str[2])
        third_letter_type.append(['vowel' if test_str[2] in vowels else 'consanent'][0])
    except Exception as e:
        third_letter.append('')
        third_letter_type.append('')
    try:
        fourth_letter.append(test_str[3])
        fourth_letter_type.append(['vowel' if test_str[3] in vowels else 'consanent'][0])

    except Exception as e:
        fourth_letter.append('')
        fourth_letter_type.append('')
    try:
        third_last_letter.append(test_str[-3])
        third_last_letter_type.append(['vowel' if test_str[-3] in vowels else 'consanent'][0])
    except Exception as e:
        third_last_letter.append('')
        third_last_letter_type.append('')
    try:
        fourth_last_letter.append(test_str[-4])
        fourth_last_letter_type.append(['vowel' if test_str[-4] in vowels else 'consanent'][0])
    except Exception as e:
        fourth_last_letter.append('')
        fourth_last_letter_type.append('')
            
    
    last_letter.append(test_str[-1])
    first_two_letter.append(test_str[:2])
    last_two_letter.append(test_str[length - 2:])
    first_three_letter.append(test_str[:3])
    last_three_letter.append(test_str[length - 3:])
    first_four_letter.append(test_str[:4])
    last_four_letter.append(test_str[length - 4:])
    last_letter_type.append(['vowel' if test_str[-1] in vowels else 'consanent'][0])
    first_letter_type.append(['vowel' if test_str[0] in vowels else 'consanent'][0])
    second_last_letter_type.append(['vowel' if test_str[-2] in vowels else 'consanent'][0])
    second_letter_type.append(['vowel' if test_str[1] in vowels else 'consanent'][0])
    Length.append(length)


    place_df['First_Letter']=first_letter
    place_df['First_Two_Letter']=first_two_letter
    place_df['First_Three_Letter']=first_three_letter
    place_df['First_Four_Letter']=first_four_letter
    place_df['Last_Letter']=last_letter
    place_df['Last_Two_Letter']=last_two_letter
    place_df['Last_Three_Letter']=last_three_letter
    place_df['Last_Four_Letter']=last_four_letter
    place_df['First_Letter_Type']=first_letter_type
    place_df['Last_Letter_Type']=last_letter_type
    place_df['Length']=Length
    place_df["Second_Letter"]=second_letter
    place_df["Second_Letter_Type"]=second_letter_type
    place_df["Third_Letter"]=third_letter
    place_df["Third_Letter_Type"]=third_letter_type
    place_df["Fourth_Letter"]=fourth_letter
    place_df["Fourth_Letter_Type"]=fourth_letter_type
    place_df["Second_Last_Letter"]=second_last_letter
    place_df["Second_Last_Letter_Type"]=second_last_letter_type
    place_df["Third_Last_Letter"]=third_last_letter
    place_df["Third_Last_Letter_Type"]=third_last_letter_type
    place_df["Fourth_Last_Letter"]=fourth_last_letter
    place_df["Fourth_Last_Letter_Type"]=fourth_last_letter_type
    for key in le.keys():
        encoder = le[key]
        place_df[key] = encoder.transform(place_df[key])
    finale=place_df[["First_Letter","First_Two_Letter","First_Three_Letter","First_Four_Letter","Last_Letter","Last_Two_Letter","Last_Three_Letter","Last_Four_Letter","Second_Letter","Second_Letter_Type","Third_Letter","Third_Letter_Type","Fourth_Letter","Fourth_Letter_Type","Second_Last_Letter","Second_Last_Letter_Type","Third_Last_Letter","Third_Last_Letter_Type","Fourth_Last_Letter","Fourth_Last_Letter_Type","Length"]]
    return finale

def predict(place,place_df):
    prediction=place+(clf.predict(place_df))[0]
    return prediction

  
def main():
    st.title("Inhabitants Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Inhabitants ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    place = st.text_input("Place","Type Here")

    result=""
    if st.button("Predict"):
        place_df=preproessing(place)
        result=predict(place,place_df)
        print(result)
    
    st.success(result)

if __name__=='__main__':
    main()
    
    
    