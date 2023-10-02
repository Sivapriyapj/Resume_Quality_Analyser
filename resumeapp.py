import streamlit as st

import pickle
import re
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
import pandas as pd
import os
from tqdm import tqdm
from spacy.tokens import DocBin


# import nltk


# nltk.download('punkt')
# nltk.download('stopwords')

rfc = pickle.load(open('rfc.pkl','rb'))
tfidf = pickle.load(open('tfif.pkl','rb'))

def cleanmytext(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText





# WEBPAGE
def main():
    st.title("RESUME QUALITY ANALYSER")
    st.image("image2.jpg")
 

    uploaded_input_file = st.file_uploader('Upload Resume here',type=['pdf','txt'])

    if uploaded_input_file is not None:
        try:
            res_byt = uploaded_input_file.read()
            res_txt = res_byt.decode('utf-8')
        except UnicodeDecodeError:
            res_txt = res_byt.decode('latin-1')
        

        clean_res = cleanmytext(res_txt)
        cleaned_resume = tfidf.transform([clean_res])
        pred_id = rfc.predict(cleaned_resume)[0]

        # code to extract entitied NER

        nlp1 = spacy.load(r"output/model-best") #load the best model
        doc = nlp1(clean_res)

        
        


        categories = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",




        
}   


        


        html = """
<style>
    @keyframes slidein {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    .green-box {
        background-color: green;
        padding: 10px;
        color: white;
        display: inline-block;
        animation: slidein 3s ease-in-out;
    }
</style>
"""

    


        # code for NER skillset list to be displayed


        # list_skills = ["prompt engineering","OpenAI API","Few shot learning"]        
        resume_category = categories.get(pred_id,"Unknown")
        st.write("This resume is suitable for :")
        st.write(f'<div style="background-color: green; padding: 10px; color: white; display: inline-block">{resume_category}</div>', unsafe_allow_html=True)
        st.write("The key skills present in the resume are :")
         # for element in list_skills:
        #     st.markdown(f'<div style="background-color: green; padding: 10px; color: white; display: inline-block">{element}</div>', unsafe_allow_html=True)
        #     st.write("")
        
        # Display each element in a green pebble-like box
        list_skills = list(doc.ents)
        for element in list_skills:
            print(element)
            print("------------------")

      

 
       
        for element in list_skills:
            html += f'<div class="green-box">{element}</div><br><br>'
            st.write("")
        st.markdown(html, unsafe_allow_html=True)
        
            

        
# python main
if __name__ == "__main__":
    main()
