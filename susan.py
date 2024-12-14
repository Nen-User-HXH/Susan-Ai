import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# OpenAI credentials
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# LangChain LLM
llm = OpenAI(temperature=0.5)

def generate_story(input_data):
    sections = [
        "A brief patient history and background, Personal information, gender, random name",
        "write the Initial assessment, the examination findings(arom,prom), and diagnosis, in addition differential diagnosis based on the case, special tests outcome",
        "Evidence-based treatment, 3 SMART goals linked to assessment findings",
        "Propose an intervention plan based on the given information, Detailed intervention plan with justification",
        "Expected outcomes and progress monitoring",
        "Reflection on reasoning and any necessary adjustments"
    ]
    
    story_text = ""
    for section in sections:
        prompt_section = f"Given a {{age}}-year-old patient with a background of {{patient_background}} seeking care in the domain of {{domain_selected}}, provide content for the section: {section}. Also, consider the keywords: {{text}}."

        story_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=list(input_data.keys()),
                template=prompt_section
            )
        )
        
        story_text += section + ":\n"
        story_text += story_chain.run(input_data) + "\n\n"
    
    return story_text

def app():
    st.title("PTCharlie the Physiotherapy Case Study Generator")

    st.write("""
Struggling with case studies? Let PTCharlie’s AI generate detailed, realistic patient scenarios for you. Just provide basic details - PTCharlie handles the rest.

PTCharlie leverages artificial intelligence to craft structured case studies with assessment findings, treatment plans, goals, and references. 
Customize cases to your needs for assignments, training, or professional development. Say goodbye to generic examples and let PTCharlie’s AI develop nuanced, evidence-based cases that bring therapy practices to life.
""")

    # Age Slider
    age = st.slider("Patient Age:", 0, 100, 30)

    with st.form(key='my_form'):
        patient_background = st.text_area("Enter some info for your case study:", placeholder="e.g., 'Jill, female, parkinson or total hip a year ago'")
        
        physio_domains = [
            "Select a physiotherapy domain",
            "Sports Physiotherapy",
            "Geriatric Physiotherapy",
            "Orthopedic Physiotherapy",
            "Pediatric Physiotherapy",
            "Neurological Physiotherapy",
            "Cardiovascular Physiotherapy"
        ]
        domain_selected = st.selectbox("Physiotherapy Specialization:", physio_domains)

        text = st.text_input("ADL Problem", placeholder="e.g., 'Difficulty with walking, transferring, balance, getting dressed, showering, toileting, and grooming'")

        if st.form_submit_button("Generate Story"):
            input_data = {
                "patient_background": patient_background,
                "domain_selected": domain_selected,
                "age": age,
                "text": text
            }
            
            with st.spinner('Generating story...'):
                story_text = generate_story(input_data)
            
            st.subheader("Generated Story:")
            st.markdown(story_text)

        if not text:
            st.info("Please complete the required inputs")

if __name__ == '__main__':
    app()
