import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Load environment variables
load_dotenv(dotenv_path=r"C:\Users\chris\Susan-Ai\.env")

# Validate API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is missing. Please check your .env file.")

os.environ["OPENAI_API_KEY"] = openai_api_key

# LangChain LLM
llm = OpenAI(temperature=0.5)

# Retry logic for rate limits
@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=10))
def generate_with_retry(prompt, max_tokens=800):
    try:
        response = llm.generate([prompt], max_tokens=max_tokens)
        if not response or not hasattr(response, "generations") or not response.generations:
            raise ValueError("Invalid response from LLM. Ensure the prompt is formatted correctly.")
        return response.generations[0][0].text.strip()
    except Exception as e:
        raise e

def generate_section(prompt, max_tokens=800):
    return generate_with_retry(prompt, max_tokens=max_tokens)

def generate_story(input_data):
    sections = {
        "History": "A brief patient history and background, including personal information and random name.",
        "Initial Assessment": "Initial assessment findings, AROM, PROM, diagnosis, differential diagnosis, and special tests.",
        "SMART Goals": "Three SMART goals based on Jill's assessment findings.",
        "Intervention Plan": "A detailed intervention plan with justification.",
        "Expected Outcomes": "Expected outcomes and progress monitoring plan.",
        "Reflection": "Reflection on reasoning and any necessary adjustments to the treatment plan."
    }
    
    story_text = ""
    for section_name, section_prompt in sections.items():
        prompt = (
            f"Generate content for the section '{section_name}':\n"
            f"Patient Age: {input_data['age']}, Background: {input_data['patient_background']}, "
            f"Specialization: {input_data['domain_selected']}. Include: {section_prompt}"
        )
        section_content = generate_section(prompt, max_tokens=800)  # Generate section content
        story_text += f"### {section_name}\n{section_content}\n\n{'-' * 50}\n\n"
    
    return story_text

def app():
    st.title("Ai-PT nasuS the Physical Therapy Case Study Generator")

    st.write("""
Welcome Cohort 8!
Struggling with case studies? Let Ai-PT nasuS make it easy. Just give a few basic details, and it'll create detailed, realistic patient scenarios for you.

Whether you need assessment findings, treatment plans, therapy goals, or references, Ai-PT nasuS has you covered. Perfect for assignments, training, or professional growth, it crafts customized, evidence-based cases that feel real and practical. No more boring, generic examples— Ai-PT nasuS brings therapy to life!!
Nothing but w's ;)
""")

    # Age Slider
    age = st.slider("Patient Age:", 0, 100, 30)

    with st.form(key='my_form'):
        patient_background = st.text_area("Enter some info for your case study:", placeholder="e.g., 'Jill, female, Parkinson or total hip a year ago'")
        
        physio_domains = [
            "Select a physiotherapy domain",
            "Sports Physiotherapy",
            "Geriatric Physiotherapy",
            "Orthopedic Physiotherapy",
            "Pediatric Physiotherapy",
            "Neurological Physiotherapy",
            "Cardiovascular Physiotherapy"
        ]
        domain_selected = st.selectbox("Physiotherapy Specialization:", physio_domains, index=3)

        text = st.text_input("ADL Problem", placeholder="e.g., 'Difficulty with walking, transferring, balance, getting dressed, showering, toileting, and grooming'")

        if st.form_submit_button("Generate Story"):
            input_data = {
                "patient_background": patient_background,
                "domain_selected": domain_selected,
                "age": age,
                "text": text
            }
            
            try:
                with st.spinner('Generating story...'):
                    story_text = generate_story(input_data)
                st.subheader("Generated Story:")
                st.markdown(story_text)
            except Exception as e:
                st.error(f"An error occurred: {e}")

        if not text:
            st.info("Please complete the required inputs")

if __name__ == '__main__':
    app()
