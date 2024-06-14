from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import EnumOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import HuggingFacePipeline
from langchain import HuggingFaceHub


import os
# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# nlp_pipeline = pipeline("text2text-generation", model=model_name)

# # Wrap the pipeline with HuggingFacePipeline
# llm = HuggingFacePipeline(pipeline=nlp_pipeline)
os.environ['HUGGINGFACEHUB_API_TOKEN']="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"
llm = HuggingFaceHub(repo_id="google/flan-t5-large",model_kwargs={"temperature":0.6,"max_length":50})

prompt_template = PromptTemplate(
    input_variables=["email_content"],
    template="""
    You are an AI trained to understand the intention of emails. Given the content of the email below, identify the intention as one of the following: [completed, pending, query].
    
    Email content:
    {email_content}
    
    What is the intention of this email? Respond with one word: completed, pending, or query.
    """
)
class IntentOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        # Simplified output parsing, assuming the model response will be one of the three words directly
        text = text.strip().lower()
        if text in ["completed", "pending", "query"]:
            return text
        return "unknown"
    
output_parser = IntentOutputParser()
llm_chain =LLMChain(llm=llm,prompt=prompt_template,output_parser=output_parser)

email_content_pending = """
Hi team,

I wanted to follow up on the status of the report that was supposed to be submitted last week. Has it been completed? If not, when can we expect it to be done?

Thanks,
John
"""

email_content_completed = """
Hi team,

Great news! The report that was due last week has been completed and submitted successfully. Thank you all for your hard work and dedication to getting this done on time.

Best regards,
John
"""

email_content_query = """
Hi team,

I have a question regarding the data analysis project. Can someone clarify the steps we need to follow to merge the datasets? I am also unsure about the deadline for the preliminary findings. Could someone provide more details on this?

Thanks,
John
"""





#displaying
import streamlit as st
st.title('Email Intent classifier')
input_text =st.text_input('Enter the Email')
if input_text:
    st.write(llm_chain.run(input_text))