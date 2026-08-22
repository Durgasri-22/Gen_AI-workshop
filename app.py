import os
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_groq import chatGroq

os.environ["GROQ_API_KEY"] = "Your_Groq_API_Key"
llm=chatGroq(model="llama-3.1-8b-instant", temperature=0)
client = chromadb.Client()
collection = client.get_or_create_collection("job_collection")
collection.add(
    documents=[
        "Machine learning and python AI solutions",
        "WordPress webdevelopment services",
        "Mangento e-commerce platform development"
    
    ],
    metadatas=[
        {"links":"https://example.com/ml-python-portfolio"},
        {"links":"https://example.com/wordpress-portfolio"},
        {"links":"https://example.com/mangento-portfolio"}
    ],
    ids=['doc1', 'doc2', 'doc3']
)

json_res=[
    {
        "title": "AI Engineer",
        "Skills": "Python, Machine Learning, NLP, APIs",
        "description": "Hiring an Ai engineer to build ML models and NLP systems"
    }
]
job = json_res[0]
raw_links=collection.query(
    query_texts=[job["Skills"]],
    n_results=2
).get("metadatas",[])

clean_links =[item["links"] for group in raw_links for item in group]
unique_links= list(set(clean_links))

prompt_email = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}
    
    ### INSTRUCTION:
    You are Mohan, a business development executive at AtliQ.
    AtliQ is an AI & Software Consulting company dedicated to facilitating
    the seamless integration of business processes through automated tools. 
    Over our experience, we have empowered numerous enterprises 
    with tailored solutions, fostering scalability, 
    process optimization, cost reduction, and heightened overall efficiency. 
    Your job is to write a cold email to the client regarding the 
    job mentioned above describing the capability of AtliQ 
    in fulfilling their needs.
    Also add the most relevant ones from the following links
    to showcase Atliq's portfolio: {link_list}
    Remember you are Mohan, BDE at AtliQ. 
    Do not provide a preamble.
    ### EMAIL (NO PREAMBLE):
    """
)
chain_email = prompt_email | llm
res = chain_email.invoke({"job_description": str(job), "link_list": unique_links})

print(res.content)



