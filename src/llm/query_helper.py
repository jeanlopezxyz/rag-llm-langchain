from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os
from langchain.chains import RetrievalQA
from vector_db.db_provider_factory import FAISS, DBFactory

############################
# LLM chain implementation #
############################

db_factory = DBFactory()

prompt_template = """
### [INST]
Instructions:
- You are a helpful assistant in writing a project proposal for products owned by Red Hat for the company name provided in the question below.
- Your job is to look at the  question and create the project proposal addressed to the company mentioned in question.
- Base your answer on the provided context and question and not on prior knowledge.
- The proposal should contain headings and sub-headings and each heading and sub-heading should be in bold.
- Generate the project proposal in markdown language.
- Each section in the proposal should contain only three items.
- Proposal should be minimum of 500 lines.

Here is context to help:
{context}

### QUESTION:
Question: {question}

[/INST]
"""

QA_CHAIN_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_qa_chain(llm):
    try:
        type = os.getenv("DB_TYPE") if os.getenv("DB_TYPE") else "REDIS"
        if type is None:
            raise ValueError("DB_TYPE is not specified")
        print(f"Retriever DB: {type}")
        retriever = db_factory.get_retriever(type)
    except Exception as e:
        print(e)
        print(
            f"{type} server is unavailable. Project proposal will be generated without RAG content."
        )
        retriever = db_factory.get_retriever(FAISS)

    return RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
    )
