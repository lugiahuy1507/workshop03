import os
import json
from uuid import uuid4
from openai import OpenAI
from scipy.spatial import distance
from pinecone import Pinecone, ServerlessSpec
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage

os.environ["OPENAI_API_KEY"] = "sk-apxTWWhdEMvQr1PMX3Wy1Q"

llm = ChatOpenAI(
    model="GPT-4o-mini",
    temperature=0,
    api_key="sk-JkSdnK4_AkRYI-AxStZwYA",
    base_url="https://aiportalapi.stu-platform.live/jpe"
)

clientEmbedding = OpenAI(
    base_url="https://aiportalapi.stu-platform.live/jpe",
    api_key="sk-apxTWWhdEMvQr1PMX3Wy1Q",
)

clientGPT4 = OpenAI(base_url="https://aiportalapi.stu-platform.live/jpe",
                    api_key="sk-JkSdnK4_AkRYI-AxStZwYA")

embeddingModel = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://aiportalapi.stu-platform.live/jpe")


pc = Pinecone(
    api_key="pcsk_6yPRum_x9649kHnB9VbQ5FsWDCv6nN5J7Saj4bt6bwTStu6iJkC3BAxBbeUVkWvPkprSc")

message_history = [
    SystemMessage(
        content="You are a helpful assistant helping users pick laptops or phones.")
]


def create_vectors(vector_store, laptops, namespace):
    documents = []
    for i, laptop in enumerate(laptops):
        doc = Document(
            page_content=laptop["title"],
            metadata={
                "title": laptop["title"],
                "specs": [s.strip() for s in laptop["specs"].split(',')],
                "usage": laptop["usage"],
                "price": laptop["price"],
                "year": laptop["year"],
            }
        )
        documents.append(doc)

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)


def retrieve(vector_store, top_k, filter=None):
    results = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k,
            "filter": filter
        })

    return results


def extract_filter_from_query(user_query):
    system_prompt = """
    You are a smart assistant that extracts technical filters from user queries about laptop or cellphone recommendations.

    Your job is to extract:
    - A "max_price" or "min_price" if the user clearly mentions a number related to cost (in USD).
    - A list of "spec_keywords" which should include only technical hardware specifications, such as:
    - RAM size (e.g. "16GB RAM")
    - SSD/HDD size (e.g. "1TB SSD", "512GB SSD")
    - GPU or CPU mentions (e.g. "NVIDIA", "Intel i7", "AMD Ryzen")

    Do NOT include vague or subjective words like "strong", "rugged", "nice looking", "beautiful", "durable", etc.

    ### Output Rules:
    - Respond **only** with a valid JSON object containing keys that have actual values.
    - If no valid data can be extracted, respond with `{}`.
    - Do not include keys with `null` or `None` values.
    - Your response must be a **valid JSON object only**.

    ### Example:
    User Query: "I'm a developer, I wanna find a laptop under 1300$ for good coding and relaxing, NVIDIA GPU and about 16GB RAM and 1TB SSD, nice looking"
    Output:
        {
            "max_price": 1300,
            "spec_keywords": ["NVIDIA", "16GB RAM", "1TB SSD"]
        }

    User Query: "I love play game, I don't care about price, as long as it is really strong, rugged appearance is a plus"
    Output:
        {}
    """
    response = clientGPT4.chat.completions.create(
        model="GPT-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {user_query}"}
        ]
    )

    raw = response.choices[0].message.content
    filter_dict = json.loads(raw)  # type: ignore
    pinecone_filter = {}
    if "max_price" in filter_dict:
        pinecone_filter["price"] = {
            "$lte": filter_dict["max_price"]}
    if "min_price" in filter_dict:
        pinecone_filter.setdefault("price", {})[
            "$gte"] = filter_dict["min_price"]
    if "spec_keywords" in filter_dict:
        pinecone_filter.setdefault("specs", {})[
            "$in"] = filter_dict["spec_keywords"]

    return pinecone_filter


def format_docs(docs):
    return "\n".join([
        f"{doc.metadata['title']} – {', '.join(doc.metadata['specs'])} – for {doc.metadata['usage']} – ${doc.metadata['price']} – released in {doc.metadata['year']}"
        for doc in docs
    ])


session_state = {
    "message_history": [],
    "retrieve_result_history": [],
    "last_query": None,
    "last_filter": None
}


def question_answering(user_query, retrieve_result):
    print(f"retrieve_result: {retrieve_result}")
    qa_template = PromptTemplate.from_template("""
        You're a helpful and friendly AI assistant helping someone choose the best **cellphone or laptop**.

        First, read the user's request:
        "{user_query}"

        Previous conversation:
        {history}

        Previously recommended options:
        {retrieve_result_history}

        Here are the most relevant options I found:
        {laptop_context}

        ---

        Instructions:
        - If the user's request is unclear, just a greeting (e.g., "hello", "hi", "how are you"), or not specific to laptops/phones, respond in a warm and welcoming way and ask them to tell you more about what they're looking for.
        - If none of the laptops/cellphones in the context really match the user's request, be honest and say:
        > "I couldn’t find an exact match, but here are a few options that might still interest you!"
        - Otherwise, explain why each option could be suitable, and then **recommend one as the best fit**.

        Use a friendly tone and speak like you're helping a friend shop.
    """)

    last_two = session_state["message_history"][-2:] if len(
        session_state["message_history"]) >= 2 else session_state["message_history"]

    formatted_history = ""
    for msg in last_two:
        if isinstance(msg, HumanMessage):
            formatted_history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted_history += f"Assistant: {msg.content}\n"

    retrieved_docs = retrieve_result.invoke(user_query)

    formatted_retrieved = format_docs(retrieved_docs)
    formatted_history_retrieved = format_docs(
        session_state["retrieve_result_history"]) if session_state["retrieve_result_history"] else "None"

    print(f"formatted_history_retrieved: {formatted_history_retrieved}")

    rag_chain = ({"laptop_context": lambda _: formatted_retrieved,
                  "user_query": RunnablePassthrough(),
                  "retrieve_result_history": lambda _: formatted_history_retrieved,
                  "history": lambda _: formatted_history}) | qa_template | llm

    response = rag_chain.invoke(user_query).content

    session_state["retrieve_result_history"] = retrieved_docs
    session_state["message_history"].append(
        HumanMessage(content=user_query))  # type: ignore
    session_state["message_history"].append(
        AIMessage(content=response))  # type: ignore

    return response


def create_and_import_db(namespace):
    is_exist = True
    if namespace not in pc.list_indexes().names():
        pc.create_index(
            name=namespace,
            dimension=1536,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        is_exist = False

    index = pc.Index(namespace)

    vector_store = PineconeVectorStore(index=index, embedding=embeddingModel)

    if is_exist == False:
        with open("data.json", 'r') as f:
            data = json.load(f)
            create_vectors(vector_store, data, namespace)

    return vector_store
