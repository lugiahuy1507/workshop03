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
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch


os.environ["OPENAI_API_KEY"] = "sk-apxTWWhdEMvQr1PMX3Wy1Q"
os.environ["TAVILY_API_KEY"] = "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"

tavily_api = TavilySearch(
    max_results=3,
    topic="general",
    include_images=True,
)

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


def search_web(query: str) -> str:
    return tavily_api.run(query)


def create_vectors(vector_store, laptops, namespace):
    documents = []
    for i, laptop in enumerate(laptops):

        metadata = laptop.get("metadata", {})

        doc = Document(
            page_content=f"""
                {laptop["price_usd"]} to get {laptop["product_name"]} from {laptop["provider"]}
                with {laptop["specs"]}
                for {laptop["recommendation"]}
            """,
            metadata={
                "category": metadata.get("category", "NA"),
                "brand": metadata.get("brand", "NA"),
                "model": metadata.get("model", "NA"),
                "screen_size_inch": metadata.get("screen_size_inch", "NA"),
                "resolution": str(metadata.get("resolution", "NA")),
                "release_year": metadata.get("release_year", "NA"),
                "provider": metadata.get("provider", "NA"),
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


def format_web_info(web_infos):
    return "\n".join([
        f"Url:{web_info['url']} – Title:{web_info['title']} – - Content:{web_info['content']} – Score:${web_info['score']}"
        for web_info in web_infos
    ])


session_state = {
    "message_history": [],
    "retrieve_result_history": [],
    "last_query": None,
    "last_filter": None
}


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


class ChatState(TypedDict):
    conversation: Annotated[list, add_messages]
    retrieved_docs: list[Document]
    user_query: str
    last_filter: dict
    current_step: str
    web_search: str


def extract_filter_node(state: ChatState) -> ChatState:
    query = state["conversation"][-1].content
    filter_dict = extract_filter_from_query(query)
    state["user_query"] = query
    state["last_filter"] = filter_dict
    state["current_step"] = "retrieve"
    return state


def retrieve_node(state: ChatState) -> ChatState:
    vector_store = create_and_import_db("laptop-index")
    retriever = retrieve(vector_store, top_k=3, filter=state["last_filter"])
    docs = retriever.invoke(state["user_query"])
    state["retrieved_docs"] = docs
    state["current_step"] = "tavily_search"
    return state


# def tavily_search_node(state: ChatState) -> ChatState:
#     state["retrieved_docs"]

#     query = state["user_query"]
#     web_result = search_web(query)
#     # save result for use in generate_answer
#     state["web_search"] = web_result["results"][:3]  # type: ignore
#     state["current_step"] = "generate_answer"
#     return state

def should_trust_retrieved_docs(query, docs) -> bool:
    context = format_docs(docs)
    system_prompt = """
    You are a helpful assistant. A user has asked a question, and you have some documents retrieved from a database.
    Your job is to judge whether these documents are likely to be relevant to answering the user query.

    Respond with only "YES" if the documents clearly match the intent of the user query with high relevance (>= 0.5).
    Respond with "NO" if the content seems irrelevant, off-topic, or weakly related to the user query.
    """

    user_prompt = f"""
    User Query:
    {query}

    Retrieved Docs:
    {context}
    """

    result = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    print(f"result.content.upper(){result.content.upper()}")  # type: ignore

    return "YES" in result.content.upper()  # type: ignore

# Updated tavily_search_node with check


def tavily_search_node(state: ChatState) -> ChatState:
    query = state["user_query"]
    docs = state.get("retrieved_docs", [])

    # Check if retrieved docs are trustworthy
    if not docs or not should_trust_retrieved_docs(query, docs):
        web_result = search_web(query)
        state["web_search"] = web_result["results"][:3]  # type: ignore
    else:
        state["web_search"] = []  # type: ignore

    state["current_step"] = "generate_answer"
    return state


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
        
        Web search information from TavilySearch:
        {web_info}

        ---

        Instructions:
        - If the user's message is a general greeting (e.g., "hello", "hi", "how are you") or too vague, reply warmly and ask them for more details about what they're looking for in a laptop or cellphone.
        - If both local context and web info are available, blend them naturally in your answer.
        - If local context doesn't directly match the request, use insights from the **web_info** to complement and enrich your recommendations — but **don’t mention any lack of data**.
        - If web_info includes additional specs, reviews, or options, feel free to include them to enhance your response.
        - Clearly explain why each option is relevant.
        - End by confidently recommending **one device as the best fit**, using a friendly, helpful tone — as if you're giving advice to a friend.

        Use a friendly tone and speak like you're helping a friend shop.
    """)


def generate_answer_node(state: ChatState) -> ChatState:
    query = state["user_query"]
    docs = state["retrieved_docs"]
    web_info = state["web_search"]
    # Format docs and message history
    formatted_docs = format_docs(docs)
    print(f"formatted_docs: {formatted_docs}")
    formatted_history = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state["conversation"]
    )

    print(f"web_info {web_info}")

    rag_chain = ({"laptop_context": lambda _: formatted_docs,
                  "user_query": RunnablePassthrough(),
                  "retrieve_result_history": lambda _: format_docs(docs),
                  "web_info": lambda _: format_web_info(web_info),
                  "history": lambda _: formatted_history}) | qa_template | llm

    response = rag_chain.invoke(query).content
    state["conversation"].append(AIMessage(content=response))
    state["current_step"] = "done"
    return state


def route(state: ChatState) -> str:
    return state["current_step"]


def define_workflow():
    graph = StateGraph(ChatState)
    graph.add_node("extract_filter", extract_filter_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("tavily_search", tavily_search_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("extract_filter")
    graph.add_conditional_edges("extract_filter", route, ["retrieve"])
    graph.add_conditional_edges("retrieve", route, ["tavily_search"])
    graph.add_conditional_edges("tavily_search", route, ["generate_answer"])
    graph.add_edge("generate_answer", END)

    return graph.compile()
