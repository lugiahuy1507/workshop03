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


# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#set logging pattern timestamp file:line message
formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
# Set logging to file
file_handler = logging.FileHandler('app.log')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


os.environ["OPENAI_API_KEY"] = "sk-apxTWWhdEMvQr1PMX3Wy1Q"
os.environ["TAVILY_API_KEY"] = "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"

tavily_api = TavilySearch()

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
                    api_key="sk-WTGn3uqXmRn3Heq0-hbF4Q")

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

def extract_filter_from_search_result(user_query, search_result):
    """
    Extract relevant keywords from web search results.
    """
    system_prompt = """
            You are a highly accurate and context-aware keyword extraction agent. Your goal is to extract the most relevant and meaningful keywords from a given text. 
            Organize them by category where appropriate, and avoid generic or uninformative terms. Follow these guidelines:

            ### 🛠️ Instructions:
            1. **Focus on core entities**, product names, technical specs, models, and formats.
            2. **Group keywords** into logical categories (e.g., Products, Features, Specs).
            3. **Remove duplicates**, filler words, and uninformative words like "image", "phone", "video", "review", unless contextually significant.
            4. **Preserve important numerical details** (e.g., resolutions, frame rates, version numbers).
            5. **Output in clean, bullet-pointed format**, sorted by category with clear headers.
            6. Use original casing for brand/product names (e.g., Apple iPhone 17 Pro Max).
            7. Never add keywords that are not explicitly present in the input.

            ---

            ### ✅ Output Format Example:

            **📱 Product Keywords**  
            - Apple iPhone 17 Pro Max  
            - iPhone 17 Air  
            - iPhone 16 Pro Max
            - ...

            **📹 Video Specifications**  
            - 4K@24/25/30/60/100/120fps  
            - 1080p@25/30/60/120/240fps  
            - 10-bit HDR  
            - ...

            **📦 Other Attributes**  
            - Dolby Vision HDR  
            - ProRes  
            - Stereo sound recording  
            - 3D (spatial) video/audio
    """
    
    few_shot_examples = [
        {
            "role": "user", "content": """extract key word from this text
                    Apple iPhone 17 Pro Max - Full phone specifications | Video | 4K@24/25/30/60/100/120fps, 1080p@25/30/60/120/240fps, 10-bit HDR, Dolby Vision HDR (up to 60fps), ProRes, 3D (spatial) video/audio, stereo sound rec. 
                    Apple iPhone 17 Pro Max - user opinions and reviews *   Image 4: Phone Apple iPhone 16 Pro Max *   Image 6: Phone Apple iPhone 17 Air *   Image 7: Phone Apple iPhone 16 Pro *   Image 8: Phone Apple iPhone 15 Pro Max *   Image 10: Phone Apple iPhone 17 Pro Max *   Image 11: Phone Apple iPhone 16 Pro Max *   Image 12: Phone Apple iPhone 13 *   Image 13: Phone Apple iPhone 11 *   Image 14: Phone Apple iPhone 16 *   Image 15: Phone Apple iPhone 15
"""
        },
        {
            "role": "assistant", "content": """ Product Keywords
            Apple
            iPhone
            iPhone 17 Pro Max
            iPhone 17 Air
            iPhone 16 Pro Max
            iPhone 16 Pro
            iPhone 15 Pro Max
            iPhone 13
            iPhone 11
            iPhone 16
            iPhone 15"""
        }
    ]            


    user_prompt = f"""
        User query: {user_query}
        Extract relevant keywords from the following web search results:
        {search_result}
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(few_shot_examples)
    messages.append({"role": "user", "content": user_prompt})
    

    response = clientGPT4.chat.completions.create(
        model="GPT-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()


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


def question_answering(user_query, retrieve_result):
    logger.info(f"retrieve_result: {retrieve_result}")
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

    logger.info(f"formatted_history_retrieved: {formatted_history_retrieved}")

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

def sematic_rerank(query: str, docs: list[Document], threshold: float = 0.5):
    """
    Rerank the retrieved documents based on their relevance to the user query.
    """
    embeddings = embeddingModel.embed_documents([doc.page_content for doc in docs])
    query_embedding = embeddingModel.embed_query(query)
    distances = distance.cdist([query_embedding], embeddings, "cosine")[0]
    # convert distance to percentage confidence
    confidence_scores = [1 - dist for dist in distances]
    sorted_docs = sorted(zip(docs, confidence_scores), key=lambda x: x[1])
    true_relevant = [doc[0] for doc in sorted_docs if doc[1] > threshold]
    logger.info(f"Confidence scores: {confidence_scores}")
    logger.info(f"Reranked {len(true_relevant)} documents with confidence above {threshold}")
    return true_relevant

class ChatState(TypedDict):
    conversation: Annotated[list, add_messages]
    retrieved_docs: list[Document]
    reranked_docs: list[Document]
    user_query: str
    last_filter: dict
    current_step: str
    web_search: str
    retrieve_keywords: str

def extract_filter_node(state: ChatState) -> ChatState:
    logger.info(f"Enter node 'extract_filter' with state: {state}")
    try:
        query = state["conversation"][-1].content if state["conversation"] else ""
        filter_dict = extract_filter_from_query(query)
        state["user_query"] = query
        state["retrieve_keywords"] = query
        state["last_filter"] = filter_dict
        state["web_search"] = ""
        state["retrieved_docs"] = ""
        state["reranked_docs"] = ""
        state["current_step"] = "retrieve"
    except Exception as e:
        logger.error(f"Error occurred while extracting filter: {e}")
    return state


def retrieve_node(state: ChatState) -> ChatState:
    logger.info(f"Enter node 'retrieve' with state: {state}")
    vector_store = create_and_import_db("laptop-index")
    retriever = retrieve(vector_store, top_k=3, filter=state["last_filter"])
    docs = retriever.invoke(state["user_query"])
    state["retrieved_docs"] = docs
    state["current_step"] = "rerank"
    
    return state


def tavily_search_node(state: ChatState) -> ChatState:
    logger.info(f"Enter node 'tavily_search' with state: {state}")
    query = state["user_query"]
    web_result = search_web(query)
    # save result for use in generate_answer
    try:
        logger.info(f"Web search results: {web_result}")
        state["web_search"] = web_result["results"][:2]  # type: ignore
    except Exception as e:
        logger.error(f"Error occurred while searching the web: {e}")
    state["current_step"] = "extract_search_result"
    return state


def rerank_node(state: ChatState) -> ChatState:
    logger.info(f"Enter node 'rerank' with state: {state}")
    if not state["retrieved_docs"]:
        state["reranked_docs"] = []
    else:
        state["reranked_docs"] = sematic_rerank(
            state["user_query"], state["retrieved_docs"], threshold=0.7)

    if len(state["reranked_docs"]) == 0:
        state["current_step"] = "tavily_search"
    else:
        state["current_step"] = "generate_answer"
    return state

def extract_search_result_node(state: ChatState) -> ChatState:
    logger.info(f"Enter node 'extract_search_result' with state: {state}")
    """
    Extract relevant keywords from web search results.
    """
    state["retrieve_keywords"] = extract_filter_from_search_result(
        state["user_query"], state["web_search"])
    state["current_step"] = "retrieve"
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
        - Introduce to user about our product (at <Previously recommended options>) to help them make a decision.

        Use a friendly tone and speak like you're helping a friend shop.
    """)

def generate_answer_node(state: ChatState) -> ChatState:
    logger.info(f"Enter node 'generate_answer' with state: {state}")
    query = state["user_query"]
    docs = state["reranked_docs"] if state["reranked_docs"] else state["retrieved_docs"]
    web_info = state["web_search"]
    # Format docs and message history
    formatted_docs = format_docs(docs)
    formatted_history = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state["conversation"]
    )

    logger.info(f"web_info {format_web_info(web_info)}")

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
    logger.info("================== START WORKFLOW DEFINITION ===================")
    graph = StateGraph(ChatState)
    graph.add_node("extract_filter", extract_filter_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("tavily_search", tavily_search_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("extract_search_result", extract_search_result_node)

    graph.set_entry_point("extract_filter")
    graph.add_conditional_edges("extract_filter", route, ["retrieve"])
    graph.add_conditional_edges("retrieve", route, ["rerank"])
    graph.add_conditional_edges("tavily_search", route, ["extract_search_result"])
    graph.add_conditional_edges("extract_search_result", route, ["retrieve"])
    graph.add_conditional_edges("rerank", route, {"tavily_search": "tavily_search", "generate_answer": "generate_answer"})
    graph.add_edge("generate_answer", END)
    return graph.compile()
