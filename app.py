import streamlit as st
import asyncio
import hashlib
import os

os.environ["USER_AGENT"] = "MyAgent/0.1"

from langchain_community.document_loaders import SitemapLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler


with st.sidebar:
    OPENAI_API_KEY = st.text_input(
        label="OPENAI_API_KEY",
    )
    url = st.text_input(
        label="Write down a URL",
        value="https://developers.cloudflare.com/sitemap-0.xml",
        disabled=True,
    )
    st.write("https://github.com/animasana/assignment08/blob/main/app.py")


history = StreamlitChatMessageHistory()


if not OPENAI_API_KEY:
    st.error("Input your own openai api key.")
    history.clear()
    st.stop()


class ChatCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        history.add_ai_message(self.message)

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


silent_llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.1,
    api_key=OPENAI_API_KEY,    
)

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.1,
    api_key=OPENAI_API_KEY,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)


answers_prompt = ChatPromptTemplate.from_template(
    template="""    
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
    """
)


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | silent_llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke({"question": question, "answers": condensed})


def parse_page(soup):
    divs = soup.find_all("div", class_="sidebar-pane")
    if divs:
        for div in divs:
            div.decompose()

    footer = soup.find("footer")
    if footer:
        footer.decompose()

    bottom_divs = soup.find_all("div", class_="astro-fxeopwe4")
    if bottom_divs:
        for bottom_div in bottom_divs:
            bottom_div.decompose()

    return str(soup.get_text()).replace("\n", " ").replace("xa0", " ")


def sha256_key_encoder(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=5000,
        chunk_overlap=1000,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            "https://developers.cloudflare.com/ai-gateway/",
            "https://developers.cloudflare.com/vectorize/",
            "https://developers.cloudflare.com/workers-ai/",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 3
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    cache_dir = LocalFileStore(root_path=f"./.cache/embeddings/cloudflare")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=cache_dir,
        key_encoder=sha256_key_encoder,
    )
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=cached_embeddings,
    )
    retriever = vectorstore.as_retriever()
    return retriever


def send_human_message(message):
    st.chat_message("human").markdown(message)
    history.add_user_message(message)


def paint_history():
    for msg in history.messages:
        st.chat_message(msg.type).markdown(msg.content)


st.set_page_config(
    page_title="Assignment08",
    page_icon="🖥️",
)


st.markdown(
    """
    # Assignment08 SiteGPT
    
    Ask questions grounded in the content of Cloudflare.

    Start by writing your own OPENAI_API_KEY on the sidebar.
    """
)


if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


retriever = load_website(url)
st.chat_message("ai").write("I'm ready! Ask away!")
paint_history()
message = st.chat_input("Ask a question to the website")
if message:
    send_human_message(message)
    chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    with st.spinner("Waiting a response..."):
        with st.chat_message("ai"):
            chain.invoke(message)
