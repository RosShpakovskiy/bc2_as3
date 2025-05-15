import streamlit as st
import chromadb
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import ollama

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="constitution")

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🇰🇿 AI Constitutional Expert")

# -----------------------------------------
# Load Constitution and Index if Needed
# -----------------------------------------
def load_preloaded_constitution():
    constitution_path = "data/akorda.kz-Constitution of the Republic of Kazakhstan.pdf"
    if os.path.exists(constitution_path):
        try:
            loader = PyPDFLoader(constitution_path)
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\nArticle ", "\nSection ", "\nCHAPTER "]
            )
            docs = text_splitter.split_documents(pages)

            for doc in docs:
                text = doc.page_content
                article_match = re.search(r'Article (\d+)', text, re.IGNORECASE)
                doc.metadata["article"] = article_match.group(1) if article_match else "preamble"

            texts = [doc.page_content for doc in docs]
            embeddings = embedding_model.encode(texts).tolist()
            metadatas = [{"article": doc.metadata["article"]} for doc in docs]
            ids = [f"art_{doc.metadata['article']}_{i}" for i, doc in enumerate(docs)]

            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            st.session_state.constitution_docs = docs  # Save for keyword search
            st.success("Constitution preloaded with article metadata!")

        except Exception as e:
            st.error(f"Error loading constitution: {str(e)}")
    else:
        st.warning("Preloaded Constitution PDF not found in data/ folder")

if "constitution_docs" not in st.session_state:
    load_preloaded_constitution()

# -----------------------------------------
# Helper Functions
# -----------------------------------------
def get_article_number(prompt: str) -> str:
    patterns = [r"article (\d+)", r"art\.? (\d+)", r"\b(\d+)(?:th|st|nd|rd) article"]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def find_article_by_exact_phrase(phrase: str, docs: list):
    for doc in docs:
        if phrase.lower() in doc.page_content.lower():
            return doc.metadata.get("article", "unknown"), doc.page_content
    return None, None

# -----------------------------------------
# Chat Interface
# -----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the Constitution (e.g. 'Quote Article 55' or ask a legal phrase)"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Step 1: Check for exact phrase in quotes
    phrase_match = re.search(r'"([^"]+)"', prompt)
    answer = ""
    found_by_phrase = False

    if phrase_match and "constitution_docs" in st.session_state:
        phrase = phrase_match.group(1)
        article, matched_text = find_article_by_exact_phrase(phrase, st.session_state.constitution_docs)
        if article:
            answer = f"Article {article} states:\n\n{matched_text}"
            found_by_phrase = True

    # Step 2: Fallback to embedding search
    if not found_by_phrase:
        article_number = get_article_number(prompt)
        query_embedding = embedding_model.encode(prompt).tolist()
        where_filter = {"article": {"$eq": article_number}} if article_number else None

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=4,
                where=where_filter
            )
            context = "\n".join(results['documents'][0])
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            context = ""

        if context:
            response = ollama.chat(
                model="llama2",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a constitutional expert. Respond using ONLY the provided context.
For article requests:
- If context contains the full article: return it verbatim with "Article X states: [exact text]"
- If partial: "Excerpt from Article X: [text] (full text not available)"
- No interpretations. Say "Not found in Constitution" if missing."""
                    },
                    {
                        "role": "user",
                        "content": f"CONTEXT:\n{context}\n\nQUERY: {prompt}"
                    }
                ]
            )
            answer = response['message']['content']
        else:
            answer = "No relevant constitutional text found for this query."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
