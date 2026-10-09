"""Streamlit UI for AxiomRAG."""
import os, tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
import agent

st.set_page_config(page_title="AxiomRAG",page_icon="🧠",layout="wide")
st.markdown("""<style>.block-container{padding-top:1.8rem}.hero{padding:1.5rem;border-radius:18px;color:white;background:linear-gradient(120deg,#172554,#312e81,#581c87);margin-bottom:1rem}</style>""",unsafe_allow_html=True)

def load_pdf(file):
    path=None
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as f:
            f.write(file.getvalue()); path=f.name
        return PyPDFLoader(path).load()
    finally:
        if path and os.path.exists(path): os.unlink(path)

def load_youtube(url):
    if "v=" in url: vid=url.split("v=",1)[1].split("&",1)[0]
    elif "youtu.be/" in url: vid=url.split("youtu.be/",1)[1].split("?",1)[0]
    else: raise ValueError("Enter a valid YouTube URL.")
    transcript=YouTubeTranscriptApi().fetch(vid)
    text=" ".join(getattr(t,"text","") for t in transcript)
    if not text.strip(): raise ValueError("Transcript is empty.")
    return [Document(page_content=text,metadata={"source":url})]

def telemetry(items,prefix):
    if not items: return
    with st.expander("🔎 Critic and self-healing diagnostics"):
        for item in items:
            st.markdown(f"**Attempt {item.get('iteration')} — {item.get('verdict')}**")
            st.caption("Query: "+item.get("query",""))
            if item.get("failure_reason"): st.warning(item["failure_reason"])
            st.write("Draft:",item.get("answer",""))
            for i,ch in enumerate(item.get("chunks",[]),1):
                if isinstance(ch,dict):
                    content=ch.get("content",""); meta=ch.get("metadata",{})
                    st.caption(f"Evidence {i}: {meta}")
                else: content=str(ch)
                st.text_area(f"Evidence chunk {i}",content,height=100,disabled=True,key=f"{prefix}_{item.get('iteration')}_{i}")

for key,val in {"messages":[],"qa_chain":None,"current_source":None,"loaded_docs":None,"num_chunks":0,"total_chars":0,"indexed_settings":None}.items():
    if key not in st.session_state: st.session_state[key]=val
with st.sidebar:
    st.title("🧠 AxiomRAG")
    source_type=st.radio("Source",["PDF","YouTube"])
    pdf=st.file_uploader("Upload PDF",type=["pdf"]) if source_type=="PDF" else None
    url=st.text_input("YouTube URL") if source_type=="YouTube" else ""
    st.divider()
    chunk_size=st.slider("Chunk size",200,2000,1000,100)
    overlap=st.slider("Chunk overlap",0,300,50,10)
    k=st.slider("Retrieved chunks",1,5,3)
    model=st.text_input("Ollama model","llama3")
    attempts=st.slider("Maximum attempts",1,5,3)
    load=st.button("Load / rebuild source",type="primary",use_container_width=True)
    if st.button("Clear chat",use_container_width=True): st.session_state.messages=[]; st.rerun()

st.markdown('<div class="hero"><h1>AxiomRAG</h1><p>Retrieve. Verify. Refine. Respond.</p><p>Intelligent retrieval · Grounding critic · Bounded retries · Regression tests</p></div>',unsafe_allow_html=True)
if load:
    try:
        with st.spinner("Reading source and indexing..."):
            if source_type=="PDF":
                if not pdf: st.error("Upload a PDF first."); st.stop()
                source=f"pdf:{pdf.name}:{pdf.size}"; docs=load_pdf(pdf)
            else:
                if not url.strip(): st.error("Enter a YouTube URL first."); st.stop()
                source="youtube:"+url.strip(); docs=load_youtube(url.strip())
            chain,count=agent.build_pipeline(docs,chunk_size,overlap,k,model.strip() or "llama3")
            st.session_state.qa_chain=chain; st.session_state.loaded_docs=docs
            st.session_state.current_source=source; st.session_state.num_chunks=count
            st.session_state.total_chars=sum(len(d.page_content) for d in docs)
            st.session_state.indexed_settings=(chunk_size,overlap,k,model); st.session_state.messages=[]
        st.success(f"Indexed {count} chunks.")
    except Exception as e: st.error(f"Source/indexing error: {e}")

if st.session_state.qa_chain is None:
    st.info("Choose a source in the sidebar and click Load / rebuild source.")
    a,b,c=st.columns(3)
    a.markdown("### 1. Intelligent retrieval\nFAISS finds relevant chunks.")
    b.markdown("### 2. Critic verification\nChecks grounding and relevance.")
    c.markdown("### 3. Self-healing retries\nRewrites failed searches within a fixed limit.")
else:
    a,b,c=st.columns(3); a.metric("Chunks",st.session_state.num_chunks); b.metric("Characters",f"{st.session_state.total_chars:,}"); c.metric("Max attempts",attempts)
    st.caption("Source: "+str(st.session_state.current_source))
    if (chunk_size,overlap,k,model)!=st.session_state.indexed_settings: st.warning("Settings changed; click Load / rebuild source to apply.")
    for i,m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("history"): telemetry(m["history"],f"old_{i}")
    q=st.chat_input("Ask a question about your source...")
    if q:
        st.session_state.messages.append({"role":"user","content":q})
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving, answering, and verifying..."):
                    result=agent.run_query(st.session_state.qa_chain,q,attempts,st.session_state.messages[:-1][-6:])
                st.markdown(result["final_answer"])
                if result["failed"]: st.warning("Could not verify a supported answer within the retry limit.")
                else: st.success("Answer passed critic checks.")
                st.session_state.messages.append({"role":"assistant","content":result["final_answer"],"history":result["run_history"]})
                telemetry(result["run_history"],f"new_{len(st.session_state.messages)}")
            except Exception as e: st.error(f"Query error: {e}")
