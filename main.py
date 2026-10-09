"""AxiomRAG CLI; supports PDF paths/URLs and YouTube transcripts."""
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
import agent

def load_youtube(url):
    if "v=" in url: vid=url.split("v=",1)[1].split("&",1)[0]
    elif "youtu.be/" in url: vid=url.split("youtu.be/",1)[1].split("?",1)[0]
    else: raise ValueError("Invalid YouTube URL.")
    transcript=YouTubeTranscriptApi().fetch(vid)
    text=" ".join(getattr(t,"text","") for t in transcript)
    if not text.strip(): raise ValueError("Transcript was empty.")
    return [Document(page_content=text,metadata={"source":url})]

def main():
    print("\nAxiomRAG — Retrieve. Verify. Refine. Respond.")
    choice=input("1. PDF (path/URL)  2. YouTube transcript: ").strip()
    try:
        if choice=="1": docs=PyPDFLoader(input("PDF path or URL: ").strip()).load()
        elif choice=="2": docs=load_youtube(input("YouTube URL: ").strip())
        else: print("Invalid option."); return
        if not docs: print("No content loaded."); return
        print("Building index; first run may take a minute...")
        app,count=agent.build_pipeline(docs,1000,50,3,"llama3")
    except Exception as e: print(f"Setup error: {e}"); return
    print(f"Indexed {count} chunks. Ask questions; type exit to quit.")
    history=[]
    while True:
        q=input("\nYou: ").strip()
        if q.lower()=="exit": break
        if not q: continue
        try:
            result=agent.run_query(app,q,3,history[-6:])
            for item in result["run_history"]: print(f"Attempt {item['iteration']} | {item['query']} | {item['verdict']}")
            print("\nAxiomRAG:",result["final_answer"])
            history.extend([{"role":"user","content":q},{"role":"assistant","content":result["final_answer"]}])
        except Exception as e: print("Query error:",e)
if __name__=="__main__": main()
