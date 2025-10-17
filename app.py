import gradio as gr
import os
import pandas as pd

# Gerekli kütüphaneler (LangChain 0.3.x yapısına uygun)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate


# 🔹 Sistem Prompt (güncel)
SYSTEM_TEMPLATE = """
Sen bir Erasmus+ bilgilendirme uzmanısın. Kullanıcının sorusuna sadece ve kesinlikle 
SAĞLANAN KAYNAK METİNLERİ kullanarak cevap ver. 
Eğer kaynak metinlerde cevap yoksa, "Bu konuda elimde kesin bir bilgi bulunmamaktadır, 
lütfen üniversitenizin Uluslararası İlişkiler Ofisi'ne danışın." şeklinde cevapla. 
Cevaplarını her zaman Türkçe olarak ve net bir dille sun.

KAYNAK: {context}
"""

# 🔹 RAG Zinciri Kurulumu
def setup_rag_chain():
    """RAG zincirini kurar ve döndürür."""
    try:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("API anahtarı (GEMINI_API_KEY) Hugging Face Secrets'ta bulunamadı.")
    except Exception as e:
        print(f"HATA: API Anahtarı yüklenemedi: {e}")
        return None
    
    try:
        DATA_FILE_PATH = 'erasmus_chatbot_dataset.csv' 
        df = pd.read_csv(DATA_FILE_PATH)
        df['kaynak_metin'] = "Kategori: " + df['kategori'] + ". Soru: " + df['soru'] + ". Cevap: " + df['cevap']
        documents_for_rag = df['kaynak_metin'].tolist()
    except Exception as e:
        print(f"HATA: Veri Seti Okuma Başarısız: {e}")
        return None

    # 🔹 Metinleri parçalara ayır
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    texts = text_splitter.create_documents(documents_for_rag)

    # 🔹 Embedding ve veritabanı
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    db = Chroma.from_documents(texts, embeddings)

    # 🔹 LLM (Gemini 2.5)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    # 🔹 Prompt Şablonu
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ])

    # 🔹 RetrievalQA zinciri
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(k=5),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": qa_prompt,
            "document_variable_name": "context"
        }
    )
    
    print("✅ RAG Zinciri başarıyla kuruldu.")
    return qa_chain


qa_chain = setup_rag_chain()


# 🔹 Chatbot cevabı
def chatbot_response(message, history):
    if qa_chain is None:
        return "❌ Chatbot kurulumu başarısız oldu. Lütfen API anahtarını kontrol edin."

    response = qa_chain({"query": message})
    answer = response['result']
    source_docs = response['source_documents']

    sources_text = "\n\n---\n**Kaynaklar:**\n"
    for doc in source_docs:
        content = doc.page_content.replace('Soru:', 'Soru: ').replace('. Cevap:', ' | Cevap: ')
        sources_text += f"- {content[:150]}...\n"

    return answer + sources_text


# 🔹 Gradio Arayüzü
iface = gr.ChatInterface(
    fn=chatbot_response,
    title="🇪🇺 Erasmus RAGent Chatbot (Optimize Edilmiş)",
    description="RAG (Retrieval Augmented Generation) ile Erasmus+ Bilgilendirme Sistemi. Sorularınızı sorabilirsiniz!",
    chatbot=gr.Chatbot(height=500),
    examples=[
        "Erasmus'a kimler başvurabilir?",
        "Öğrenim süresi ne kadardır?",
        "Hibesiz Erasmus yapılabilir mi?"
    ],
    theme="soft"
)

if __name__ == "__main__":
    iface.launch()
