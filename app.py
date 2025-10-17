import gradio as gr
import os
import pandas as pd

# LangChain şema importu (Dokümanları oluşturmak için)
from langchain.schema import Document 

# Gerekli kütüphaneler
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma # langchain_community'den otomatik çekilir
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate 

# GÜNCELLEME: Sistem Şablonu (System Prompt) - Biraz daha esnek
SYSTEM_TEMPLATE = """
Sen bir Erasmus+ bilgilendirme uzmanısın. Kullanıcının sorusuna, öncelikle 
SAĞLANAN KAYNAK METİNLERİ kullanarak cevap ver. 
Cevaplarını her zaman Türkçe olarak ve net bir dille sun.

Eğer kaynak metinlerde cevap yoksa:
1. Konuyla ilgili genel, güvenilir bilgin varsa, "Kaynaklarda doğrudan cevap bulunmamaktadır, ancak genel bilgi şöyledir:" diyerek cevap verebilirsin.
2. Hiçbir bilgin yoksa, "Bu konuda elimde yeterli ve kesin bir bilgi bulunmamaktadır, lütfen üniversitenizin Uluslararası İlişkiler Ofisi'ne danışın." şeklinde kibarca cevapla.

KAYNAK: {context}
"""

# RAG ZİNCİRİNİ BAŞLATMA FONKSİYONU
def setup_rag_chain():
    """RAG zincirini kurar ve döndürür."""
    
    # 1. API Anahtarını Yükleme (Hugging Face Secrets'tan)
    try:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
             raise ValueError("API anahtarı (GEMINI_API_KEY) Hugging Face Secrets'ta bulunamadı.")
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    except Exception as e:
        print(f"HATA: API Anahtarı yüklenemedi: {e}")
        return None
    
    # 2. Veri Setini Okuma ve Dokümanları Oluşturma (GÜNCEL YÖNTEM)
    try:
        DATA_FILE_PATH = 'erasmus_chatbot_dataset.csv' 
        df = pd.read_csv(DATA_FILE_PATH)
        
        # Her Soru-Cevap çiftini tek bir LangChain Document objesi yap
        documents_for_rag = []
        for index, row in df.iterrows():
            doc_content = f"Kategori: {row['kategori']}. Soru: {row['soru']}. Cevap: {row['cevap']}"
            documents_for_rag.append(Document(page_content=doc_content))
            
    except Exception as e:
        print(f"HATA: Veri Seti Okuma veya Doküman Oluşturma Başarısız: {e}")
        return None

    # Artık metin bölmeye (splitting) gerek yok, her satır zaten tek doküman
    texts = documents_for_rag

    # API anahtarını doğrudan ilet
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    db = Chroma.from_documents(texts, embeddings) # Dokümanlar doğrudan vektörleştirilir

    # API anahtarını doğrudan ilet
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0, 
        google_api_key=GEMINI_API_KEY
    )
    
    # Prompt Şablonunu Oluşturma
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ])

    # RetrievalQA Zinciri
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(k=5), # k=5 (5 doküman çek) kullanmaya devam ediyoruz
        return_source_documents=True,
        chain_type_kwargs={
             "prompt": qa_prompt,
             "document_variable_name": "context"
        }
    )
    
    print("RAG Zinciri başarıyla kuruldu ve optimize edildi.")
    return qa_chain

# Global olarak RAG zincirini bir kere yükle
qa_chain = setup_rag_chain()

def chatbot_response(message, history):
    """Gelen mesajı RAG zincirine gönderir ve cevabı döndürür."""
    
    if qa_chain is None:
        return "Chatbot kurulumu başarısız oldu. Lütfen Hugging Face Secrets ve logları kontrol edin."

    response = qa_chain({"query": message})
    answer = response['result']
    source_docs = response['source_documents']
    
    # Kaynak Bilgisini Hazırlama
    sources_text = "\n\n***\n**Kaynaklar:**\n"
    for doc in source_docs:
        # Kaynak metin formatını daha anlaşılır hale getir
        content = doc.page_content.replace('Soru:', 'Soru: ').replace('. Cevap:', ' | Cevap: ')
        sources_text += f"*{content[:150]}...*\n"

    full_response = answer + sources_text
    
    return full_response

# GRADIO ARAYÜZÜ
iface = gr.ChatInterface(
    fn=chatbot_response,
    title="🇪🇺 Erasmus RAGent Chatbot (Optimize Edilmiş)", 
    description="RAG (Retrieval Augmented Generation) ile Erasmus+ Bilgilendirme Sistemi. Sorularınızı sorabilirsiniz!",
    chatbot=gr.Chatbot(height=500),
    examples=["Erasmus'a kimler başvurabilir?", "Öğrenim süresi ne kadardır?", "Hibesiz Erasmus yapılabilir mi?"],
    theme="soft"
)

if __name__ == "__main__":
    iface.launch()
