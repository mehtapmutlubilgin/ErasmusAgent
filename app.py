%%writefile app.py
# ==============================================================================
# PROJE: Erasmus RAGent Chatbot (Akbank GenAI Bootcamp)
# DOSYA: app.py (Gradio Arayüzü ve RAG Pipeline Tanımı)
# ==============================================================================

import gradio as gr
import os
import pandas as pd
from langchain_core.documents import Document 

# LangChain İçe Aktarmaları: RAG Mimarisi için gerekli temel bileşenler
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# HATA DÜZELTME: Chroma'yı doğrudan langchain_community'den import et
from langchain_community.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate 

# ==============================================================================
# RAG PARAMETRELERİ VE OPTİMİZASYONLARI
# ==============================================================================

# LLM'e rolünü ve davranış kurallarını belirten sistem şablonu (System Prompt)
# Amaç: LLM'in cevabı kısıtlı kaynaklara dayandırmasını sağlamak ve halüsinasyonu önlemek.
SYSTEM_TEMPLATE = """
Sen bir Erasmus+ bilgilendirme uzmanısın. Kullanıcının sorusuna, öncelikle 
SAĞLANAN KAYNAK METİNLERİ kullanarak cevap ver. 
Cevaplarını her zaman Türkçe olarak ve net bir dille sun.

Eğer kaynak metinlerde cevap yoksa:
1. Konuyla ilgili genel, güvenilir bilgin varsa, "Kaynaklarda doğrudan cevap bulunmamaktadır, ancak genel bilgi şöyledir:" diyerek cevap verebilirsin.
2. Hiçbir bilgin yoksa, "Bu konuda elimde yeterli ve kesin bir bilgi bulunmamaktadır, lütfen üniversitenizin Uluslararası İlişkiler Ofisi'ne danışın." şeklinde kibarca cevapla.

KAYNAK: {context}
"""

# ==============================================================================
# RAG ZİNCİRİNİ BAŞLATMA FONKSİYONU (SETUP)
# ==============================================================================
def setup_rag_chain():
    """RAG zincirini kurar ve döndürür. Uygulama başladıktan sonra yalnızca bir kez çalışır."""
    
    # 1. API Anahtarını Yükleme (Güvenlik Kriteri)
    # Hugging Face Spaces'ta 'GEMINI_API_KEY' adıyla Secret olarak tanımlanmıştır.
    try:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
             raise ValueError("API anahtarı (GEMINI_API_KEY) Hugging Face Secrets'ta bulunamadı.")
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY 
    except Exception as e:
        print(f"HATA: API Anahtarı yüklenemedi. Detay: {e}")
        return None
    
    # 2. Veri Setini Okuma ve Dokümanları Oluşturma (Veri Hazırlık Kriteri)
    try:
        DATA_FILE_PATH = 'erasmus_chatbot_dataset.csv' # HF Spaces'taki yerel dosya yolu
        df = pd.read_csv(DATA_FILE_PATH)
        
        # Optimizasyon: Her Soru-Cevap çifti tek bir Document objesi yapılır.
        documents_for_rag = []
        for index, row in df.iterrows():
            doc_content = f"Kategori: {row['kategori']}. Soru: {row['soru']}. Cevap: {row['cevap']}"
            documents_for_rag.append(Document(page_content=doc_content))
            
        texts = documents_for_rag 
            
    except Exception as e:
        print(f"HATA: Veri Seti Okuma veya Doküman Oluşturma Başarısız. Detay: {e}")
        return None

    # 3. Gömme (Embedding) Modeli Tanımı
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    
    # 4. Vektör Veritabanı Oluşturma
    db = Chroma.from_documents(texts, embeddings)

    # 5. Büyük Dil Modeli (LLM) Tanımı
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0, 
        google_api_key=GEMINI_API_KEY
    )
    
    # 6. Prompt Şablonunu Uygulama
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ])

    # 7. RetrievalQA Zinciri Kurulumu (RAG Pipeline)
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
    
    print("RAG Zinciri başarıyla kuruldu ve optimize edildi.")
    return qa_chain

# Global olarak RAG zincirini bir kere yükle
qa_chain = setup_rag_chain()

# ==============================================================================
# GRADIO CHATBOT CEVAP FONKSİYONU
# ==============================================================================
def chatbot_response(message, history):
    """Kullanıcı mesajını RAG zincirine gönderir ve çekilen kaynaklarla birlikte cevabı döndürür."""
    
    if qa_chain is None:
        return "Chatbot kurulumu başarısız oldu. Lütfen logları kontrol edin."

    response = qa_chain({"query": message})
    answer = response['result']
    source_docs = response['source_documents']
    
    # Kaynak Bilgisini Hazırlama (Şeffaflık)
    sources_text = "\n\n***\n**Cevap Kaynakları:**\n"
    for doc in source_docs:
        content = doc.page_content.replace('Soru:', 'Soru: ').replace('. Cevap:', ' | Cevap: ')
        sources_text += f"*{content[:150]}...*\n"

    full_response = answer + sources_text
    
    return full_response

# ==============================================================================
# GRADIO ARAYÜZ TANIMLAMASI (Web Arayüzü Kriteri)
# ==============================================================================
iface = gr.ChatInterface(
    fn=chatbot_response,
    title="🇪🇺 Erasmus RAGent Chatbot (Optimize Edilmiş)", 
    description="RAG (Retrieval Augmented Generation) ile Erasmus+ Bilgilendirme Sistemi. Sorularınızı sorabilirsiniz!",
    chatbot=gr.Chatbot(height=500),
    examples=["Erasmus'a kimler başvurabilir?", "Green Erasmus (Yeşil Erasmus) nedir?", "Hibe ödemesi ne zaman yapılır?"],
    theme="soft"
)

if __name__ == "__main__":
    iface.launch()
