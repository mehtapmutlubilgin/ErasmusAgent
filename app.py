
import gradio as gr
import os
import pandas as pd

# Gerekli kütüphaneler (Hugging Face'e yüklediğinizden emin olun)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate # GÜNCELLEME 3: Prompt için

# GÜNCELLEME 3: Sistem Şablonu (System Prompt)
SYSTEM_TEMPLATE = """
Sen bir Erasmus+ bilgilendirme uzmanısın. Kullanıcının sorusuna sadece ve kesinlikle 
SAĞLANAN KAYNAK METİNLERİ kullanarak cevap ver. 
Eğer kaynak metinlerde cevap yoksa, "Bu konuda elimde kesin bir bilgi bulunmamaktadır, 
lütfen üniversitenizin Uluslararası İlişkiler Ofisi'ne danışın." şeklinde cevapla. 
Cevaplarını her zaman Türkçe olarak ve net bir dille sun.

KAYNAK: {context}
"""

# RAG ZİNCİRİNİ BAŞLATMA FONKSİYONU
def setup_rag_chain():
    """RAG zincirini kurar ve döndürür."""
    
    # 1. API Anahtarını Yükleme (Hugging Face Secrets'tan)
    try:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
             # Eğer anahtar yoksa, hata döndür.
             raise ValueError("API anahtarı (GEMINI_API_KEY) Hugging Face Secrets'ta bulunamadı.")
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    except Exception as e:
        print(f"HATA: API Anahtarı yüklenemedi: {e}")
        return None
    
    # 2. Veri Setini Okuma (Yerel Dizin)
    try:
        DATA_FILE_PATH = 'erasmus_chatbot_dataset.csv' 
        df = pd.read_csv(DATA_FILE_PATH)
        df['kaynak_metin'] = "Kategori: " + df['kategori'] + ". Soru: " + df['soru'] + ". Cevap: " + df['cevap']
        documents_for_rag = df['kaynak_metin'].tolist()
    except Exception as e:
        print(f"HATA: Veri Seti Okuma Başarısız: {e}")
        return None

    # 3. RAG Mimarisi Kurulumu
    
    # GÜNCELLEME 1: RecursiveCharacterTextSplitter kullanarak daha iyi parçalama
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Maksimum parça boyutu
        chunk_overlap=50,      # Parçaların örtüşme miktarı (bağlam kaybını önler)
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents(documents_for_rag)

    # API anahtarını doğrudan ilet
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    db = Chroma.from_documents(texts, embeddings)

    # API anahtarını doğrudan ilet
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0, # Yaratıcılığı azaltıp kesin bilgiye yönlendirdik
        google_api_key=GEMINI_API_KEY
    )
    
    # GÜNCELLEME 3: Prompt Şablonunu Oluşturma
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ])

    # GÜNCELLEME 2 & 3: RetrievalQA Zinciri
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(k=5), # k değerini 5'e çıkardık (Daha fazla bağlam)
        return_source_documents=True,
        # Prompt'u chain_type_kwargs üzerinden ekleyerek System Prompt'u etkinleştir
        chain_type_kwargs={
             "prompt": qa_prompt,
             "document_variable_name": "context" # Prompt'taki context değişkenini eşleştirir
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
