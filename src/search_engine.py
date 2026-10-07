import os
from dotenv import load_dotenv
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import chromadb

load_dotenv()

class SearchEngine:
    def __init__(self, file_path):
        self.reader = pypdf.PdfReader(file_path)
        self.pages = {}
        self.chunk = []
        for i,page in enumerate(self.reader.pages):
            self.pages[i] = page.extract_text()

    def show(self):
        for i,page in self.pages.items():
            print("="*10+f"Trang thứ {i+1}"+"="*10)
            print(page)

    def find_text(self,text:str):
        found = False
        result = []
        for i in range(len(self.pages)):
            index = self.pages[i].lower().find(text.lower())
            if index != -1:
                start = max(0,index-50)
                end = min(index+50,len(self.pages[i]))
                result.append({
                    "page": i,
                    "text": self.pages[i][start:end]
                    })
                found = True
        if not found:
            print("Không tìm thấy!")
            return
        return result

    def chunking(self):
        for i in range(len(self.pages)):
            words = self.pages[i].split()
            for j in range(0,len(words),100):
                self.chunk.append(
                    {
                        "page": i+1,
                        "text": " ".join(words[j:min(j+100,len(words))])
                    }
                )

    def vectorize(self):
        self.vectorizer = TfidfVectorizer()
        words = []
        for i in range(len(self.chunk)):
            words.append(self.chunk[i]["text"])
        self.tfidf_matrix = self.vectorizer.fit_transform(words)

    def search_query(self,query:str,n):
        tfidf_query = self.vectorizer.transform([query])
        similarity = cosine_similarity(self.tfidf_matrix,tfidf_query)
        result = []
        for i in range(len(similarity)):
            result.append({
                "score": float(similarity[i][0]),
                "chunk": self.chunk[i]
            })
        result.sort(key=lambda x: x["score"],reverse=True)
        return result[:n]

    def embedding(self,query,n):
        self.model = SentenceTransformer("BAAI/bge-m3",similarity_fn_name="cosine")
        query_vector = self.model.encode(query,normalize_embeddings=True)
        result = []
        texts = [self.chunk[i]["text"] for i in range(len(self.chunk))]
        self.embed = self.model.encode(texts,normalize_embeddings=True)
        cos_similarity = cos_sim(self.embed,query_vector)
        for i in range(len(cos_similarity)):
            result.append({
                "score": float(cos_similarity[i][0]),
                "chunk": self.chunk[i]
            })
        result.sort(key=lambda x: x["score"],reverse=True)
        return result[:n]

    def create_vector_db(self):
        client = chromadb.PersistentClient("./vector_db")
        self.collection = client.get_or_create_collection(name="research_documents")
        embedding = self.embed
        documents,pages = [],[]
        ids = [f"chunk_{i}" for i in range(len(self.embed))]
        for chunk in self.chunk:
            documents.append(chunk["text"])
            pages.append(chunk["page"])
        meta_datas = [{"page": page} for page in pages]
        self.collection.add(embeddings=embedding,
                       ids = ids,
                       documents=documents,
                       metadatas=meta_datas
                       )
        print("vector database created!")

    def query_vector_db(self,query:str,n):
        query_vector = self.model.encode(query,normalize_embeddings=True)
        reply = self.collection.query([query_vector],n_results=n)
        return reply
        

    def generate_answer(self,query:str,n):
        answer = self.query_vector_db(query,n)
        print("===CONTEXT===")
        for i in range(n):
            print(f"---{answer['ids'][0][i]}---")
            print(f"page: {answer['metadatas'][0][i]['page']}")
            print(f"score: {answer['distances'][0][i]}")
            print(f"content: {answer['documents'][0][i]}")
                
if __name__=="__main__":
    parser = SearchEngine(os.getenv("FILE_PATH"))
    parser.find_text('dEep learning')
    parser.chunking()
    print(parser.chunk[0])
    parser.vectorize()
    print(len(parser.vectorizer.get_feature_names_out()))
    print(parser.tfidf_matrix.shape)
    query = "deep learning for image classification"
    parser.search_query(query,3)
    parser.embedding(query,3)
    parser.create_vector_db()
    parser.query_vector_db(query,3)
    parser.generate_answer(query,3)