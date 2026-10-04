import os
from dotenv import load_dotenv
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        for i in range(len(self.pages)):
            index = self.pages[i].lower().find(text.lower())
            if index != -1:
                start = max(0,index-50)
                end = min(index+50,len(self.pages[i]))
                print("="*10+f"Trang thứ {i+1}"+"="*10)
                print(self.pages[i][start:end])
                found = True
        if not found:
            print("Không tìm thấy!")

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
        for  i in range(n):
            print(f"---Chunk {i+1}---")
            print("Score: "+ str(result[i]["score"]))
            print(result[i]["chunk"]["text"])
                
if __name__=="__main__":
    parser = SearchEngine(os.getenv("FILE_PATH"))
    parser.find_text('dEep learning')
    parser.chunking()
    print(parser.chunk[0])
    parser.vectorize()
    print(len(parser.vectorizer.get_feature_names_out()))
    print(parser.tfidf_matrix.shape)
    parser.search_query("deep learning for image classification",3)