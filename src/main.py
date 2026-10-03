import os
from dotenv import load_dotenv
import pypdf

load_dotenv()

class PDFParser:
    def __init__(self, file_path):
        self.reader = pypdf.PdfReader(file_path)
        self.pages = {}
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
                

if __name__=="__main__":
    parser = PDFParser(os.getenv("FILE_PATH"))
    parser.find_text('dEep learning')