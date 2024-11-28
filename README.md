# Vaidya-GPT-Prototype(PHASE-I)
## **Overview**
This project demonstrates a **Retrieval-Augmented Generation (RAG)** model implementation designed to answer questions based on medical literature. The RAG model combines **retrieval-based** techniques and **Generative AI** for accurate and contextually relevant responses.

---

## **Dataset**
The model is trained and tested using the following medical books:
1. **Medical Book**: General medical reference.
2. **Gray's Anatomy for Students**: Detailed anatomy reference.
3. **Harrison's Principles of Internal Medicine**: Comprehensive internal medicine resource.
4. **Oxford Handbook of Clinical Medicine**: Clinical reference for healthcare professionals.
5. **Where There Is No Doctor**: Health care guide for rural and remote areas.
6. **Current Medical Diagnosis & Treatment**: Diagnostic and treatment guidelines.
7. **Davidson’s Principles and Practice of Medicine**: Principles of modern medicine.
8. **Harrison’s Pulmonary and Critical Care Medicine**: Specialized reference for pulmonary medicine.

Each book is processed to extract relevant information, chunked into smaller text segments, and stored for retrieval during question answering.

---

## **Technology Stack**
### **Languages and Frameworks**
- **Python 3.10.0**
- **LangChain**: Framework for building language model applications.
- **Chroma**: Vectorstore for document embeddings.
- **Google Generative AI**: For embedding generation and answer generation.
- **FAISS**: For efficient similarity searches.
**You can download the above techstack from requirements.txt--> {https://github.com/eeshan15/Vaidya-GPT-Prototype/blob/main/requirements.txt} and 
  complete code from test.ipynb file ---> https://github.com/eeshan15/Vaidya-GPT-Prototype/blob/main/test.ipynb**
### **Dependencies**
- `langchain`
- `langchain_community`
- `langchain_chroma`
- `langchain_google_genai`
- `PyPDFLoader`: For loading PDF data.
- `dotenv`: For managing environment variables.
- `FAISS`: For vector-based similarity search.
- `tqdm`: For progress visualization.

---

## **Model Architecture**
1. **Document Loader**:
   - PDFs are loaded using `PyPDFLoader`.

2. **Text Splitting**:
   - Documents are split into chunks of 1000 characters using `RecursiveCharacterTextSplitter`.

3. **Embeddings**:
   - Generated using the **Google Generative AI Embeddings** model (`embedding-001`).

4. **Vectorstore**:
   - Chunks and their embeddings are stored in **Chroma** or **FAISS** for efficient retrieval.

5. **Generative AI**:
   - Uses the **Gemini 1.5 Pro** version for natural language response generation.

6. **Retrieval Chain**:
   - Combines retrieved documents with a generative model to answer questions.

---

## **Key Features**
- **Multiple Document Support**:
  The model is designed to handle multiple books and provide consolidated answers.
- **Customizable Prompting**:
  The system uses dynamic prompts to tailor responses based on retrieved content.
- **High Accuracy**:
  Tested to deliver accurate and concise answers to medical queries.

---

## **Usage Instructions**

### **Environment Setup**
1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install langchain langchain_chroma langchain_google_genai dotenv tqdm
   ```
3. Place your medical books in the `data/` directory.

### **Running the Notebook**
1. Open `test.ipynb` in Jupyter Notebook or an equivalent editor.
2. Run all the cells sequentially.

### **Generating Vectorstore**
Run the following code to generate and save the vectorstore:
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load and process documents
loader = PyPDFLoader("data/Medical_book.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create vectorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="vectorstore")
vectorstore.persist()
```

### **Querying the Model**
After generating the vectorstore, run the following snippet to query:
```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
question = "What is Myopia?"
response = rag_chain.invoke({"input": question})
print(response["answer"])
```

---

## **Potential Improvements**
- Extend the model to include more specialized datasets.
- Optimize retrieval and generation for faster response times.
- Integrate a user-friendly interface with **Streamlit**.

---

## **Conclusion**
This RAG model effectively answers complex medical questions by leveraging retrieval and generation, offering a valuable tool for students, practitioners, and researchers in medicine.

---
