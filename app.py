from flask import Flask, request, jsonify, stream_with_context, Response
import json
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import fitz

from db import db  
from models import User, File 
import os
from werkzeug.utils import secure_filename


from dotenv import load_dotenv

# LangChain / vectorstore imports
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma



# from utilize import model_LLM



app = Flask(__name__)

# Configurations
app.config['JWT_SECRET_KEY'] = 'super-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

# Extensions
CORS(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
db.init_app(app)





#-------------------------------------------------------------------------------------


# Chroma persist directory
CHROMA_DIR = os.path.join(os.getcwd(), 'chroma_db')
os.makedirs(CHROMA_DIR, exist_ok=True)

# Embedding model spec
EMBEDDING_MODEL = "models/embedding-001"  # adjust if needed



def model_LLM(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env file")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    # Keep it simple: pass the prompt as a single user message
    resp = llm.invoke(prompt)
    # The SDK may return different shapes; we assume .content is valid as in your snippet
    return getattr(resp, 'content', str(resp))




# Indexing function for PDF (call this after file upload)
def index_pdf(file_path: str, file_id: int, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load PDF, split into chunks, embed and persist to Chroma under collection file_{file_id}.
       Designed for PDF-only indexing (user selected option 1).
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_split = splitter.split_documents(docs)

    # Build embeddings and Chroma instance
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    collection_name = f"file_{file_id}"
    db = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=CHROMA_DIR)

    # Clear existing docs in this collection (optional: comment out to append)
    try:
        db._collection.delete()  # low-level clear; depends on Chroma internals
    except Exception:
        pass

    # Add docs and persist
    db.add_documents(docs_split)
    # db.persist()
    return {
        'collection': collection_name,
        'num_chunks': len(docs_split),
        'persist_directory': CHROMA_DIR
    }


# Retrieval + LLM pipeline
def process_with_rag(file_id: int, user_message: str, k: int = 3) -> str:
    """Retrieve relevant chunks from the chroma collection for file_id and call LLM with context + question."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env file")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    collection_name = f"file_{file_id}"

    db = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=CHROMA_DIR)

    # similarity_search returns list of Document objects
    docs = db.similarity_search(user_message, k=k)
    if not docs:
        context = ""
    else:
        # join top-k passages
        context = "\n\n".join([d.page_content for d in docs])

    # Build a concise prompt that instructs the model to answer using context and be honest when unsure
    prompt = f"""
You are an assistant that answers questions using ONLY the provided context. If the answer is not in the context, say "I don't know" or ask for clarifying info.

Context:\n{context}\n\nQuestion: {user_message}\n
Answer:
"""

    response = model_LLM(prompt)
    return response








# --- Routes ---
# Existing route modified to use RAG. Keep your JWT and User lookup logic.
@app.route('/process-message/<int:file_id>', methods=['POST'])
@jwt_required()
def process_message(file_id):
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    # Find file in DB by ID and user
    file_record = File.query.filter_by(id=file_id, user_id=user.id).first()
    if not file_record:
        return jsonify({'message': 'File not found in database'}), 404

    file_path = file_record.filepath
    if not os.path.isfile(file_path):
        return jsonify({'message': 'File not found on disk'}), 404

    try:
        data = request.get_json()
        message = data.get('message')
        if not message:
            return jsonify({'message': 'Message is required'}), 400

        # Run retrieval + generation
        try:
            res = process_with_rag(file_id, message)
        except Exception as e:
            # If RAG fails, fallback to direct model call (safer than total failure)
            app.logger.exception('RAG retrieval failed, falling back to LLM only')
            res = model_LLM(message)

        return jsonify({
            'id': file_record.id,
            'filename': file_record.filename,
            'filepath': file_record.filepath,
            'user_id': file_record.user_id,
            'content': res
        }), 200

    except Exception as e:
        app.logger.exception('Error processing message')
        return jsonify({'message': f'Error processing message: {str(e)}'}), 500




# This is key: models must be loaded before create_all
with app.app_context():
    db.create_all()

# Routes
@app.route('/auth/register', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    from models import User  # <- also import here if needed

    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'User already exists'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User created successfully'}), 201

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    from models import User  # <- same here

    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({'message': 'Invalid credentials'}), 401

    access_token = create_access_token(identity=email)
    user_data = {
        "email":user.email
    }
    return jsonify({'access_token': access_token, "user_data":user_data}), 200



@app.route('/upload', methods=['POST'])
@jwt_required()
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400

    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()
    if not user:
        return jsonify({'message': 'User not found'}), 404

    filename = secure_filename(file.filename)
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user.email)
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, filename)
    file.save(file_path)

    # Save file info in database
    new_file = File(filename=filename, filepath=file_path, user_id=user.id)
    db.session.add(new_file)
    db.session.commit()


    # Index the file into Chroma
    try:
        index_info = index_pdf(file_path, new_file.id)
        app.logger.info(f"Indexed file into collection {index_info['collection']} with {index_info['num_chunks']} chunks.")
    except Exception as e:
        app.logger.exception(f"Failed to index PDF: {str(e)}")

    return jsonify({'message': 'File uploaded and saved to DB', 'filename': filename}), 201



@app.route('/my-files', methods=['GET'])
@jwt_required()
def get_user_files():
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    # Get all files for the current user from the database
    user_files = File.query.filter_by(user_id=user.id).all()

    # Return list of files with their details
    files_data = [
        {
            'id': file.id,
            'filename': file.filename,
            'filepath': file.filepath,
            'user_id': file.user_id
        }
        for file in user_files
    ]

    return jsonify({'files': files_data}), 200



@app.route('/read-file/<int:file_id>', methods=['GET'])
@jwt_required()
def read_pdf_file(file_id):
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    # Find file in DB by ID and user
    
    file_record = File.query.filter_by(id=file_id, user_id=user.id).first()
    if not file_record:
        return jsonify({'message': 'File not found in database'}), 404

    file_path = file_record.filepath

    if not os.path.isfile(file_path):
        return jsonify({'message': 'File not found on disk'}), 404

    try:
        if not file_record.filename.lower().endswith('.pdf'):
            return jsonify({'message': 'Only PDF files are supported'}), 400

        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()

        return jsonify({
            'id': file_record.id,
            'filename': file_record.filename,
            'filepath': file_record.filepath,
            'content': text.split('\n')[0]
        }), 200
    except Exception as e:
        return jsonify({'message': f'Error reading PDF: {str(e)}'}), 500





@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify({'message': f'Hello, {current_user}! This is a protected route.'})



@app.route('/delete-file/<int:file_id>', methods=['DELETE'])
@jwt_required()
def delete_file(file_id):
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    # Look up file by id and user_id
    file_record = File.query.filter_by(id=file_id, user_id=user.id).first()
    if not file_record:
        return jsonify({'message': 'File not found in database'}), 404

    file_path = file_record.filepath

    # Check if physical file exists
    if not os.path.isfile(file_path):
        # File doesn't exist physically, but exists in DB - clean up DB
        db.session.delete(file_record)
        db.session.commit()
        return jsonify({'message': 'File not found on disk, removed from database'}), 404

    try:
        # Delete from database first
        db.session.delete(file_record)
        db.session.commit()

        # Delete physical file
        os.remove(file_path)

        return jsonify({
            'message': 'File deleted successfully',
            'file_id': file_id,
            'filename': file_record.filename
        }), 200

    except Exception as e:
        # If physical deletion fails, rollback database changes
        db.session.rollback()
        return jsonify({'message': f'Error deleting file: {str(e)}'}), 500

    

# if __name__ == '__main__':
#     app.run(debug=True)






