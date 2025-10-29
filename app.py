import os
import sys
import pickle
import numpy as np

# Configurações de ambiente para sistemas headless
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print(f"🚀 Python version: {sys.version}")

# Verificar importações críticas
try:
    from flask import Flask, render_template, request, jsonify
    print("✅ Flask importado")
except ImportError as e:
    print(f"❌ Flask não disponível: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print("✅ Pillow importado")
except ImportError as e:
    print(f"❌ Pillow não disponível: {e}")
    sys.exit(1)

try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV importado")
except ImportError as e:
    print(f"❌ OpenCV não disponível: {e}")
    CV2_AVAILABLE = False

try:
    import psycopg2
    from psycopg2 import sql
    print("✅ psycopg2 importado")
except ImportError as e:
    print(f"❌ psycopg2 não disponível: {e}")
    sys.exit(1)

# Tenta importar DeepFace apenas se OpenCV estiver disponível
if CV2_AVAILABLE:
    try:
        from deepface import DeepFace
        DEEPFACE_AVAILABLE = True
        print("✅ DeepFace importado")
    except ImportError as e:
        print(f"❌ DeepFace não disponível: {e}")
        DEEPFACE_AVAILABLE = False
else:
    DEEPFACE_AVAILABLE = False

import base64
import io
import uuid
from datetime import datetime
import re

app = Flask(__name__)

# Configurações
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Criar pastas se não existirem
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-123')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

def clean_database_url(url):
    """Limpa a URL do banco de dados removendo 'psql' e aspas"""
    if not url:
        return url
    
    # Remove o comando psql e aspas se presentes
    url = re.sub(r'^psql\s*[\'"]?', '', url)
    url = re.sub(r'[\'"]\s*$', '', url)
    
    # Remove channel_binding parameter se existir
    url = re.sub(r'[&?]channel_binding=require', '', url)
    
    return url.strip()

# Configuração do Neon PostgreSQL
def get_db_connection():
    try:
        DATABASE_URL = os.getenv('DATABASE_URL')
        print(f"🔗 DATABASE_URL: {DATABASE_URL[:50]}...")  # Log parcial para segurança
        
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL não encontrada")
        
        # Limpa a URL se necessário
        DATABASE_URL = clean_database_url(DATABASE_URL)
        print(f"🔗 DATABASE_URL limpa: {DATABASE_URL[:50]}...")
        
        conn = psycopg2.connect(DATABASE_URL)
        print("✅ Conexão com o banco estabelecida!")
        return conn
    except Exception as e:
        print(f"❌ Erro na conexão com o banco: {e}")
        return None

def init_database():
    """Inicializa o banco com tabela de embeddings"""
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            
            # Verifica se a tabela pessoas existe
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pessoas (
                    id SERIAL PRIMARY KEY,
                    nome VARCHAR(255) NOT NULL,
                    email VARCHAR(255),
                    telefone VARCHAR(50),
                    embedding BYTEA,
                    data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ativo BOOLEAN DEFAULT true
                )
            """)
            
            # Verifica se a tabela registros_reconhecimento existe
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS registros_reconhecimento (
                    id SERIAL PRIMARY KEY,
                    pessoa_id INTEGER REFERENCES pessoas(id),
                    metodo VARCHAR(50),
                    confianca DECIMAL(5,2),
                    data_reconhecimento TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            print("✅ Banco de dados inicializado com sucesso!")
            
            # Insere dados de exemplo se estiver vazio
            insert_sample_data()
        else:
            print("❌ Não foi possível conectar ao banco")
    except Exception as e:
        print(f"⚠️ Aviso na inicialização do banco: {e}")

def insert_sample_data():
    """Insere dados de exemplo para teste"""
    try:
        conn = get_db_connection()
        if not conn:
            return
            
        cursor = conn.cursor()
        
        # Verifica se já existem pessoas
        cursor.execute("SELECT COUNT(*) FROM pessoas")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("📝 Inserindo dados de exemplo...")
            cursor.execute("""
                INSERT INTO pessoas (nome, email, telefone) VALUES 
                ('João Silva', 'joao.silva@email.com', '(11) 99999-9999'),
                ('Maria Santos', 'maria.santos@email.com', '(11) 88888-8888')
            """)
            
            cursor.execute("""
                INSERT INTO registros_reconhecimento (pessoa_id, metodo, confianca) VALUES 
                (1, 'upload', 95.50),
                (2, 'camera', 88.75)
            """)
            
            conn.commit()
            print("✅ Dados de exemplo inseridos!")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ Erro ao inserir dados de exemplo: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def base64_to_image(base64_string):
    """Converte string base64 para imagem PIL"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        raise ValueError(f"Erro ao converter imagem: {str(e)}")

def extract_embedding(image_path):
    """Extrai embedding facial usando DeepFace"""
    if not DEEPFACE_AVAILABLE:
        return None
    
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )
        
        if embedding_objs:
            embedding_array = np.array(embedding_objs[0]['embedding'])
            return pickle.dumps(embedding_array)
        return None
        
    except Exception as e:
        print(f"Erro ao extrair embedding: {e}")
        return None

def facial_recognition_from_embedding(image_path):
    """Realiza reconhecimento facial comparando embeddings"""
    if not DEEPFACE_AVAILABLE:
        return {"error": "Sistema de reconhecimento não disponível"}
    
    try:
        input_embedding = extract_embedding(image_path)
        if input_embedding is None:
            return {"error": "Não foi possível extrair embedding da imagem"}
        
        input_array = pickle.loads(input_embedding)
        
        conn = get_db_connection()
        if not conn:
            return {"error": "Erro de conexão com o banco"}
            
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, nome, email, telefone, embedding 
            FROM pessoas 
            WHERE ativo = true AND embedding IS NOT NULL
        ''')
        
        pessoas = cursor.fetchall()
        conn.close()
        
        if not pessoas:
            return {"error": "Nenhuma pessoa cadastrada no sistema"}
        
        best_match = None
        best_confidence = 0
        
        for pessoa in pessoas:
            pessoa_id, nome, email, telefone, db_embedding = pessoa
            
            if db_embedding:
                db_array = pickle.loads(db_embedding)
                similarity = cosine_similarity(input_array, db_array)
                confidence = similarity * 100
                
                if confidence > best_confidence and confidence > 70:
                    best_confidence = confidence
                    best_match = {
                        'id': pessoa_id,
                        'nome': nome,
                        'email': email,
                        'telefone': telefone
                    }
        
        if best_match:
            return {
                "success": True,
                "person": best_match,
                "confidence": round(best_confidence, 2)
            }
        else:
            return {"success": False, "message": "Pessoa não identificada"}
            
    except Exception as e:
        return {"error": f"Erro no reconhecimento: {str(e)}"}

def cosine_similarity(a, b):
    """Calcula similaridade cosseno entre dois vetores"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def save_recognition_log(person_id, metodo, confianca):
    """Salva registro de reconhecimento"""
    try:
        conn = get_db_connection()
        if not conn:
            return
            
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO registros_reconhecimento (pessoa_id, metodo, confianca)
            VALUES (%s, %s, %s)
        ''', (person_id, metodo, confianca))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erro ao salvar log: {e}")

# Rotas principais
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cadastro')
def cadastro():
    return render_template('cadastro.html')

@app.route('/pessoas')
def pessoas():
    """Lista pessoas cadastradas"""
    try:
        conn = get_db_connection()
        if not conn:
            return render_template('pessoas.html', pessoas=[], error="Erro de conexão com o banco")
            
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, nome, email, telefone, data_cadastro 
            FROM pessoas 
            WHERE ativo = true 
            ORDER BY nome
        ''')
        
        pessoas_data = []
        for row in cursor.fetchall():
            pessoas_data.append({
                'id': row[0],
                'nome': row[1],
                'email': row[2],
                'telefone': row[3],
                'data_cadastro': row[4]
            })
        
        conn.close()
        
        return render_template('pessoas.html', pessoas=pessoas_data)
    except Exception as e:
        return render_template('pessoas.html', pessoas=[], error=str(e))

@app.route('/estatisticas')
def estatisticas():
    """Página de estatísticas"""
    return render_template('estatisticas.html')

# APIs
@app.route('/api/estatisticas', methods=['GET'])
def api_estatisticas():
    """API para estatísticas"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Erro de conexão com o banco"})
            
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM pessoas WHERE ativo = true')
        total_pessoas = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM registros_reconhecimento')
        total_reconhecimentos = cursor.fetchone()[0]
        
        cursor.execute('SELECT metodo, COUNT(*) FROM registros_reconhecimento GROUP BY metodo')
        metodo_data = cursor.fetchall()
        reconhecimentos_metodo = {metodo: count for metodo, count in metodo_data}
        
        cursor.execute('''
            SELECT DATE(data_reconhecimento) as data, COUNT(*) 
            FROM registros_reconhecimento 
            WHERE data_reconhecimento >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY DATE(data_reconhecimento)
            ORDER BY data
        ''')
        timeline_data = cursor.fetchall()
        reconhecimentos_7_dias = {str(data): count for data, count in timeline_data}
        
        conn.close()
        
        return jsonify({
            "total_pessoas": total_pessoas,
            "total_reconhecimentos": total_reconhecimentos,
            "reconhecimentos_metodo": reconhecimentos_metodo,
            "reconhecimentos_7_dias": reconhecimentos_7_dias
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/cadastrar_pessoa', methods=['POST'])
def cadastrar_pessoa():
    """API para cadastrar nova pessoa com embedding"""
    try:
        nome = request.form.get('nome', '').strip()
        email = request.form.get('email', '').strip()
        telefone = request.form.get('telefone', '').strip()
        
        if not nome:
            return jsonify({"error": "Nome é obrigatório"})
        
        if 'foto' not in request.files:
            return jsonify({"error": "Foto é obrigatória"})
        
        file = request.files['foto']
        if file.filename == '':
            return jsonify({"error": "Nenhuma foto selecionada"})
        
        if file and allowed_file(file.filename):
            temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            
            image = Image.open(file.stream)
            image.save(temp_path, 'JPEG')
            
            embedding = extract_embedding(temp_path)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if embedding is None:
                return jsonify({"error": "Não foi possível extrair características faciais da imagem"})
            
            conn = get_db_connection()
            if not conn:
                return jsonify({"error": "Erro de conexão com o banco"})
                
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO pessoas (nome, email, telefone, embedding)
                VALUES (%s, %s, %s, %s) RETURNING id
            ''', (nome, email, telefone, embedding))
            
            pessoa_id = cursor.fetchone()[0]
            conn.commit()
            conn.close()
            
            return jsonify({
                "success": True,
                "message": f"Pessoa {nome} cadastrada com sucesso!",
                "pessoa_id": pessoa_id
            })
        else:
            return jsonify({"error": "Tipo de arquivo não permitido. Use JPG, PNG ou JPEG"})
            
    except Exception as e:
        return jsonify({"error": f"Erro no cadastro: {str(e)}"})

@app.route('/api/recognize_upload', methods=['POST'])
def recognize_upload():
    """Reconhecimento por upload de arquivo"""
    if not DEEPFACE_AVAILABLE:
        return jsonify({"error": "Sistema de reconhecimento facial não disponível no momento"})
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nenhum arquivo selecionado"})
        
        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            image = Image.open(file.stream)
            image.save(filepath, 'JPEG')
            
            result = facial_recognition_from_embedding(filepath)
            
            if result.get('success') and 'person' in result and 'id' in result['person']:
                save_recognition_log(result['person']['id'], 'upload', result['confidence'])
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify(result)
        else:
            return jsonify({"error": "Tipo de arquivo não permitido"})
            
    except Exception as e:
        return jsonify({"error": f"Erro no processamento: {str(e)}"})

@app.route('/api/recognize_camera', methods=['POST'])
def recognize_camera():
    """Reconhecimento por câmera"""
    if not DEEPFACE_AVAILABLE:
        return jsonify({"error": "Sistema de reconhecimento facial não disponível no momento"})
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Nenhuma imagem recebida"})
        
        image = base64_to_image(data['image'])
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath, 'JPEG')
        
        result = facial_recognition_from_embedding(filepath)
        
        if result.get('success') and 'person' in result and 'id' in result['person']:
            save_recognition_log(result['person']['id'], 'camera', result['confidence'])
        
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Erro no processamento: {str(e)}"})

@app.route('/api/pessoas', methods=['GET'])
def api_pessoas():
    """API para listar pessoas"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify([])
            
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, nome, email, telefone, data_cadastro 
            FROM pessoas 
            WHERE ativo = true 
            ORDER BY nome
        ''')
        
        pessoas = []
        for row in cursor.fetchall():
            pessoas.append({
                'id': row[0],
                'nome': row[1],
                'email': row[2],
                'telefone': row[3],
                'data_cadastro': row[4].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        conn.close()
        return jsonify(pessoas)
        
    except Exception as e:
        return jsonify([])

@app.route('/api/deletar_pessoa/<int:pessoa_id>', methods=['DELETE'])
def deletar_pessoa(pessoa_id):
    """API para deletar pessoa"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Erro de conexão com o banco"})
            
        cursor = conn.cursor()
        cursor.execute('UPDATE pessoas SET ativo = false WHERE id = %s', (pessoa_id,))
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "message": "Pessoa removida com sucesso"})
        
    except Exception as e:
        return jsonify({"error": f"Erro ao remover pessoa: {str(e)}"})

@app.route('/health')
def health_check():
    """Health check para Render"""
    db_status = "connected" if get_db_connection() else "disconnected"
    
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "deepface_available": DEEPFACE_AVAILABLE,
        "opencv_available": CV2_AVAILABLE,
        "python_version": sys.version.split()[0]
    })

@app.route('/api/test')
def api_test():
    """API de teste"""
    conn = get_db_connection()
    db_status = "connected" if conn else "disconnected"
    
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pessoas")
            pessoa_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
        except Exception as e:
            pessoa_count = f"error: {e}"
    else:
        pessoa_count = "N/A"
    
    return jsonify({
        "message": "API funcionando!",
        "database": db_status,
        "pessoas_cadastradas": pessoa_count,
        "deepface": DEEPFACE_AVAILABLE,
        "opencv": CV2_AVAILABLE
    })

if __name__ == '__main__':
    print("🚀 Iniciando Face Confirmation System...")
    print(f"✅ DeepFace disponível: {DEEPFACE_AVAILABLE}")
    print(f"✅ OpenCV disponível: {CV2_AVAILABLE}")
    
    init_database()
    
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"📡 Iniciando servidor na porta {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)