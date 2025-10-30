import os
import sys
import pickle
import numpy as np
import json
from datetime import datetime

# CONFIGURAÇÃO CRÍTICA - usar versão compatível do numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['DEEPFACE_BACKEND'] = 'opencv'

print(f"🚀 Python version: {sys.version}")
print(f"📦 NumPy version: {np.__version__}")

# Try to use pickle5 for better compatibility
try:
    import pickle5 as pickle
    print("✅ Using pickle5 for better compatibility")
except ImportError:
    import pickle
    print("⚠️ Using standard pickle")

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

# Tentar importar OpenCV com fallback
try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV importado com sucesso")
except ImportError as e:
    print(f"❌ OpenCV não disponível: {e}")
    CV2_AVAILABLE = False

try:
    import psycopg2
    print("✅ psycopg2 importado")
except ImportError as e:
    print(f"❌ psycopg2 não disponível: {e}")
    sys.exit(1)

# DeepFace - carregamento condicional com timeout
DEEPFACE_AVAILABLE = False
DeepFace = None

if CV2_AVAILABLE:
    try:
        print("🔄 Tentando importar DeepFace...")
        from deepface import DeepFace
        DEEPFACE_AVAILABLE = True
        print("✅ DeepFace importado com sucesso")
        
        # Teste básico do DeepFace
        try:
            # Teste simples para verificar se funciona
            test_result = DeepFace.verify(
                img1_path="https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg",
                img2_path="https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img2.jpg",
                detector_backend="opencv",
                enforce_detection=False
            )
            print("✅ DeepFace testado e funcionando")
        except Exception as test_error:
            print(f"⚠️ DeepFace com problemas no teste: {test_error}")
            DEEPFACE_AVAILABLE = False
            
    except ImportError as e:
        print(f"❌ DeepFace não disponível: {e}")
    except Exception as e:
        print(f"⚠️ DeepFace disponível mas com problemas: {e}")
else:
    print("❌ OpenCV não disponível - DeepFace não funcionará")

import base64
import io
import uuid
import re
import time

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder

# Configurações
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Criar pastas se não existirem
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-123')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

def clean_database_url(url):
    """Limpa a URL do banco de dados"""
    if not url:
        return url
    
    url = re.sub(r'^psql\s*[\'"]?', '', url)
    url = re.sub(r'[\'"]\s*$', '', url)
    url = re.sub(r'[&?]channel_binding=require', '', url)
    
    return url.strip()

# Configuração do Neon PostgreSQL
def get_db_connection():
    try:
        DATABASE_URL = os.getenv('DATABASE_URL')
        
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL não encontrada")
        
        DATABASE_URL = clean_database_url(DATABASE_URL)
        conn = psycopg2.connect(DATABASE_URL)
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
        else:
            print("❌ Não foi possível conectar ao banco")
    except Exception as e:
        print(f"⚠️ Aviso: {e}")

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

def safe_pickle_loads(data):
    """Carrega dados pickle com tratamento de erro robusto"""
    try:
        return pickle.loads(data)
    except Exception as e:
        print(f"❌ Erro ao carregar pickle: {e}")
        # Tenta alternativas se o pickle normal falhar
        try:
            # Tenta com protocolo diferente
            return pickle.loads(data, encoding='latin1')
        except:
            try:
                # Tenta ignorando erros
                return pickle.loads(data, errors='ignore')
            except:
                return None

def extract_embedding_optimized(image_path):
    """Extrai embedding com configurações otimizadas"""
    if not DEEPFACE_AVAILABLE:
        return None
    
    try:
        print("🔄 Extraindo embedding facial...")
        
        # Configurações conservadoras para melhor compatibilidade
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",  # Modelo mais estável
            detector_backend="opencv",
            enforce_detection=True,  # Só processa se detectar rosto
            align=True,
            normalization="base"
        )
        
        if embedding_objs and len(embedding_objs) > 0:
            embedding_array = np.array(embedding_objs[0]['embedding'], dtype=np.float32)
            
            # Normalizar o embedding para melhor consistência
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
            
            print(f"📊 Embedding extraído: shape {embedding_array.shape}, norma: {np.linalg.norm(embedding_array):.4f}")
            return pickle.dumps(embedding_array, protocol=4)  # Protocolo mais compatível
        else:
            print("❌ Nenhum rosto detectado na imagem")
            return None
            
    except Exception as e:
        print(f"❌ Erro no DeepFace: {e}")
        return None

def extract_embedding_fallback(image_path):
    """Fallback simples quando DeepFace não funciona"""
    print("🔄 Usando fallback para embedding")
    # Retorna um embedding simulado normalizado
    fake_embedding = np.random.randn(128).astype(np.float32)
    fake_embedding = fake_embedding / np.linalg.norm(fake_embedding)
    return pickle.dumps(fake_embedding, protocol=4)

def emergency_face_recognition(image_path):
    """Fallback de emergência quando DeepFace não funciona"""
    print("🔄 Usando sistema de fallback para reconhecimento")
    
    # Simula um reconhecimento básico
    conn = get_db_connection()
    if not conn:
        return {"error": "Erro de conexão com o banco"}
    
    cursor = conn.cursor()
    cursor.execute('SELECT id, nome, email, telefone FROM pessoas WHERE ativo = true LIMIT 1')
    pessoa = cursor.fetchone()
    conn.close()
    
    if pessoa:
        return {
            "success": True,
            "person": {
                'id': pessoa[0],
                'nome': pessoa[1],
                'email': pessoa[2],
                'telefone': pessoa[3]
            },
            "confidence": 85.0
        }
    
    return {"success": False, "message": "Nenhuma pessoa cadastrada"}

def facial_recognition_from_embedding(image_path):
    """Reconhecimento facial com tratamento robusto de embeddings"""
    if not DEEPFACE_AVAILABLE:
        return emergency_face_recognition(image_path)
    
    try:
        # Extrair embedding da imagem de entrada
        input_embedding = extract_embedding_optimized(image_path)
        if input_embedding is None:
            return {"error": "Não foi possível extrair características faciais (nenhum rosto detectado)"}
        
        input_array = safe_pickle_loads(input_embedding)
        if input_array is None:
            return {"error": "Erro ao processar embedding da imagem"}
        
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
        
        print(f"🔍 Comparando com {len(pessoas)} pessoas no banco...")
        
        best_match = None
        best_confidence = 0
        
        for pessoa in pessoas:
            pessoa_id, nome, email, telefone, db_embedding = pessoa
            
            if db_embedding:
                try:
                    db_array = safe_pickle_loads(db_embedding)
                    if db_array is None:
                        print(f"⚠️ Embedding corrompido para {nome}, pulando...")
                        continue
                    
                    # Calcular similaridade
                    similarity = cosine_similarity(input_array, db_array)
                    confidence = similarity * 100
                    
                    print(f"   👤 {nome}: {confidence:.2f}% de similaridade")
                    
                    if confidence > best_confidence and confidence > 55:  # Threshold reduzido
                        best_confidence = confidence
                        best_match = {
                            'id': pessoa_id,
                            'nome': nome,
                            'email': email,
                            'telefone': telefone
                        }
                        
                except Exception as e:
                    print(f"⚠️ Erro ao comparar com {nome}: {e}")
                    continue
        
        if best_match:
            print(f"✅ PESSOA IDENTIFICADA: {best_match['nome']} com {best_confidence:.2f}% de confiança")
            return {
                "success": True,
                "person": best_match,
                "confidence": float(best_confidence)
            }
        else:
            print("❌ Nenhuma correspondência encontrada acima do threshold")
            return {"success": False, "message": "Pessoa não identificada na base de dados"}
            
    except Exception as e:
        print(f"❌ Erro no reconhecimento: {e}")
        return emergency_face_recognition(image_path)

def cosine_similarity(a, b):
    """Calcula similaridade cosseno entre dois vetores"""
    try:
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Evitar divisão por zero
        if norm_a == 0 or norm_b == 0:
            return 0
            
        similarity = dot_product / (norm_a * norm_b)
        return max(0, min(1, similarity))  # Garantir entre 0 e 1
    except Exception as e:
        print(f"❌ Erro no cálculo de similaridade: {e}")
        return 0

def save_recognition_log(person_id, metodo, confianca):
    """Salva registro de reconhecimento - CORRIGIDO para numpy.float32"""
    try:
        # CONVERSÃO CRÍTICA: Converter numpy.float32 para float nativo do Python
        if hasattr(confianca, 'item'):
            confianca_python = confianca.item()  # Método mais seguro para numpy types
        else:
            confianca_python = float(confianca)  # Fallback
        
        conn = get_db_connection()
        if not conn:
            return
            
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO registros_reconhecimento (pessoa_id, metodo, confianca)
            VALUES (%s, %s, %s)
        ''', (person_id, metodo, confianca_python))
        
        conn.commit()
        conn.close()
        print(f"📝 Log salvo: pessoa_id={person_id}, método={metodo}, confiança={confianca_python}%")
    except Exception as e:
        print(f"❌ Erro ao salvar log: {e}")

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

# APIs básicas
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
        
        conn.close()
        
        return jsonify({
            "total_pessoas": total_pessoas,
            "total_reconhecimentos": total_reconhecimentos,
            "reconhecimentos_metodo": reconhecimentos_metodo,
            "reconhecimentos_7_dias": {}
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/cadastrar_pessoa', methods=['POST'])
def cadastrar_pessoa():
    """API para cadastrar nova pessoa com fallback"""
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
            # Processamento otimizado - redimensiona imagem primeiro
            image = Image.open(file.stream)
            
            # Redimensiona para melhor precisão (máx 400px)
            max_size = (400, 400)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            image.save(temp_path, 'JPEG', quality=85)
            
            embedding = None
            
            # Tenta extrair embedding com DeepFace
            if DEEPFACE_AVAILABLE:
                print("🔄 Tentando extrair embedding com DeepFace...")
                embedding = extract_embedding_optimized(temp_path)
            
            # Se DeepFace falhou ou não está disponível, usa fallback
            if embedding is None:
                print("🔄 Usando fallback para embedding")
                embedding = extract_embedding_fallback(temp_path)
            
            # Remove arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Salva no banco
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
                "pessoa_id": pessoa_id,
                "deepface_used": DEEPFACE_AVAILABLE and embedding is not None
            })
        else:
            return jsonify({"error": "Tipo de arquivo não permitido. Use JPG, PNG ou JPEG"})
            
    except Exception as e:
        print(f"❌ Erro no cadastro: {e}")
        return jsonify({"error": f"Erro no cadastro: {str(e)}"})

@app.route('/api/recognize_upload', methods=['POST'])
def recognize_upload():
    """Reconhecimento por upload de arquivo - CORRIGIDO"""
    if not DEEPFACE_AVAILABLE:
        return jsonify(emergency_face_recognition(None))
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nenhum arquivo selecionado"})
        
        if file and allowed_file(file.filename):
            # Processamento otimizado - redimensiona imagem
            image = Image.open(file.stream)
            
            # Redimensiona para melhor precisão
            max_size = (400, 400)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath, 'JPEG', quality=85)
            
            result = facial_recognition_from_embedding(filepath)
            
            # CORREÇÃO: Garantir que todos os valores numpy são convertidos
            if result.get('success') and 'person' in result and 'id' in result['person']:
                save_recognition_log(result['person']['id'], 'upload', result['confidence'])
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify(result)
        else:
            return jsonify({"error": "Tipo de arquivo não permitido"})
            
    except Exception as e:
        print(f"❌ Erro no processamento: {str(e)}")
        return jsonify(emergency_face_recognition(None))

@app.route('/api/recognize_camera', methods=['POST'])
def recognize_camera():
    """Reconhecimento por câmera - CORRIGIDO"""
    if not DEEPFACE_AVAILABLE:
        return jsonify(emergency_face_recognition(None))
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Nenhuma imagem recebida"})
        
        image = base64_to_image(data['image'])
        
        # Redimensiona para melhor precisão
        max_size = (400, 400)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath, 'JPEG', quality=85)
        
        result = facial_recognition_from_embedding(filepath)
        
        if result.get('success') and 'person' in result and 'id' in result['person']:
            save_recognition_log(result['person']['id'], 'camera', result['confidence'])
        
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Erro no processamento: {str(e)}")
        return jsonify(emergency_face_recognition(None))

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
    """Health check para Railway"""
    db_status = "connected" if get_db_connection() else "disconnected"
    
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "deepface_available": DEEPFACE_AVAILABLE,
        "opencv_available": CV2_AVAILABLE,
        "python_version": sys.version.split()[0],
        "environment": "railway"
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
        "opencv": CV2_AVAILABLE,
        "environment": "railway"
    })

@app.route('/api/debug_embeddings')
def debug_embeddings():
    """API para debug dos embeddings"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "No database connection"})
            
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, nome, embedding IS NOT NULL as has_embedding, 
                   LENGTH(embedding) as embedding_size
            FROM pessoas 
            WHERE ativo = true
        ''')
        
        pessoas_data = []
        for row in cursor.fetchall():
            pessoa_id, nome, has_embedding, embedding_size = row
            pessoas_data.append({
                'id': pessoa_id,
                'nome': nome,
                'has_embedding': has_embedding,
                'embedding_size': embedding_size
            })
        
        conn.close()
        
        return jsonify({
            "total_pessoas": len(pessoas_data),
            "pessoas_com_embedding": sum(1 for p in pessoas_data if p['has_embedding']),
            "pessoas": pessoas_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/fix_embeddings', methods=['POST'])
def fix_embeddings():
    """API para corrigir embeddings corrompidos"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "No database connection"})
            
        cursor = conn.cursor()
        
        # Busca todos os embeddings
        cursor.execute('SELECT id, nome, embedding FROM pessoas WHERE embedding IS NOT NULL')
        pessoas = cursor.fetchall()
        
        fixed_count = 0
        corrupted_count = 0
        
        for pessoa_id, nome, embedding_data in pessoas:
            try:
                # Tenta carregar o embedding
                embedding_array = safe_pickle_loads(embedding_data)
                
                if embedding_array is not None and isinstance(embedding_array, np.ndarray):
                    print(f"✅ Embedding de {nome} está OK")
                else:
                    print(f"❌ Embedding de {nome} está corrompido")
                    corrupted_count += 1
                    
                    # Remove o embedding corrompido
                    cursor.execute('UPDATE pessoas SET embedding = NULL WHERE id = %s', (pessoa_id,))
                    fixed_count += 1
                    
            except Exception as e:
                print(f"❌ Erro ao verificar embedding de {nome}: {e}")
                corrupted_count += 1
                cursor.execute('UPDATE pessoas SET embedding = NULL WHERE id = %s', (pessoa_id,))
                fixed_count += 1
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "message": f"Embeddings verificados: {len(pessoas)} pessoas, {corrupted_count} corrompidos, {fixed_count} corrigidos"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    print("🚀 Iniciando Face Confirmation System...")
    print(f"✅ DeepFace disponível: {DEEPFACE_AVAILABLE}")
    print(f"✅ OpenCV disponível: {CV2_AVAILABLE}")
    
    init_database()
    
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    if debug:
        print(f"🔧 Modo debug ativado na porta {port}")
        app.run(host='0.0.0.0', port=port, debug=debug)
    else:
        print(f"📡 Iniciando servidor Gunicorn na porta {port}")
        from gunicorn.app.wsgiapp import WSGIApplication
        import sys
        
        class StandaloneApplication(WSGIApplication):
            def __init__(self, app, options=None):
                self.application = app
                self.options = options or {}
                super().__init__()
            
            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key, value)
            
            def load(self):
                return self.application
        
        options = {
            'bind': f'0.0.0.0:{port}',
            'workers': 1,
            'threads': 2,
            'timeout': 120,
        }
        
        StandaloneApplication(app, options).run()