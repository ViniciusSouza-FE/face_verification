import requests
import json

def test_recognition():
    """Testa o reconhecimento facial"""
    url = "http://localhost:5000/api/recognize_upload"
    
    with open('test_face.jpg', 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
        
    print("Status:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))

def check_database():
    """Verifica o que está no banco"""
    import psycopg2
    import pickle
    import numpy as np
    
    conn = psycopg2.connect('postgresql://neondb_owner:npg_xXbEUf0y1Gir@ep-old-glitter-a4sg9zar-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require')
    cursor = conn.cursor()
    
    # Ver pessoas cadastradas
    cursor.execute('SELECT id, nome, embedding IS NOT NULL as has_embedding FROM pessoas WHERE ativo = true')
    pessoas = cursor.fetchall()
    
    print("Pessoas no banco:")
    for pessoa in pessoas:
        print(f"  ID: {pessoa[0]}, Nome: {pessoa[1]}, Tem embedding: {pessoa[2]}")
    
    # Ver embeddings
    cursor.execute('SELECT id, nome, embedding FROM pessoas WHERE embedding IS NOT NULL')
    embeddings = cursor.fetchall()
    
    print("\nEmbeddings:")
    for emb in embeddings:
        embedding_array = pickle.loads(emb[2])
        print(f"  {emb[1]}: shape {embedding_array.shape}, norm {np.linalg.norm(embedding_array):.4f}")
    
    conn.close()

if __name__ == '__main__':
    print("=== Verificando banco ===")
    check_database()
    print("\n=== Testando reconhecimento ===")
    test_recognition()