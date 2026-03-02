# Sistema de reconhecimento facial.

Projeto desenvolvido para a disciplina de Projeto Interdisciplinar II, com o objetivo de criar um agente de Inteligência Artificial capaz de realizar o reconhecimento facial e confirmação de identidade.
O sistema permite cadastrar pessoas com foto e posteriormente confirmar a identidade via upload de imagem ou câmera.

##Tecnologias Utilizadas
# Backend
- Python
- Flask
- DeepFace
- OpenCV
- Pillow
- Psycopg2
- PostgreSQL

# Frontend
- HTML5
- CSS3
- JavaScript
- Chart.js
- Font Awesome

# Banco de Dados
- PostgreSQL

# Bibliotecas da IA
- DeepFace (extração de embeddings faciais)
- TensorFlow (backend utilizado pelo DeepFace)

##IA
O sistema utiliza a biblioteca `DeepFace` para:
- Detectar rostos
- Extrair embeddings faciais
- Comparar similaridade entre imagens
- Retornar nível de confiança (%) da identificação

## Funcionalidades
# Cadastro de Pessoas
- Nome
- Documento
- Email (opcional)
- Telefone (opcional)
- Foto para extração de embedding

# Confirmação Facial
- Upload de imagem
- Captura via câmera
- Retorno de percentual de confiança
- Exibição dos dados da pessoa reconhecida

# Estatísticas
- Total de pessoas cadastradas
- Total de reconhecimentos
- Reconhecimentos por método (câmera/upload)
- Gráfico dos últimos 7 dias

#  Gerenciamento
- Listagem de pessoas cadastradas
- Exclusão de registros

## Deploy
O projeto já esteve hospedado anteriormente em: Railway (backend) e Neon (banco de dados PostgreSQL).
Atualmente não está em produção.

## Desenvolvido por
- Vinicius Souza
- Rafael Soares
