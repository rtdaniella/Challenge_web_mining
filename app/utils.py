import streamlit as st
import base64
import json
import requests
import psycopg2
import pdfplumber


# 📌 Fonction pour afficher un PDF dans Streamlit
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_data = f.read()
    b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
    pdf_html = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="620" height="700" type="application/pdf"></iframe>'
    st.markdown(pdf_html, unsafe_allow_html=True)


# 📌 FONCTION DE CONNEXION À LA BASE DE DONNÉES
def connect_to_db():
    try:
        conn = psycopg2.connect(
            dbname="webmining",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432",
        )
        return conn
    except Exception as e:
        print("❌ Erreur de connexion à la base de données :", e)
        return None

# 📌 EXTRACTION DU TEXTE PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

# 📌 APPEL API MISTRAL AVEC JSON STRICT
MISTRAL_API_KEY = "Yp0Uo7Vx4uSJIlc94dj3MA5ME71KpwIR"
API_URL = "https://api.mistral.ai/v1/chat/completions"

def query_mistral(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistral-medium",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        try:
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except json.JSONDecodeError:
            print("⚠️ Erreur : réponse Mistral non valide en JSON.")
            return None
    else:
        print("❌ Erreur API :", response.text)
        return None

# 📌 EXTRACTION DES DONNÉES AVEC FORMAT JSON STRICT
def extract_data(text):
    prompt = f"""
    Voici une lettre de motivation :
    "{text}"

    Formate ta réponse en JSON avec ces champs :
    {{
      "competences": "compétence1", "compétence2",
      "motivations": "motivation1", "motivation2",
      "lieu": "ville"
    }}
    """
    return query_mistral(prompt)

# 📌 INSÉRER DANS POSTGRESQL
def insert_into_db(data):
    if not data:
        print("⚠️ Pas de données à insérer.")
        return

    try:
        conn = connect_to_db()  # Utilisation de la fonction de connexion
        if conn is None:
            return

        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lm (
                id SERIAL PRIMARY KEY,
                competences JSON,
                motivations JSON,
                lieu TEXT
            )
        """)

        cursor.execute("""
            INSERT INTO lm (competences, motivations, lieu)
            VALUES (%s, %s, %s)
        """, (json.dumps(data["competences"]), json.dumps(data["motivations"]), data["lieu"]))

        conn.commit()
        print("✅ Données insérées avec succès !")

    except Exception as e:
        print("❌ Erreur PostgreSQL :", e)

    finally:
        cursor.close()
        conn.close()

# 📌 Fonction pour exécuter tout
def process_and_store_lm(lm_pdf_path):
    # Extraire le texte de la LM
    text = extract_text_from_pdf(lm_pdf_path)

    # Analyser les données avec Mistral
    result = extract_data(text)
    
    # Insérer les données extraites dans la base de données
    if result:
        insert_into_db(result)
        return result
    else:
        return None
