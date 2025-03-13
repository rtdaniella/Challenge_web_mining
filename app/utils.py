import streamlit as st
import base64
import json
import requests
import psycopg2
import pdfplumber
import pandas as pd
import string
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import plotly.express as px
from sklearn.manifold import TSNE


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
    
# 📌 Récupérer toutes les offres d'emploi depuis la base de données
def get_offres_from_db():
    conn = connect_to_db()
    query = "SELECT * FROM annonces;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# 📌 Nettoyage et prétraitement des descriptions

nltk.download('stopwords')
nltk.download('punkt')

# Charger le modèle de langue française de spaCy
nlp = spacy.load("fr_core_news_sm")

# Charger la liste des stopwords français de NLTK et du fichier CSV
stop_words = set(stopwords.words('french'))
stop_Bastin = pd.read_csv("../french_stopwords.csv", sep=";")
stop_words.update(set(stop_Bastin["token"]))

# Ajouter les stopwords personnalisés
custom_stopwords = {
    "klanik", "bosch", "activités", "assure", "permettant", "partenaires", "côtés", "digixart", "amené", 
    "pricing", "rythme", "développée", "précision", "daurade", "exécuter", "vacances", "clients", "données", 
    "data", "expérience", "équipes", "compétences", "mission", "groupe", "solutions", "travail", "salesforce", 
    "techniques", "projets", "métiers", "technique", "outils", "équipe", "poste", "traitement", "an", "agence", 
    "commerciale", "marketing", "différents", "idéalement", "activité", "logiciel", "niveau", "connaissance", 
    "maîtrise", "faciliter", "profil", "participer", "bons", "business", "projet", "manipulation", "rejoignez", 
    "légumes", "fruits", "tickets", "building", "entreprise", "assurer", "métier", "tests", "complexes", 
    "gestion", "capacité", "formation", "répondront", "construction", "outil", "team", "offre", "programme", 
    "sein", "mise", "clarté", "travailler", "manuels", "suivi", "accompagner", "définition", "famille", "direction", 
    "prime", "primes", "utilisateurs", "familiale", "annuelle", "issu", "rémunération"
}
stop_words.update(custom_stopwords)

# Initialiser le stemmer français
stemmer = SnowballStemmer("french")


def preprocess_text(text):
    """
    Nettoie et prétraite un texte donné :
    - Minuscule
    - Suppression des caractères spéciaux
    - Suppression des stopwords
    - Tokenisation
    - Stemming et lemmatisation
    """

    # Normalisation des textes
    text = text.lower()
    text = text.replace("\n", " ")
    
    # Remplacement des termes techniques
    text = text.replace("ci/cd", "ci_cd").replace("power-bi", "power_bi").replace("big.data", "big_data")
    
    # Supprimer la ponctuation sauf les underscores
    text = re.sub(r"[^\w\s_]", " ", text)
    
    # Remettre les termes techniques à leur format initial
    text = text.replace("ci_cd", "ci/cd").replace("power_bi", "power-bi").replace("big_data", "big-data")
    
    # Remplacement de certaines expressions
    text = re.sub(r"gestion de projets?", "gestion-de-projet", text)
    
    # Suppression des underscores et des chiffres isolés
    text = re.sub(r"_", " ", text)
    text = re.sub(r'\b\d+\b', '', text)

    # Supprimer les stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Tokenisation
    tokens = word_tokenize(text)

    # Stemming
    stems = " ".join([stemmer.stem(word) for word in tokens])

    # Lemmatisation avec spaCy
    doc = nlp(text)
    lemmas = " ".join([token.lemma_ for token in doc])

    return text, stems, lemmas

# 📌 Génération d'un nuage de mots basé sur les descriptions de poste
def generate_wordcloud(df):
    # Prétraiter les descriptions (utilisation des lemmes)
    processed_texts = [preprocess_text(desc)[2] for desc in df['description'].dropna()]  # [2] pour les lemmes

    # Joindre tout le texte pour le wordcloud
    text = ' '.join(processed_texts)

    if not text.strip():  # Vérifier si le texte est vide après le preprocessing
        st.warning("⚠️ Pas assez de texte après prétraitement pour générer un nuage de mots.")
        return

    # Générer le WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Convertir en image numpy
    img_array = wordcloud.to_array()

    # Créer la figure Plotly
    fig = px.imshow(img_array)
    fig.update_layout(
        title="☁️ Nuage de Mots des Descriptions de Postes",
        coloraxis_showscale=False
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Afficher dans Streamlit
    st.plotly_chart(fig)

# 📌 Analyse de la distribution de l'expérience requise
def plot_experience_distribution(df):
    # Vérifier si la colonne 'experience' contient des données valides
    if df['experience'].dropna().empty:
        st.warning("Aucune donnée disponible pour la distribution des niveaux d'expérience.")
        return

    # Compter le nombre d'offres par niveau d'expérience
    experience_counts = df['experience'].value_counts().reset_index()
    experience_counts.columns = ['Niveau d\'expérience', 'Nombre d\'offres']

    # Création du graphique interactif avec Plotly
    fig = px.bar(
        experience_counts,
        x='Niveau d\'expérience',
        y='Nombre d\'offres',
        text='Nombre d\'offres',
        title="📊 Distribution des Niveaux d'Expérience dans les Offres",
        color='Nombre d\'offres',
        color_continuous_scale='Blues'
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_title="Niveau d'expérience", yaxis_title="Nombre d'offres", showlegend=False)

    st.plotly_chart(fig)  # Affichage du graphique dans Streamlit

# 📌 Clustering des offres avec TF-IDF et KMeans
def cluster_offers(df, n_clusters=5):
    # Téléchargement des stopwords en français
    nltk.download('stopwords')
    french_stopwords = stopwords.words('french')

    # Initialisation du vectoriseur TF-IDF
    vectorizer = TfidfVectorizer(stop_words=french_stopwords, max_features=500)

    # Appliquer preprocess_text et extraire seulement le texte nettoyé (premier élément du tuple)
    X = vectorizer.fit_transform(df['description'].dropna().apply(lambda text: preprocess_text(text)[0]))

    # Appliquer le clustering KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    # Ajouter un mapping pour afficher le nom du cluster
    cluster_names = {i: f"Cluster {i+1}" for i in range(n_clusters)}
    df['cluster_name'] = df['cluster'].map(cluster_names)

    # Afficher les termes les plus importants pour chaque cluster
    cluster_centers = kmeans.cluster_centers_
    feature_names = vectorizer.get_feature_names_out()

    clusters_terms = {}
    for i, center in enumerate(cluster_centers):
        top_terms_idx = center.argsort()[-10:][::-1]  # Obtenir les indices des 10 termes les plus importants
        top_terms = [feature_names[idx] for idx in top_terms_idx]
        clusters_terms[f"Cluster {i+1}"] = top_terms

    # Retourner le DataFrame avec les clusters et les termes associés à chaque cluster
    return df[['reference', 'intitule_poste', 'cluster_name']], clusters_terms, X

# 📌 Clustering visualisation
def plot_tsne(X, labels, title="t-SNE Visualisation des Clusters"):
    # Appliquer t-SNE pour réduire la dimensionnalité
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(X.toarray())  # Convertir en tableau dense si nécessaire

    # Créer un DataFrame pour la visualisation avec Plotly
    df_tsne = pd.DataFrame(reduced_data, columns=['x', 'y'])
    df_tsne['cluster'] = labels

    # Créer le graphique avec Plotly Express
    fig = px.scatter(df_tsne, x='x', y='y', color='cluster', title=title,
                     labels={'x': 't-SNE Axe 1', 'y': 't-SNE Axe 2', 'cluster': 'Cluster'},
                     color_continuous_scale='Viridis')

    fig.update_layout(title=title, title_x=0.5, template='plotly_dark')

    return fig
