import streamlit as st
from utils import show_pdf, process_and_store_lm
import time

st.set_page_config(layout="wide")

# Titre de la page
st.markdown('<div class="title">🔍 Recrutement IA : Trouvez le Candidat Idéal !</div>', unsafe_allow_html=True)

# Style CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f7fa;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 36px;
        color: #000066;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 24px;
    }
    .tab-title {
        font-size: 20px;
        font-weight: bold;
        color: #800080;  /* Changer la couleur des titres des onglets en violet */
    }
    .stTabs [role="tab"] {
        color: black;
        font-weight: bold;
        padding: 10px 20px;

        transition: background 0.3s;
    }
            
    .stTabs [role="tab"][aria-selected="true"] {
        background: #4eb151;
        color: white;
        border-radius: 5px;
    }
            
    .stTabs [role="tab"]:hover {
        background: #46a049;
    }
    .stProgress div[role="progressbar"] {
        background-color: red;  /* Change la couleur de la barre de progression */
    }
    </style>
""", unsafe_allow_html=True)

# Création des onglets
tabs = st.tabs(["💼 Offre d'Emploi", "📄 Candidatures", "🤝 Matching"])

# Contenu pour l'onglet 1 : Offre d'emploi
with tabs[0]:
    st.subheader("🔎 Explorez les Offres d'Emploi Disponibles")
    st.write("Consultez et recherchez des offres d'emploi provenant de France Travail.")
    # Ajoute ici le code pour récupérer et afficher les offres d'emploi depuis la base de données

# Contenu pour l'onglet 2 : Candidatures
with tabs[1]:
    st.subheader("📄 Parcourez les Candidatures")
    st.write("Téléchargez un CV et une Lettre de Motivation pour un candidat.")

    # Créer deux colonnes pour l'affichage horizontal
    col1, col2 = st.columns(2)

    # Téléchargement du CV (format PDF ou image) dans la première colonne
    with col1:
        cv_file = st.file_uploader("Téléchargez votre CV (PDF ou Image)", type=["pdf", "jpg", "jpeg", "png"])
        
        if cv_file is not None:
            with st.expander("Aperçu du CV :"):
                # Si c'est un fichier PDF
                if cv_file.type == "application/pdf":
                    # Sauvegarder le fichier PDF localement temporairement
                    with open("cv_temp.pdf", "wb") as f:
                        f.write(cv_file.read())
                    show_pdf("cv_temp.pdf")
                
                # Si c'est une image
                elif cv_file.type in ["image/jpeg", "image/png", "image/jpg"]:
                    # Afficher l'image téléchargée
                    st.download_button(label="Télécharger le CV", data=cv_file, file_name="cv_image.jpg", mime="image/jpeg")
                    st.image(cv_file, caption="Aperçu de l'image du CV", use_container_width=True)

    # Téléchargement de la LM (format PDF uniquement) dans la deuxième colonne
    with col2:
        lm_file = st.file_uploader("Téléchargez votre Lettre de Motivation (PDF)", type=["pdf"])
        
        if lm_file is not None:
            with st.expander("Aperçu de la Lettre de Motivation :"):
                # Si c'est un fichier PDF
                if lm_file.type == "application/pdf":
                    # Sauvegarder le fichier PDF localement temporairement
                    with open("lm_temp.pdf", "wb") as f:
                        f.write(lm_file.read())
                    show_pdf("lm_temp.pdf")

            # Ajouter un bouton pour démarrer l'analyse
            if st.button("Démarrer l'Analyse"):
                # Initialiser la barre de progression à 0
                progress_bar = st.progress(0)
                
                # Processus complet avec la progression
                result = None
                for i in range(1, 101):  # Simule l'avancement du processus
                    # Tu peux ici ajouter des pauses pour simuler un processus long
                    time.sleep(0.05)  # Pause pour simuler un processus long
                    progress_bar.progress(i)  # Mise à jour de la barre de progression
                    
                    # Une fois la barre à 100%, lance le processus d'analyse
                    if i == 100:
                        result = process_and_store_lm("lm_temp.pdf")
                
                if result:
                    st.write("Données extraites :")
                    st.json(result)  # Affiche les compétences, motivations, et lieu extraits
                    st.success("Analyse terminée et données insérées dans la base.")
                else:
                    st.error("Erreur lors de l'analyse des données.")

# Contenu pour l'onglet 3 : Matching
with tabs[2]:
    st.subheader("🤖 Trouvez le Candidat Idéal pour l'Offre")
    st.write("Faites correspondre le CV et la Lettre de Motivation au poste disponible.")
    if cv_file and lm_file:
        st.button("Lancer le Matching")
        # Ajoute ici la logique de matching entre le CV, la LM et les offres d'emploi
        st.write("Les résultats du matching s'affichent ici.")
    else:
        st.warning("Veuillez télécharger un CV et une Lettre de Motivation pour procéder au matching.")
