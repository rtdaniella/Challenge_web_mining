import streamlit as st
from utils import show_pdf, process_and_store_lm, get_offres_from_db, generate_wordcloud, plot_experience_distribution, cluster_offers, plot_tsne
import time
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(layout="wide")

# Titre de la page
st.markdown('<div class="title">🔍 Recrutement IA : Trouvez le Candidat Idéal !</div>', unsafe_allow_html=True)

# Style CSS personnalisé
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
        background-color: #ffff4d;
        font-size: 16px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 24px;
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

    .result-card {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .result-card h3 {
        color: #000066;
        font-size: 20px;
        margin-bottom: 10px;
    }
    .result-card p {
        color: #333;
        font-size: 16px;
    }
    .section-title {
        color: #2c3e50;
        font-weight: bold;
        font-size: 18px;
        margin-top: 15px;
    }
    .highlight {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Création des onglets
tabs = st.tabs(["💼 Offre d'Emploi", "📄 Candidatures", "🤝 Matching"])

# Contenu pour l'onglet 1 : Offre d'emploi
with tabs[0]:
    st.subheader("🔎 Explorez les Offres d'Emploi Disponibles")
    st.write("Consultez et recherchez des offres d'emploi provenant de France Travail.")

    # Récupérer et afficher les offres d'emploi
    df_offres = get_offres_from_db()

    # Utiliser st.columns pour avoir deux filtres horizontaux
    col1, col2 = st.columns(2)

    # Filtre pour le niveau d'expérience dans la première colonne
    with col1:
        selected_experience = st.selectbox("Sélectionnez le niveau d'expérience", ["Tous"] + list(df_offres['experience'].unique()))

    # Filtre pour l'intitulé de poste dans la deuxième colonne
    with col2:
        selected_intitule = st.selectbox("Sélectionnez l'intitulé du poste", ["Tous"] + list(df_offres['intitule_poste'].unique()))

    # Appliquer les filtres si l'utilisateur en sélectionne
    if selected_experience != "Tous":
        df_offres = df_offres[df_offres['experience'] == selected_experience]

    if selected_intitule != "Tous":
        df_offres = df_offres[df_offres['intitule_poste'] == selected_intitule]

    # Afficher le tableau interactif des offres filtrées
    st.write(f"Affichage des offres : Niveau d'expérience '{selected_experience}' et intitulé de poste '{selected_intitule}'")

    # Construire les options pour le tableau interactif
    gb = GridOptionsBuilder.from_dataframe(df_offres)
    gb.configure_pagination(paginationPageSize=10)  # Permet la pagination
    gb.configure_default_column(
        filterable=True,  # Ajouter des filtres
        sortable=True,  # Rendre les colonnes triables
        resizable=True  # Rendre les colonnes redimensionnables
    )
    grid_options = gb.build()

    # Afficher le tableau interactif
    AgGrid(df_offres, gridOptions=grid_options, height=400, width='100%')

    
    if st.button("Nuage de Mots"):
        st.subheader("🔍 Nuage de mots des descriptions de postes :")
        generate_wordcloud(df_offres)

    if st.button("Distribution des niveaux d'expérience"):
        st.subheader("📈 Distribution des niveaux d'expérience demandés :")
        plot_experience_distribution(df_offres)

    if st.button("Clustering des Offres"):
        # Appeler la fonction cluster_offers et récupérer les valeurs
        df_clusters, clusters_terms, X = cluster_offers(df_offres)

        # Afficher le DataFrame avec les clusters
        st.subheader("Tableau des Offres avec leurs clusters:")
        st.write(df_clusters)

        # Afficher les termes les plus importants pour chaque cluster
        st.subheader("Termes les plus importants pour chaque cluster")

        for cluster, terms in clusters_terms.items():
            # Utiliser une couleur et une typographie plus modernes
            # Utiliser flexbox pour aligner le titre et les termes sur la même ligne
            st.markdown(f"""
                <div style="padding: 10px; margin: 10px 0; background-color: #f0f8ff; border-radius: 10px; display: flex; align-items: center;">
                    <h4 style="color: #006400; font-size: 16px; margin-right: 15px;">💡 <strong>{cluster} :</strong></h4>
                    <p style="color: #2f4f4f; font-size: 16px; margin: 0;">{', '.join(terms)}</p>
                </div>
            """, unsafe_allow_html=True)

        # Appliquer t-SNE et afficher le graphique interactif avec Plotly
        st.subheader("Visualisation des clusters avec t-SNE:")
        fig = plot_tsne(X, df_clusters['cluster_name'].values)  # Passer X et les labels des clusters
        st.plotly_chart(fig)














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
                    time.sleep(0.10)  # Pause pour simuler un processus long
                    progress_bar.progress(i)  # Mise à jour de la barre de progression
                    
                    # Une fois la barre à 100%, lance le processus d'analyse
                    if i == 100:
                        result = process_and_store_lm("lm_temp.pdf")
                
                if result:
                    st.write("Données extraites :")
                    
                    # Organiser les informations extraites sous forme de cartes stylisées

                    st.markdown(f'<h3>🏷️ Compétences</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p>Compétences mentionnées : <span class="highlight">{result["competences"]}</span></p>', unsafe_allow_html=True)

                    st.markdown(f'<h3>🌍 Localisation</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p>Lieu de travail souhaité : <span class="highlight">{result["lieu"]}</span></p>', unsafe_allow_html=True)

                    st.markdown(f'<h3>💬 Motivations</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p>Motivations mentionnées : <span class="highlight">{result["motivations"]}</p>', unsafe_allow_html=True)


                    st.success("Analyse terminée et données insérées dans la base.")
                else:
                    st.error("Erreur lors de l'analyse des données.")
    st.write("Allez dans l'onglet Matching pour trouvez une offre")
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
