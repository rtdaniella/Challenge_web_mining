import streamlit as st
from utils import (
    fix_json_value,
    get_db_connection,
    import_csv_to_cv,
    extract_text_from_pdf,
    query_mistral,
    extract_data,
    insert_into_db,
    process_and_store_lm,
    process_folder,
    show_pdf,
    get_offres_from_db,
    preprocess_text,
    generate_wordcloud,
    plot_experience_distribution,
    cluster_offers,
    plot_tsne,
    generate_offers_embeddings
)
import time
from st_aggrid import AgGrid, GridOptionsBuilder
import logging
import os

st.set_page_config(layout="wide")

# Titre de la page
st.markdown('<div class="title">🎯 Recrutement IA : Trouvez le Candidat Idéal !</div>', unsafe_allow_html=True)
# Fonction pour vérifier si une table existe dans la BDD
def table_exists(table_name):
    conn = get_db_connection()
    if conn is None:
        st.error("Impossible de se connecter à la base de données pour vérifier l'existence de la table.")
        return False
    cursor = conn.cursor()
    query = """
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name = %s
    );
    """
    cursor.execute(query, (table_name,))
    exists = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return exists

# Vérification automatique des tables et import des données si nécessaire
if not table_exists("cv"):
   #st.info("La table 'cv' n'existe pas. Importation des CSV...")
    import_csv_to_cv()
# else:
#     st.info("La table 'cv' existe déjà.")

if not table_exists("lm"):
    #st.info("La table 'lm' n'existe pas. Importation des PDF...")
    process_folder()
# else:
#     st.info("La table 'lm' existe déjà.")

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
        background-color: #ccffcc;
        font-size: 16px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: white;
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

# Onglet 1 : Offre d'emploi
with tabs[0]:
    st.subheader("🔎 Explorez les Offres d'Emploi Disponibles")
    st.write("Consultez et recherchez des offres d'emploi provenant de France Travail.")
    
    # Récupérer et afficher les offres d'emploi
    df_offres = get_offres_from_db()
    
    # Filtrage par expérience et intitulé de poste
    col1, col2 = st.columns(2)
    with col1:
        selected_experience = st.selectbox("🎓 Sélectionnez le niveau d'expérience", ["Tous"] + list(df_offres['experience'].unique()))
    with col2:
        selected_intitule = st.selectbox("👩‍💻 Sélectionnez l'intitulé du poste", ["Tous"] + list(df_offres['intitule_poste'].unique()))
    
    if selected_experience != "Tous":
        df_offres = df_offres[df_offres['experience'] == selected_experience]
    if selected_intitule != "Tous":
        df_offres = df_offres[df_offres['intitule_poste'] == selected_intitule]
    
    st.subheader(f"📑 Affichage des offres : Niveau d'expérience '{selected_experience}' et intitulé de poste '{selected_intitule}'")
    
    gb = GridOptionsBuilder.from_dataframe(df_offres)
    gb.configure_pagination(paginationPageSize=10)
    gb.configure_default_column(filterable=True, sortable=True, resizable=True)
    grid_options = gb.build()
    AgGrid(df_offres, gridOptions=grid_options, height=400, width='100%')
    
    if st.button("💭 Nuage de Mots"):
        st.subheader("💭 Nuage de mots des descriptions de postes :")
        generate_wordcloud(df_offres)

    if st.button("📈 Distribution des niveaux d'expérience"):
        st.subheader("📈 Distribution des niveaux d'expérience demandés :")
        plot_experience_distribution(df_offres)

    if st.button("🤖 Clustering des Offres"):
        # Appeler la fonction cluster_offers et récupérer les valeurs
        df_clusters, clusters_terms, X = cluster_offers(df_offres)

        # Afficher le DataFrame avec les clusters
        st.subheader("🗂️ Tableau des Offres avec leurs clusters:")
        st.write(df_clusters)

        # Afficher les termes les plus importants pour chaque cluster
        st.subheader("🔑 Termes les plus importants pour chaque cluster")

        for cluster, terms in clusters_terms.items():
            st.markdown(f"""
                <div style="padding: 10px; margin: 10px 0; background-color: #f0f8ff; border-radius: 10px; display: flex; align-items: center;">
                    <h4 style="color: #006400; font-size: 16px; margin-right: 15px;">💡 <strong>{cluster} :</strong></h4>
                    <p style="color: #2f4f4f; font-size: 16px; margin: 0;">{', '.join(terms)}</p>
                </div>
            """, unsafe_allow_html=True)

        # Appliquer t-SNE et afficher le graphique interactif avec Plotly
        st.subheader("🌐 Visualisation des clusters avec t-SNE:")
        fig = plot_tsne(X, df_clusters['cluster_name'].values)

        fig.update_layout(
            autosize=True,
            width=600, 
            height=600  
        )
        st.plotly_chart(fig)

# Onglet 2 : Candidatures
with tabs[1]:
    st.subheader("📄 Parcourez les Candidatures")
    st.write("Téléchargez un CV et une Lettre de Motivation pour un candidat.")
    col1, col2 = st.columns(2)
    
    with col1:
        cv_file = st.file_uploader("Téléchargez votre CV (PDF ou Image)", type=["pdf", "jpg", "jpeg", "png"])
        if cv_file is not None:
            with st.expander("Aperçu du CV :"):
                if cv_file.type == "application/pdf":
                    with open("cv_temp.pdf", "wb") as f:
                        f.write(cv_file.read())
                    show_pdf("cv_temp.pdf")
                elif cv_file.type in ["image/jpeg", "image/png", "image/jpg"]:
                    st.download_button(label="Télécharger le CV", data=cv_file, file_name="cv_image.jpg", mime="image/jpeg")
                    st.image(cv_file, caption="Aperçu de l'image du CV", use_container_width=True)
    
    with col2:
        lm_file = st.file_uploader("Téléchargez votre Lettre de Motivation (PDF)", type=["pdf"])
        if lm_file is not None:
            with st.expander("Aperçu de la Lettre de Motivation :"):
                if lm_file.type == "application/pdf":
                    with open("lm_temp.pdf", "wb") as f:
                        f.write(lm_file.read())
                    show_pdf("lm_temp.pdf")
            if st.button("Démarrer l'Analyse"):
                progress_bar = st.progress(0)
                result = None
                for i in range(1, 101):
                    time.sleep(0.10)
                    progress_bar.progress(i)
                    if i == 100:
                        # Note : process_and_store_lm attend un cv_id en argument, ici on utilise une valeur d'exemple.
                        result = process_and_store_lm("lm_temp.pdf", "IDXXXXXX")
                if result:
                    st.write("Données extraites :")
                    st.markdown('<h3>🏷️ Compétences</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p>Compétences mentionnées : <span class="highlight">{result["competences"]}</span></p>', unsafe_allow_html=True)
                    st.markdown('<h3>🌍 Localisation</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p>Lieu de travail souhaité : <span class="highlight">{result["lieu"]}</span></p>', unsafe_allow_html=True)
                    st.markdown('<h3>💬 Motivations</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p>Motivations mentionnées : <span class="highlight">{result["motivations"]}</span></p>', unsafe_allow_html=True)
                    st.success("Analyse terminée et données insérées dans la base.")


# Contenu pour l'onglet 3 : Matching
with tabs[2]:
    st.subheader("💎 Trouvez le Candidat Idéal pour l'Offre")
    
    # Diviser en 2 colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        # Dans la première colonne, on permet de sélectionner une offre
        selected_offer = st.selectbox("🎯 Sélectionnez l'Offre d'Emploi", df_offres['intitule_poste'].unique())
        if selected_offer:
            # Afficher "Offre sélectionnée" avec du style
            st.markdown(f'<div class="selected-offer">Offre sélectionnée : {selected_offer}</div>', unsafe_allow_html=True)
        
            
            # Récupérer la description de l'offre en fonction de l'intitulé sélectionné
            offer_description = df_offres[df_offres['intitule_poste'] == selected_offer]['description'].values[0]
            
            # Ajouter un style CSS personnalisé
            st.markdown("""
                <style>
                    .offer-description {
                        background-color: #ffffff;
                        border-radius: 8px;
                        padding: 20px;
                        color: #333;
                        line-height: 1.6;
                        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                    }
                    .offer-description h4 {
                        color: #46a049;
                    }
                    .offer-description ul {
                        list-style-type: disc;
                        margin-left: 20px;
                    }
                    .offer-description li {
                        margin-bottom: 10px;
                    }
                    .selected-offer {
                        background-color: #2a9d8f;
                        color: white;
                        font-size: 1.5em;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 5px;
                        text-align: center;
                        margin-bottom: 20px;
                        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Afficher la description avec du style
            st.markdown(f"""
                <div class="offer-description">
                    <h4>📝 Description du poste :</h4>
                    <p>{offer_description}</p>
                </div>
            """, unsafe_allow_html=True)

    if st.button("🔄 Générer l'Embedding de l'Offre Sélectionnée"):
        offer_data = df_offres[df_offres["reference"] == selected_offer]  # Filtrer l'offre sélectionnée
        generate_offers_embeddings(offer_data)  # Générer uniquement pour cette offre
        st.success(f"✅ Embedding généré pour l'offre {selected_offer} !")

    
    with col2:
        # Dans la deuxième colonne, afficher un message ou les résultats du matching
        #if cv_file and lm_file:
            # Ajouter un bouton avec un texte stylisé
        st.subheader("🏅 Découvrez le Candidat Idéal en Un Clic !")
        if st.button("Lancer le matching ..."):
            # Ajoute ici la logique de matching entre le CV, la LM et les offres d'emploi
            st.write(f"Matching en cours pour l'offre : {selected_offer}...")
            st.write("Les résultats du matching s'affichent ici.")
        # else:
        #     st.warning("⚠️ Veuillez télécharger un CV et une Lettre de Motivation pour procéder au matching.")
