import streamlit as st
from utils import (
    get_db_connection,
    import_csv_to_cv,
    process_and_store_lm,
    process_folder,
    show_pdf,
    show_image,
    get_offres_from_db,
    generate_wordcloud,
    plot_experience_distribution,
    cluster_offers,
    plot_tsne,
    generate_offers_embeddings,
    generate_candidate_embeddings,
    get_candidatures
)
import time
from st_aggrid import AgGrid, GridOptionsBuilder
import logging
import os
import pandas as pd 
import PIL.Image
from processor import process_cv_with_gemini, insert_cv_dataframe
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
st.set_page_config(layout="wide")


# On définit cv_id dans l'espace global de la session
if 'cv_id' not in st.session_state:
    st.session_state.cv_id = None

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
    # st.info("La table 'cv' n'existe pas. Importation des CSV...")
    import_csv_to_cv()
else:
    print("La table 'cv' existe déjà.")

if not table_exists("lm"):
    # st.info("La table 'lm' n'existe pas. Importation des PDF...")
    process_folder()
else:
    print("La table 'lm' existe déjà.")

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
     
     # Récupérer et afficher les candidatures
    df_candidatures = get_candidatures()

    st.write("Téléchargez un CV et une Lettre de Motivation pour un candidat.")
    col1, col2 = st.columns(2)
    
    # with col1:
    #     cv_file = st.file_uploader("Téléchargez votre CV (PDF ou Image)", type=["pdf", "jpg", "jpeg", "png"])
    #     if cv_file is not None:
    #         with st.expander("Aperçu du CV :"):
    #             if cv_file.type == "application/pdf":
    #                 with open("cv_temp.pdf", "wb") as f:
    #                     f.write(cv_file.read())
    #                 show_pdf("cv_temp.pdf")
    #             elif cv_file.type in ["image/jpeg", "image/png", "image/jpg"]:
    #                 st.download_button(label="Télécharger le CV", data=cv_file, file_name="cv_image.jpg", mime="image/jpeg")
    #                 st.image(cv_file, caption="Aperçu de l'image du CV", use_container_width=True)
    
    # with col2:
    #     lm_file = st.file_uploader("Téléchargez votre Lettre de Motivation (PDF)", type=["pdf"])
    #     if lm_file is not None:
    #         with st.expander("Aperçu de la Lettre de Motivation :"):
    #             if lm_file.type == "application/pdf":
    #                 with open("lm_temp.pdf", "wb") as f:
    #                     f.write(lm_file.read())
    #                 show_pdf("lm_temp.pdf")
    #         if st.button("Démarrer l'Analyse"):
    #             progress_bar = st.progress(0)
    #             result = None
    #             for i in range(1, 101):
    #                 time.sleep(0.10)
    #                 progress_bar.progress(i)
    #                 if i == 100:
    #                     # Note : process_and_store_lm attend un cv_id en argument, ici on utilise une valeur d'exemple.
    #                     result = process_and_store_lm("lm_temp.pdf", "IDXXXXXX")
    #             if result:
    #                 st.write("Données extraites :")
    #                 st.markdown('<h3>🏷️ Compétences</h3>', unsafe_allow_html=True)
    #                 st.markdown(f'<p>Compétences mentionnées : <span class="highlight">{result["competences"]}</span></p>', unsafe_allow_html=True)
    #                 st.markdown('<h3>🌍 Localisation</h3>', unsafe_allow_html=True)
    #                 st.markdown(f'<p>Lieu de travail souhaité : <span class="highlight">{result["lieu"]}</span></p>', unsafe_allow_html=True)
    #                 st.markdown('<h3>💬 Motivations</h3>', unsafe_allow_html=True)
    #                 st.markdown(f'<p>Motivations mentionnées : <span class="highlight">{result["motivations"]}</span></p>', unsafe_allow_html=True)
    #                 st.success("Analyse terminée et données insérées dans la base.")
   
 
    with col1:
        cv_image = st.file_uploader("Téléchargez le CV (Image)", type=["jpg", "jpeg", "png"], key="cv_uploader")
        if cv_image is not None:
            # Stocker l'image dans la session_state pour réutilisation
            if "cv_preview_img" not in st.session_state:
                st.session_state.cv_preview_img = PIL.Image.open(cv_image)
            
            # Afficher un expander contenant l'aperçu du CV en utilisant show_image()
            with st.expander("Aperçu du CV"):
                temp_path = "cv_temp.jpg"
                st.session_state.cv_preview_img.save(temp_path)
                show_image(temp_path)  # Fonction définie dans utils.py
            
            # Bouton pour analyser et valider le CV
            if st.button("Valider", key="validate_cv"):
                progress_bar = st.progress(0)
                result_df = None
                for i in range(1, 101):
                    time.sleep(0.05)
                    progress_bar.progress(i)
                    if i == 100:
                        result_df = process_cv_with_gemini(st.session_state.cv_preview_img)
                if result_df is not None:
                    st.write("Données extraites du CV :")
                    st.write(result_df)  # Affichage du DataFrame extrait
                    # Récupérer l'ID_CV (on suppose qu'il n'y a qu'une seule ligne)
                    st.session_state.cv_id = result_df.iloc[0]["ID_CV"]
                    # Insérer le CV dans la base de données
                    insert_cv_dataframe(result_df)
                    st.success("Le CV a été inséré dans la base avec succès.")
                else:
                    st.error("L'analyse du CV a échoué.")




    with col2:
        lm_file = st.file_uploader("Téléchargez la Lettre de Motivation (PDF)", type=["pdf"], key="lm_uploader")
        if lm_file is not None:
            # Afficher un aperçu du PDF dans un expander
            with st.expander("Aperçu de la Lettre de Motivation"):
                lm_content = lm_file.getvalue()
                with open("lm_temp.pdf", "wb") as f:
                    f.write(lm_content)
                show_pdf("lm_temp.pdf")
            
            # Bouton pour valider l'insertion de la lettre de motivation
            if st.button("Valider la Lettre de Motivation", key="validate_lm"):
                if st.session_state.cv_id is not None:
                    if process_and_store_lm("lm_temp.pdf", st.session_state.cv_id):
                        st.success("Les données ont été insérées pour le CV et la lettre de motivation associée.")
                        # Ici, on ne réinitialise pas cv_id pour conserver l'affichage du CV
                    else:
                        st.error("L'insertion de la lettre de motivation a échoué.")
                else:
                    st.error("Veuillez ajouter le CV avant d'ajouter la lettre de motivation.")


# Contenu pour l'onglet 3 : Matching
with tabs[2]:
    st.subheader("💎 Trouvez le Candidat Idéal pour l'Offre")
    
    # Diviser en 2 colonnes
    col1, col2 = st.columns(2)
    
    # with col1:
    #     # Dans la première colonne, on permet de sélectionner une offre
    #     selected_offer = st.selectbox("🎯 Sélectionnez l'Offre d'Emploi", df_offres['intitule_poste'].unique())
    #     if selected_offer:
    #         # Afficher "Offre sélectionnée" avec du style
    #         st.markdown(f'<div class="selected-offer">Offre sélectionnée : {selected_offer}</div>', unsafe_allow_html=True)
        
            
    #         # Récupérer la description de l'offre en fonction de l'intitulé sélectionné
    #         offer_description = df_offres[df_offres['intitule_poste'] == selected_offer]['description'].values[0]
            
    #         # Ajouter un style CSS personnalisé
    #         st.markdown("""
    #             <style>
    #                 .offer-description {
    #                     background-color: #ffffff;
    #                     border-radius: 8px;
    #                     padding: 20px;
    #                     color: #333;
    #                     line-height: 1.6;
    #                     box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    #                 }
    #                 .offer-description h4 {
    #                     color: #46a049;
    #                 }
    #                 .offer-description ul {
    #                     list-style-type: disc;
    #                     margin-left: 20px;
    #                 }
    #                 .offer-description li {
    #                     margin-bottom: 10px;
    #                 }
    #                 .selected-offer {
    #                     background-color: #2a9d8f;
    #                     color: white;
    #                     font-size: 1.5em;
    #                     font-weight: bold;
    #                     padding: 10px;
    #                     border-radius: 5px;
    #                     text-align: center;
    #                     margin-bottom: 20px;
    #                     box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    #                 }
    #             </style>
    #         """, unsafe_allow_html=True)
            
    #         # Afficher la description avec du style
    #         st.markdown(f"""
    #             <div class="offer-description">
    #                 <h4>📝 Description du poste :</h4>
    #                 <p>{offer_description}</p>
    #             </div>
    #         """, unsafe_allow_html=True)

    # if st.button("🔄 Générer l'Embedding de l'Offre Sélectionnée"):
    #     offer_data = df_offres[df_offres["reference"] == selected_offer]  # Filtrer l'offre sélectionnée
    #     generate_offers_embeddings(offer_data)  # Générer uniquement pour cette offre
    #     st.success(f"✅ Embedding généré pour l'offre {selected_offer} !")

    # with col2:
    #     # Dans la deuxième colonne, afficher un message ou les résultats du matching
    #     #if cv_file and lm_file:
    #         # Ajouter un bouton avec un texte stylisé
    #     st.subheader("🏅 Découvrez le Candidat Idéal en Un Clic !")
    #     if st.button("Lancer le matching ..."):
    #         # Ajoute ici la logique de matching entre le CV, la LM et les offres d'emploi
    #         st.write(f"Matching en cours pour l'offre : {selected_offer}...")
    #         st.write("Les résultats du matching s'affichent ici.")
    #     # else:
    #     #     st.warning("⚠️ Veuillez télécharger un CV et une Lettre de Motivation pour procéder au matching.")


    with col1:
        # Dans la première colonne, on permet de sélectionner une offre
        selected_offer = st.selectbox("🎯 Sélectionnez l'Offre d'Emploi", df_offres['intitule_poste'].unique())
        if selected_offer:
            st.markdown(f'<div class="selected-offer">Offre sélectionnée : {selected_offer}</div>', unsafe_allow_html=True)
            # Récupérer la description de l'offre selon l'intitulé sélectionné
            offer_description = df_offres[df_offres['intitule_poste'] == selected_offer]['description'].values[0]
            st.markdown(f"""
                <div class="offer-description">
                    <h4>📝 Description du poste :</h4>
                    <p>{offer_description}</p>
                </div>
            """, unsafe_allow_html=True)
        
        if st.button("🔄 Générer l'Embedding de l'Offre Sélectionnée", key="gen_offer_embed"):
            offer_data = df_offres[df_offres["reference"] == selected_offer]
            # Ici, on suppose que generate_offers_embeddings retourne un tableau numpy
            offer_embedding = generate_offers_embeddings(offer_data, text_column="description")
            st.success(f"✅ Embedding généré pour l'offre {selected_offer} !")

    with col2:
        st.subheader("🏅 Découvrez le Candidat Idéal en Un Clic !")
        if st.button("Lancer le matching ...", key="match"):
            st.write(f"Matching en cours pour l'offre : {selected_offer}...")
            
            # 1. Récupérer la jointure entre cv et lm
            candidatures_df = get_candidatures()  # jointure sur cv."ID_CV" et lm.cv_id
            
            # 2. Générer les embeddings pour chaque candidature
            candidate_embeddings = generate_candidate_embeddings(candidatures_df)
            
            # 3. Générer l'embedding pour l'offre sélectionnée
            offer_data = df_offres[df_offres["reference"] == selected_offer]
            
            offer_embedding = generate_offers_embeddings(offer_data, text_column="description")
            
            # 4. Calculer la similarité (cosine similarity)
            sim_matrix = cosine_similarity(candidate_embeddings, offer_embedding)
            # Pour chaque candidat, sim_matrix[i, 0] donne son score par rapport à l'offre
            best_candidate_idx = np.argmax(sim_matrix, axis=0)[0]  # index du meilleur candidat
            best_candidate = candidatures_df.iloc[best_candidate_idx]
            best_score = sim_matrix[best_candidate_idx, 0]
            
            # 5. Afficher les résultats du matching
            st.markdown("### Meilleur Candidat Trouvé :")
            st.write(best_candidate)  # Affiche toutes les informations du candidat
            st.write(f"**Score de similarité :** {best_score:.3f}")