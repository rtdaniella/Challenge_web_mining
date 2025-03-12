import streamlit as st
from PIL import Image
import base64
import os

# Fonction pour afficher un PDF dans Streamlit
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_data = f.read()
    b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
    pdf_html = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="700" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_html, unsafe_allow_html=True)

# Configuration de l'application Streamlit
st.title("Application de Téléchargement CV et LM")

st.markdown("""
    Téléchargez un CV (en format PDF ou Image) et une Lettre de Motivation (LM en PDF) pour afficher un aperçu du fichier téléchargé.
""")

# Créer deux colonnes pour l'affichage horizontal
col1, col2 = st.columns(2)

# Téléchargement du CV (format PDF ou image) dans la première colonne
with col1:
    cv_file = st.file_uploader("Téléchargez votre CV (PDF ou Image)", type=["pdf", "jpg", "jpeg", "png"])
    
    if cv_file is not None:
        st.subheader("Aperçu du CV :")
        
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
            st.write("Aperçu de l'image du CV :")
            st.image(cv_file, caption="Aperçu de l'image du CV", use_container_width=True)

# Téléchargement de la LM (format PDF uniquement) dans la deuxième colonne
with col2:
    lm_file = st.file_uploader("Téléchargez votre Lettre de Motivation (PDF)", type=["pdf"])
    
    if lm_file is not None:
        st.subheader("Aperçu de la Lettre de Motivation :")
        
        # Si c'est un fichier PDF
        if lm_file.type == "application/pdf":
            # Sauvegarder le fichier PDF localement temporairement
            with open("lm_temp.pdf", "wb") as f:
                f.write(lm_file.read())
            show_pdf("lm_temp.pdf")
