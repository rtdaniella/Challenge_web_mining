import re
import json
import random
import xml.etree.ElementTree as ET
import pandas as pd
import google.generativeai as genai
import os
import logging
import pandas as pd
from utils import get_db_connection, fix_json_value  # Assurez-vous que fix_json_value est bien importé


# Configure l'API Gemini
GEMINI_API_KEY = 'AIzaSyDtpKAglnd1t0TsOB8WaYTQ6PlaoCIkXcs'
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

def get_text_or_empty(element, tag):
    """
    Renvoie element.find(tag).text si la balise existe.
    Sinon, renvoie une chaîne vide.
    """
    if element is None:
        return ""
    node = element.find(tag)
    return node.text.strip() if (node is not None and node.text) else ""

def parse_xml_to_df(xml_data):
    """
    Nettoie le XML, parse les informations du CV et retourne un DataFrame.
    """
    # Nettoyage du XML
    xml_data = re.sub(r"[\\\n\\\t]", "", xml_data)
    xml_data = xml_data[xml_data.find("<CV>"):]
    xml_data = re.sub(r">\s+<", "><", xml_data)
    xml_data = xml_data.strip()
    xml_data = xml_data[:xml_data.rfind("</CV>") + len("</CV>")]

    # Parser le XML
    root = ET.fromstring(xml_data)

    # Extraction des informations personnelles
    infos_perso = root.find("./InformationsPersonnelles")
    cv_info = {
        "ID_CV": f"ID{random.randint(100000, 999999)}",
        "Nom": get_text_or_empty(infos_perso, "Nom"),
        "Prenom": get_text_or_empty(infos_perso, "Prenom"),
        "Adresse": get_text_or_empty(infos_perso, "Adresse"),
        "CodePostal": get_text_or_empty(infos_perso, "CodePostal"),
        "Ville": get_text_or_empty(infos_perso, "Ville"),
        "NumeroTelephone": get_text_or_empty(infos_perso, "NumeroTelephone"),
        "Email": get_text_or_empty(infos_perso, "Email"),
    }

    # Extraction des formations
    formations = []
    for formation in root.findall("./Formations/Formation"):
        formations.append({
            "Nom": get_text_or_empty(formation, "Nom"),
            "Dates": get_text_or_empty(formation, "Dates"),
            "Institution": get_text_or_empty(formation, "Institution"),
        })
    cv_info["Formations"] = json.dumps(formations, ensure_ascii=False)

    # Extraction des expériences professionnelles
    experiences = []
    for exp in root.findall("./ExperiencesProfessionnelles/Experience"):
        experiences.append({
            "Poste": get_text_or_empty(exp, "Poste"),
            "Dates": get_text_or_empty(exp, "Dates"),
            "Entreprise": get_text_or_empty(exp, "Entreprise"),
            "Description": get_text_or_empty(exp, "Description"),
        })
    cv_info["Experiences"] = json.dumps(experiences, ensure_ascii=False)

    # Extraction des projets
    projets = []
    for projet in root.findall("./Projets/Projet"):
        projets.append({
            "Nom": get_text_or_empty(projet, "Nom"),
            "Description": get_text_or_empty(projet, "Description"),
            "Technologies": get_text_or_empty(projet, "Technologies"),
        })
    cv_info["Projets"] = json.dumps(projets, ensure_ascii=False)

    # Extraction des compétences
    competences = []
    for comp in root.findall("./Competences/Competence"):
        comp_text = comp.text.strip() if (comp is not None and comp.text) else ""
        if comp_text:
            competences.append(comp_text)
    cv_info["Competences"] = ", ".join(competences)

    df = pd.DataFrame([cv_info])
    return df

def process_cv_with_gemini(img):
    """
    À partir d'une image (fichier PIL), cette fonction :
      1. Envoie l'image et un prompt à l'API Gemini pour générer un XML structuré.
      2. Nettoie et parse la réponse XML pour créer un DataFrame.
      3. Retourne le DataFrame.
    """
    prompt = """
Tu vas recevoir une image contenant un CV. Ton objectif est d'extraire et structurer les informations sous forme de XML.

### Consignes :
1️⃣ Récupère les informations pertinentes du CV (informations personnelles, formations, expériences, etc).
2️⃣ Respecte **STRICTEMENT** le format XML suivant :

Sois sûr que l'identifiant est unique et aléatoire, utilise une fonction random pour cela.
Ne prends pas les publications.

```xml
<CV>
    <ID_CV>[Générer un identifiant unique aléatoire]</ID_CV>
    <InformationsPersonnelles>
        <Nom>[Nom du candidat]</Nom>
        <Prenom>[Prénom du candidat]</Prenom>
        <Adresse>[Adresse]</Adresse>
        <CodePostal>[CodePostal]</CodePostal>
        <Ville>[Ville]</Ville>
        <NumeroTelephone>[Numéro de téléphone]</NumeroTelephone>
        <Email>[Email]</Email>
    </InformationsPersonnelles>
    <Formations>
        <Formation>
            <Nom>[Nom de la formation]</Nom>
            <Dates>[Dates]</Dates>
            <Institution>[Institution]</Institution>
        </Formation>
    </Formations>
    <ExperiencesProfessionnelles>
        <Experience>
            <Poste>[Titre du poste]</Poste>
            <Dates>[Dates]</Dates>
            <Entreprise>[Entreprise]</Entreprise>
            <Description>[Brève description]</Description>
        </Experience>
    </ExperiencesProfessionnelles>
    <Projets>
        <Projet>
            <Nom>[Nom du projet]</Nom>
            <Description>[Brève description du projet]</Description>
            <Technologies>[Technologies utilisées]</Technologies>
        </Projet>
    </Projets>
    <Competences>
        <Competence>[Compétence 1]</Competence>
        <Competence>[Compétence 2]</Competence>
        <Competence>[Compétence 3]</Competence>
    </Competences>
</CV>
"""
    # Appel à l'API Gemini avec le prompt et l'image (objet PIL)
    response = model.generate_content(
        [prompt, img],
        stream=True,
    )
    response.resolve()
    xml_data = response.text
    df = parse_xml_to_df(xml_data)
    return df


def insert_cv_dataframe(df):
    """
    Insère dans la table 'cv' les données contenues dans le DataFrame 'df'.
    Les colonnes 'Formations', 'Experiences' et 'Projets' sont converties en JSONB.
    """
    conn = get_db_connection()
    if conn is None:
        logging.error("Impossible de se connecter à la base de données.")
        return

    cursor = conn.cursor()

    # Requête d'insertion dans cv
    insert_query = """
    INSERT INTO cv (
        "ID_CV", 
        "Nom", 
        "Prenom", 
        "Adresse", 
        "CodePostal", 
        "Ville", 
        "NumeroTelephone", 
        "Email", 
        "Formations", 
        "Experiences", 
        "Projets", 
        "Competences"
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, 
        %s::jsonb, 
        %s::jsonb, 
        %s::jsonb, 
        %s
    )
    ON CONFLICT ("ID_CV") DO NOTHING;
    """
    
    # Parcours du DataFrame et insertion ligne par ligne
    for idx, row in df.iterrows():
        row_id_cv      = row.get("ID_CV", None)
        row_nom        = row.get("Nom", None)
        row_prenom     = row.get("Prenom", None)
        row_adresse    = row.get("Adresse", None)
        row_codepostal = row.get("CodePostal", None)
        row_ville      = row.get("Ville", None)
        row_telephone  = row.get("NumeroTelephone", None)
        row_email      = row.get("Email", None)
        # Pour les colonnes JSONB, on utilise fix_json_value pour gérer les valeurs manquantes
        row_formations  = fix_json_value(row.get("Formations", None))
        row_experiences = fix_json_value(row.get("Experiences", None))
        row_projets     = fix_json_value(row.get("Projets", None))
        row_competences = row.get("Competences", None)
        
        try:
            cursor.execute(
                insert_query,
                (
                    row_id_cv,
                    row_nom,
                    row_prenom,
                    row_adresse,
                    row_codepostal,
                    row_ville,
                    row_telephone,
                    row_email,
                    row_formations,
                    row_experiences,
                    row_projets,
                    row_competences
                )
            )
        except Exception as e:
            logging.error("Erreur lors de l'insertion de la ligne %s: %s", idx, e)
    
    conn.commit()
    cursor.close()
    conn.close()
    logging.info("Le DataFrame a été inséré dans la table 'cv'.")
