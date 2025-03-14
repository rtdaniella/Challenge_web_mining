import os
import time
import requests
import pyarrow
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import psycopg2

from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.8",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Host": "candidat.francetravail.fr",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Sec-GPC": "1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "sec-ch-ua": '"Chromium";v="134", "Not:A-Brand";v="24", "Brave";v="134"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

cookies = {
    "TCPID": "125331445152517760379",
    "dtCookie": "v_4_srv_8_sn_A03B54816D5AF0755CC6317FD915D035_perc_100000_ol_0_mul_1_app-3A6e350f5c3735afd9_1_rcs-3Acss_0",
    "BIGipServerPOOL_PWNOT-00PT28_HTTPS_PN055-VIPA_3_PN055": "2940013066.56131.0000",
    "TS016fc3b0": "0150c672c31d6f6c7a73fc28f33de8c9cb7dc5105bab64f0d4292f3e76519df0c24f2cfae5e0d5cb90ff9f7ca9259859d850aa9eed",
    "JSESSIONID_RECH_OFFRE": "6HqNhHJ-DbDp_Y5VWI_-YG5oCPFoFAzpLXQmdlZ8YZenhh6JuYLm!-292444737",
    "userBadge": "0",
}
session = requests.Session()
session.headers.update(headers)

def parse_job():
    with open("/opt/airflow/dags/urls.txt", "r") as f:
        urls = [url for url in f.readlines()]
        
    domaine = "https://candidat.francetravail.fr"

    for i in range(0,len(urls), 2):
        r = session.get(url=domaine + str(urls[i]).replace("\n", ""), cookies=cookies)
        if r.status_code != 200:
            break
        soup = BeautifulSoup(r.text)

        competences = []
        annonce = dict()

        annonce["intitule_poste"] = soup.find("span", {"itemprop": "title"}).text
        annonce["description"] = soup.find("div", {"itemprop": "description"}).text
        annonce["experience"] = soup.find("span", {"itemprop": "experienceRequirements"}).text

        skill = soup.select("ul.skill-list.list-unstyled > li")
        for item in skill:
            exp = item.find_all("span", {"itemprop": "experienceRequirements"})
            educ = item.find_all("span", {"itemprop": "educationRequirements"})
            comp = item.find("span", {"class": "skill skill-competence"})
            sav = item.find("span", {"class": "skill skill-savoir"})
            lang = item.find("span", {"class": "skill skill-langue"})
            perm = item.find("span", {"class": "skill skill-permis"})
            if len(exp) != 0:
                experience = exp[0].text
                exp_requis = True if item.find("span", {"class": "skill-required"}) else False
                annonce["experience"] = (experience, exp_requis)
            if educ:
                education = educ[0].text
                educ_requis = True if item.find("span", {"class": "skill-required"}) else False
                annonce["education"] = education
            if lang:
                langue = lang.find("span", {"class": "skill-name"}).text
                langue_requis = True if item.find("span", {"class": "skill-required"}) else False
                competences.append(langue)
            if perm:
                permis = item.find("span", {"class": "skill-name"}).text
                permis_requis = True if item.find("span", {"class": "skill-required"}) else False
                competences.append(permis)
            if comp:
                competence = comp.find("span", {"class": "skill-name"}).text
                requis = True if comp.find("span", {"class": "skill-required"}) else False
                competences.append(competence)
            if sav:
                savoir = sav.find("span", {"class": "skill-name"}).text
                requis = True if sav.find("span", {"class": "skill-required"}) else False
                competences.append(savoir)
        annonce["competences"] = competences
        infos = soup.find("div", {"class": "description-aside col-sm-4 col-md-5"}).select("dd")
        divers = [item.text for item in infos[2:]]
        annonce["divers"] = divers
        annonce["reference"] = urls[i].split("/")[-1]
        to_db(annonce)
        time.sleep(2)



def to_db(annonce: dict):
    '''
    TODO:
        - corriger l'auto-increment en cas de doublon dans une table
    '''
    # Connexion à la base de données PostgreSQL
    conn = psycopg2.connect(
        dbname="webmining",
        user="postgres",
        password="postgres",
        host="my_postgres_container",
        port="5432"
    )
    cur = conn.cursor()

    for competence in annonce["competences"]:
        cur.execute("""
            INSERT INTO competences (nom)
            VALUES (%s)
            ON CONFLICT (nom) DO NOTHING;
        """, (competence,))

    # Insert postes (job positions)
    cur.execute("""
        INSERT INTO annonces (intitule_poste, description, experience, divers, reference)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (reference) DO NOTHING;
        RETURNING reference;
    """, (annonce["intitule_poste"], annonce["description"], annonce["experience"], annonce["divers"], annonce["reference"]))
    
    # print(cur.fetchone())
    annonce_id = cur.fetchone()[0]
    print(annonce_id)

    for competence in annonce["competences"]:
        cur.execute("""
        INSERT INTO annonce_competences (annonce_reference, competence_id)
        SELECT %s, id FROM competences WHERE nom = %s;
        """, (annonce_id, competence))


    # Validation des modifications et fermeture de la connexion
    conn.commit()
    cur.close()
    conn.close()

def get_daily_listing():
    pass
    urls = []
    offre_FT = [
        "https://candidat.francetravail.fr/offres/recherche?emission=1&lieux=69381&motsCles=data+engineer&offresPartenaires=true&range=0-19&rayon=20&tri=0",
        "https://candidat.francetravail.fr/offres/recherche?emission=1&lieux=69381&motsCles=data+analyst&offresPartenaires=true&range=0-19&rayon=30&tri=0",
        "https://candidat.francetravail.fr/offres/recherche?emission=1&lieux=69381&motsCles=data+scientist&offresPartenaires=true&range=0-19&rayon=30&tri=0",
        "https://candidat.francetravail.fr/offres/recherche?emission=1&lieux=69381&motsCles=big+data&offresPartenaires=true&range=0-19&rayon=30&tri=0",
        "https://candidat.francetravail.fr/offres/recherche?emission=1&lieux=69381&motsCles=intelligence+artificielle&offresPartenaires=true&range=0-19&rayon=30&tri=0",
        "https://candidat.francetravail.fr/offres/recherche?emission=1&motsCles=architecte+cloud&offresPartenaires=true&range=0-19&rayon=10&tri=0"]

    for url in offre_FT:
        while True:
            r = requests.get(url, headers= headers)
            if r.status_code != 200:
                print("waiting 5s")
                time.sleep(5)
                continue
            soup = BeautifulSoup(r.text)
            urls = urls + [link.get("href") for link in soup.select('ul[data-container-type="zone"] > li > a')]
            time.sleep(2)
            break
    with open("/opt/airflow/dags/urls.txt", "w") as f:
        for url in urls:
            f.write(url+ "\n")
            # f.writelines(offre_FT)
    

default_args = {
    'owner': 'joel',
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
    'start_date': datetime(2025, 3, 13),  # Adjust this date
}

with DAG(
    dag_id= "parse_FT",
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    catchup=False
)as dag:
    
    get_daily_listings_task = PythonOperator(
        task_id = 'get_daily_listing_task',
        python_callable= get_daily_listing
    )

    parse_job_offerings_task =PythonOperator(
        task_id = 'parse_job_offerings_task',
        python_callable= parse_job,
    )

    get_daily_listings_task >> parse_job_offerings_task