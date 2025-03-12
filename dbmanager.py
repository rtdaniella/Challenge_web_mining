import psycopg2
from psycopg2 import sql

# Configuration de la connexion PostgreSQL
DB_CONFIG = {
    "dbname": "cv_lm_db",
    "user": "postgres",
    "password": "daniella",
    "host": "localhost",
    "port": 5432
}


def get_db_connection():
    """Ouvre une connexion à PostgreSQL."""
    return psycopg2.connect(**DB_CONFIG)

def create_tables():
    """Crée les tables dans la base de données."""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Créer la table des compétences
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS competences (
            id SERIAL PRIMARY KEY,
            competences TEXT[]
        );
        """)
        
        # Créer la table des motivations
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS motivations (
            id SERIAL PRIMARY KEY,
            motivations TEXT[]
        );
        """)
        
        # Créer la table des lieux
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS lieux (
            id SERIAL PRIMARY KEY,
            lieu TEXT
        );
        """)
        
        # Commit les changements dans la base de données
        connection.commit()
        print("Tables créées avec succès.")
    except Exception as e:
        print(f"Erreur lors de la création des tables : {e}")
    finally:
        cursor.close()
        connection.close()

def insert_competences(competences):
    """Insère des compétences dans la table des compétences."""
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Insertion des compétences
        cursor.execute("""
        INSERT INTO competences (competences)
        VALUES (%s)
        """, (competences,))
        
        # Commit les changements
        connection.commit()
        print("Compétences insérées avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'insertion des compétences : {e}")
    finally:
        cursor.close()
        connection.close()

def insert_motivations(motivations):
    """Insère des motivations dans la table des motivations."""
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Insertion des motivations
        cursor.execute("""
        INSERT INTO motivations (motivations)
        VALUES (%s)
        """, (motivations,))
        
        # Commit les changements
        connection.commit()
        print("Motivations insérées avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'insertion des motivations : {e}")
    finally:
        cursor.close()
        connection.close()

def insert_lieu(lieu):
    """Insère le lieu de disponibilité dans la table des lieux."""
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Insertion du lieu
        cursor.execute("""
        INSERT INTO lieux (lieu)
        VALUES (%s)
        """, (lieu,))
        
        # Commit les changements
        connection.commit()
        print("Lieu inséré avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'insertion du lieu : {e}")
    finally:
        cursor.close()
        connection.close()

create_tables()