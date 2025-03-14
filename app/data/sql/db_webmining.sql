CREATE TABLE IF NOT EXISTS annonces(
    reference TEXT PRIMARY KEY,
    intitule_poste TEXT NOT NULL,
    description TEXT,
    experience TEXT,
    divers TEXT    
);
CREATE TABLE IF NOT EXISTS competences (
    id SERIAL PRIMARY KEY,
    nom TEXT NOT NULL UNIQUE,
    requis BOOLEAN
);

CREATE TABLE IF NOT EXISTS annonce_competences (
    annonce_reference TEXT REFERENCES annonces(reference) ON DELETE CASCADE,
    competence_id INT REFERENCES competences(id) ON DELETE CASCADE,
    PRIMARY KEY (annonce_reference, competence_id)
);
