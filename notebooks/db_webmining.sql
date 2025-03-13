CREATE TABLE postes (
    id SERIAL PRIMARY KEY,
    intitule_poste TEXT NOT NULL,
    description TEXT,
    experience TEXT,
    langue TEXT,
    permis TEXT,
    divers TEXT,
    reference TEXT UNIQUE
);
CREATE TABLE competences (
    id SERIAL PRIMARY KEY,
    nom TEXT NOT NULL UNIQUE,
    requis BOOLEAN NOT NULL
);

CREATE TABLE savoirs (
    id SERIAL PRIMARY KEY,
    nom TEXT NOT NULL UNIQUE,
    requis BOOLEAN NOT NULL
);

CREATE TABLE poste_competences (
    poste_id INT REFERENCES postes(id) ON DELETE CASCADE,
    competence_id INT REFERENCES competences(id) ON DELETE CASCADE,
    PRIMARY KEY (poste_id, competence_id)
);

CREATE TABLE poste_savoirs (
    poste_id INT REFERENCES postes(id) ON DELETE CASCADE,
    savoirs_id INT REFERENCES savoirs(id) ON DELETE CASCADE,
    PRIMARY KEY (poste_id, savoirs_id)
);