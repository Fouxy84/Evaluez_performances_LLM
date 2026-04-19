# Intégration SQL + RAG pour Évaluation NBA

## 📋 Vue d'ensemble

Ce projet intègre un système SQL complet avec le RAG existant pour analyser les statistiques NBA. Le système détecte automatiquement si une question nécessite une requête structurée (SQL) ou une analyse contextuelle (RAG).

## 🏗️ Architecture

### Composants Principaux

```
┌─────────────────────────────────────────────────────────┐
│                    Utilisateur                          │
└────────────────┬──────────────────────────────────────┘
                 │
         Question/Requête
                 │
         ┌───────▼──────────┐
         │ Query Detector   │  (Détection automatique)
         └───────┬──────────┘
                 │
        ┌────────┴────────┐
        │                 │
    [SQL]           [RAG/Hybrid]
        │                 │
        ▼                 ▼
   ┌─────────┐      ┌──────────┐
   │SQL Tool │      │RAG Agent │
   │(Struct) │      │(Context) │
   └────┬────┘      └────┬─────┘
        │                 │
        └────────┬────────┘
                 │
         ┌───────▼──────────┐
         │Response & Caching│
         └───────────────────┘
```

### Tables de Base de Données

#### `teams` (Équipes)
```sql
CREATE TABLE teams (
    id INTEGER PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    abbreviation VARCHAR(3) UNIQUE NOT NULL,
    city VARCHAR(50)
);
```

#### `players` (Joueurs)
```sql
CREATE TABLE players (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    team_id INTEGER FOREIGN KEY,
    position VARCHAR(10),  -- PG, SG, SF, PF, C
    jersey_number INTEGER,
    height FLOAT,  -- cm
    weight FLOAT,  -- kg
    birth_date DATE
);
```

#### `matches` (Matchs)
```sql
CREATE TABLE matches (
    id INTEGER PRIMARY KEY,
    date DATE NOT NULL,
    season INTEGER,
    match_type VARCHAR(20),  -- regular, playoff, finals
    home_team_id INTEGER FOREIGN KEY NOT NULL,
    away_team_id INTEGER FOREIGN KEY NOT NULL,
    home_score INTEGER,
    away_score INTEGER
);
```

#### `stats` (Statistiques)
```sql
CREATE TABLE stats (
    id INTEGER PRIMARY KEY,
    player_id INTEGER FOREIGN KEY NOT NULL,
    match_id INTEGER FOREIGN KEY NOT NULL,
    -- Scoring
    points INTEGER DEFAULT 0,
    field_goals_made INTEGER,
    field_goals_attempted INTEGER,
    field_goal_percentage FLOAT,
    three_pointers_made INTEGER,
    three_pointers_attempted INTEGER,
    three_point_percentage FLOAT,
    -- Rebounds
    total_rebounds INTEGER,
    offensive_rebounds INTEGER,
    defensive_rebounds INTEGER,
    -- Assists & Ball Handling
    assists INTEGER,
    turnovers INTEGER,
    steals INTEGER,
    blocks INTEGER,
    -- Other
    fouls_personal INTEGER,
    plus_minus INTEGER,
    efficiency_rating FLOAT,
    minutes_played FLOAT
);
```

#### `reports` (Rapports)
```sql
CREATE TABLE reports (
    id INTEGER PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    player_id INTEGER FOREIGN KEY,
    content TEXT,
    query_used VARCHAR(500),
    generated_at DATETIME,
    report_type VARCHAR(50)  -- comparison, trend, performance
);
```

## 🚀 Démarrage Rapide

### 1. Installation des dépendances

```bash
pip install sqlalchemy==2.0.23 pandas openpyxl
```

### 2. Initialiser la base de données

```bash
python integrate_sql.py
```

Cela va:
- Créer les tables SQLAlchemy
- Charger les données Excel depuis `./inputs/`
- Valider les données via Pydantic
- Générer un rapport de synthèse

### 3. Tester les requêtes SQL

```bash
python sql_tool.py
```

### 4. Visualiser la documentation

```bash
python DATABASE_DOCUMENTATION.py
```

## 📊 Exemples de Requêtes

### Requête SQL Simple: Statistiques d'un joueur

```python
from sql_tool import SQLTool
from utils.database import SessionLocal

sql_tool = SQLTool(SessionLocal())

# Obtenir les stats de saison
result = sql_tool.player_stats("Stephen Curry", season=2024)
print(sql_tool.format_results(result))
```

**Output:**
```
             name  avg_points  avg_assists  avg_rebounds  avg_3p_pct
Stephen Curry     28.4        7.2          4.6           42.5
```

### Requête de Comparaison: Deux joueurs

```python
result = sql_tool.player_comparison("Stephen Curry", "Kevin Durant", season=2024)
print(sql_tool.format_results(result))
```

**Output:**
```
         name  avg_points  avg_assists  avg_rebounds  avg_3p_pct
Stephen Curry     28.4        7.2          4.6           42.5
Kevin Durant      29.1        5.3          6.8           45.2
```

### Requête Complexe: Leaders à 3 points

```sql
SELECT 
    p.name,
    t.name AS team,
    ROUND(AVG(s.three_point_percentage), 1) AS avg_3p_pct,
    ROUND(AVG(s.three_pointers_made), 1) AS avg_3pm,
    COUNT(*) AS games
FROM stats s
JOIN players p ON s.player_id = p.id
JOIN teams t ON p.team_id = t.id
JOIN matches m ON s.match_id = m.id
WHERE m.season = 2024 AND s.three_pointers_attempted >= 3
GROUP BY p.id, p.name, t.id, t.name
HAVING COUNT(*) >= 10
ORDER BY avg_3p_pct DESC
LIMIT 20;
```

## 🔗 Intégration avec l'Agent RAG

### Configuration du RAG Agent avec SQL Tool

```python
# Dans evaluate_ragas.py ou votre agent

from sql_tool import create_sql_tool
from utils.database import SessionLocal

class RAGEvaluatorWithSQL:
    def __init__(self):
        self.vs = VectorStoreManager()
        self.client = MistralClient(api_key=MISTRAL_API_KEY)
        
        # Ajouter le SQL Tool
        self.db = SessionLocal()
        self.sql_tool = create_sql_tool(self.db)
    
    async def answer_question(self, question: str) -> str:
        """
        Répondre aux questions en utilisant RAG + SQL
        """
        # Détecter le type de requête
        if self._is_numerical_query(question):
            # Utiliser le SQL Tool
            result = self.sql_tool.query_nba_stats(question)
            return result
        else:
            # Utiliser le RAG standard
            return self.query_rag(question).answer
    
    def _is_numerical_query(self, question: str) -> bool:
        """Détecter si c'est une requête numérique/SQL"""
        sql_keywords = [
            "combien", "quel", "meilleur", "compare", "moyenne",
            "points", "assists", "pourcentage", "victoires", "saison"
        ]
        return any(kw in question.lower() for kw in sql_keywords)
```

## 📈 Cas d'Usage Typiques

### 1. Comparaison Domicile/Extérieur

**Question:** "Comment performe LeBron James à domicile vs en extérieur?"

```sql
SELECT 
    p.name,
    CASE WHEN m.home_team_id = p.team_id THEN 'Home' ELSE 'Away' END AS location,
    ROUND(AVG(s.points), 1) AS avg_points,
    ROUND(AVG(s.assists), 1) AS avg_assists,
    COUNT(*) AS games
FROM stats s
JOIN players p ON s.player_id = p.id
JOIN matches m ON s.match_id = m.id
WHERE p.name LIKE 'LeBron James' AND m.season = 2024
GROUP BY p.id, location
ORDER BY location;
```

### 2. Tendances Saisonnières

**Question:** "Quel est le meilleur mois de Stephen Curry?"

```sql
SELECT 
    EXTRACT(MONTH FROM m.date) AS month,
    ROUND(AVG(s.points), 1) AS avg_points,
    ROUND(AVG(s.assists), 1) AS avg_assists,
    COUNT(*) AS games
FROM stats s
JOIN players p ON s.player_id = p.id
JOIN matches m ON s.match_id = m.id
WHERE p.name LIKE 'Stephen Curry' AND m.season = 2024
GROUP BY EXTRACT(MONTH FROM m.date)
ORDER BY avg_points DESC;
```

### 3. Matchups Spécifiques

**Question:** "Comment Curry performe contre les Warriors opponents?"

```sql
SELECT 
    p.name,
    t.name AS opponent,
    COUNT(*) AS games,
    ROUND(AVG(s.points), 1) AS avg_points,
    ROUND(AVG(s.three_point_percentage), 1) AS avg_3p_pct,
    MAX(s.points) AS max_points
FROM stats s
JOIN players p ON s.player_id = p.id
JOIN matches m ON s.match_id = m.id
JOIN teams t ON (CASE 
    WHEN m.home_team_id = p.team_id THEN m.away_team_id
    ELSE m.home_team_id
END) = t.id
WHERE p.name LIKE 'Stephen Curry'
GROUP BY p.id, p.name, t.id, t.name
ORDER BY avg_points DESC;
```

### 4. Agrégations Multi-Critères

**Question:** "Quels sont les leaders en assists par position?"

```sql
SELECT 
    p.position,
    p.name,
    ROUND(AVG(s.assists), 1) AS avg_assists,
    ROUND(AVG(s.points), 1) AS avg_points,
    COUNT(*) AS games
FROM stats s
JOIN players p ON s.player_id = p.id
JOIN matches m ON s.match_id = m.id
WHERE m.season = 2024 AND p.position IS NOT NULL
GROUP BY p.position, p.id, p.name
ORDER BY p.position, avg_assists DESC;
```

## 🔒 Sécurité & Validation

### Validation des Données

- **Pydantic Models** pour la validation à l'ingestion
- **Type checking** pour les colonnes
- **Range validation** pour les pourcentages (0-100)

### Sécurité des Requêtes

- **Parameterized queries** pour éviter les injections SQL
- **Whitelist keywords** dans les requêtes générées
- **Query validation** avec forbidden keywords blocking

### Gestion des Erreurs

```python
from sql_tool import SQLTool

sql_tool = SQLTool()

# Safe query execution
result = sql_tool.safe_query(
    "SELECT * FROM players WHERE name LIKE :player_name",
    {"player_name": "%Curry%"}
)

if result is not None:
    print(sql_tool.format_results(result))
else:
    print("Query failed or no results")
```

## 📝 Maintenance

### Ajouter de nouvelles statistiques

1. Mettre à jour `db_schema.py` avec les nouvelles colonnes
2. Créer une migration (ou recréer la base)
3. Ajouter les validations dans `data_validators.py`
4. Ajouter des templates SQL dans `sql_tool.py`

### Optimiser les performances

```sql
-- Ajouter des index
CREATE INDEX idx_player_name ON players(name);
CREATE INDEX idx_match_date ON matches(date);
CREATE INDEX idx_stat_player ON stats(player_id);
CREATE INDEX idx_stat_match ON stats(match_id);

-- Ou via SQLAlchemy
from sqlalchemy import Index

class Stat(Base):
    __table_args__ = (
        Index('idx_stat_player', 'player_id'),
        Index('idx_stat_match', 'match_id'),
    )
```

## 📚 Ressources

- **Documentation SQL:** `DATABASE_DOCUMENTATION.py`
- **Exemples de requêtes:** `sql_tool.py` (FEW_SHOT_EXAMPLES)
- **Schéma complet:** `db_schema.py`
- **Pipeline ingestion:** `load_excel_to_db.py`

## ✅ Checklist de Déploiement

- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Fichiers Excel dans `./inputs/`
- [ ] Base de données initialisée
- [ ] SQL Tool testé
- [ ] Intégration RAG effectuée
- [ ] Requêtes few-shot validées
- [ ] Documentation accessible

## 🐛 Troubleshooting

### Problème: "No module named 'sqlalchemy'"
**Solution:** `pip install sqlalchemy==2.0.23`

### Problème: "nba_stats.db not found"
**Solution:** Exécuter `python integrate_sql.py` d'abord

### Problème: "Excel files not loaded"
**Solution:** Vérifier que les fichiers sont dans `./inputs/`

### Problème: "Query returns no results"
**Solution:** Vérifier l'orthographe du joueur/équipe (case-insensitive)

## 📞 Support

Pour des questions ou des améliorations, consultez:
- `DATABASE_DOCUMENTATION.py` pour la structure complète
- `sql_tool.py` pour les exemples few-shot
- `load_excel_to_db.py` pour la validation des données
