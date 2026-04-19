# utils/load_excel_to_db.py
"""
Charge les 6 onglets du fichier Excel `regular NBA_Corr_Sql.xlsx` dans la base
PostgreSQL `oc_mlops_projet_3`.

Correspondance onglets → tables :
  teams                            → public.teams
  players                          → public.players
  analyse_joueurs_une_equipe       → public.analyse_joueurs_une_equipe
  analyse_nbr_joueurs_et_points_p  → public.analyse_nbr_joueurs_et_points_par_equipe
  analyse_top_15_joueurs_nombre_p  → public.analyse_top_15_joueurs_nombre_points
  stats_joueurs_saison_reguliere   → public.stats_joueurs_saison_reguliere

Les float stockés sous forme de chaîne avec virgule (ex. "51,9") sont convertis
en float Python standard avant l'insertion.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, ValidationError
import logfire
from utils.config import (
    PG_HOST, PG_PORT, PG_DB,
    PG_ADMIN, PG_ADMIN_PASSWORD,
    EXCEL_INPUTS_FOR_SQL, LOGFIRE_TOKEN,
)

if LOGFIRE_TOKEN:
    logfire.configure(token=LOGFIRE_TOKEN, send_to_logfire=True)
else:
    logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic() # Logfire enregistre automatiquement la validation de Pydantic

# ---------------------------------------------------------------------------
# Modèles Pydantic — un par table
# ---------------------------------------------------------------------------

class TeamModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str = Field(max_length=3)
    name: Optional[str] = None


class PlayerModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: int
    team_id: str = Field(max_length=3)
    name: Optional[str] = None
    age: Optional[int] = None


class AnalyseJoueursUneEquipeModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    team_id: str = Field(max_length=3)
    player_id: int
    sum_oreb: Optional[int] = None
    sum_dreb: Optional[int] = None
    sum_pie: Optional[float] = None
    sum_ast: Optional[int] = None
    sum_stl: Optional[int] = None
    sum_blk: Optional[int] = None


class AnalyseNbrJoueursPointsModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    team_id: str = Field(max_length=3)
    sum_players_team: Optional[int] = None
    sum_points_team: Optional[int] = None


class AnalyseTop15Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    player_id: int
    pts: Optional[int] = None
    fgm: Optional[int] = None
    fg_perc: Optional[float] = None
    field_3p_perc: Optional[float] = Field(None, alias="3p_perc")   # alias : nom commence par un chiffre
    ft_perc: Optional[float] = None
    oreb: Optional[int] = None
    pie: Optional[float] = None


class StatsJoueursSaisonModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    player_id: int
    team_id: str
    gp: Optional[int] = None
    w: Optional[int] = None
    l: Optional[int] = None
    min: Optional[float] = None
    pts: Optional[int] = None
    fgm: Optional[int] = None
    fga: Optional[int] = None
    fg_perc: Optional[float] = None
    field_15_00: Optional[int] = Field(None, alias="15_00")         # alias : nom commence par un chiffre
    field_3pa: Optional[int] = Field(None, alias="3pa")             # alias : nom commence par un chiffre
    field_3p_perc: Optional[float] = Field(None, alias="3p_perc")   # alias : nom commence par un chiffre
    ftm: Optional[int] = None
    fta: Optional[int] = None
    ft_perc: Optional[float] = None
    oreb: Optional[int] = None
    dreb: Optional[int] = None
    reb: Optional[int] = None
    ast: Optional[int] = None
    tov: Optional[int] = None
    stl: Optional[int] = None
    blk: Optional[int] = None
    pf: Optional[int] = None
    fp: Optional[int] = None
    dd2: Optional[int] = None
    td3: Optional[int] = None
    plus_minus: Optional[float] = None
    offrtg: Optional[float] = None
    defrtg: Optional[float] = None
    netrtg: Optional[float] = None
    ast_perc: Optional[float] = None
    ast_div_to: Optional[float] = None
    ast_ratio: Optional[float] = None
    oreb_perc: Optional[float] = None
    dreb_perc: Optional[float] = None
    reb_perc: Optional[float] = None
    to_ratio: Optional[float] = None
    efg_perc: Optional[float] = None
    ts_perc: Optional[float] = None
    usg_perc: Optional[float] = None
    pace: Optional[float] = None
    pie: Optional[float] = None
    poss: Optional[int] = None


# Mapping table → modèle Pydantic de validation
TABLE_MODELS = {
    "teams":                                    TeamModel,
    "players":                                  PlayerModel,
    "analyse_joueurs_une_equipe":               AnalyseJoueursUneEquipeModel,
    "analyse_nbr_joueurs_et_points_par_equipe": AnalyseNbrJoueursPointsModel,
    "analyse_top_15_joueurs_nombre_points":     AnalyseTop15Model,
    "stats_joueurs_saison_reguliere":           StatsJoueursSaisonModel,
}

# ---------------------------------------------------------------------------
# Mapping onglet Excel → nom exact de la table PostgreSQL
# ---------------------------------------------------------------------------
# Les noms d'onglets Excel sont tronqués à 31 caractères, d'où la différence
# entre la clé (onglet) et la valeur (nom de table complet).
SHEET_TABLE_MAP = {
    "teams":                            "teams",
    "players":                          "players",
    "analyse_joueurs_une_equipe":       "analyse_joueurs_une_equipe",
    "analyse_nbr_joueurs_et_points_p":  "analyse_nbr_joueurs_et_points_par_equipe",
    "analyse_top_15_joueurs_nombre_p":  "analyse_top_15_joueurs_nombre_points",
    "stats_joueurs_saison_reguliere":   "stats_joueurs_saison_reguliere",
}

# Ordre d'insertion respectant les clés étrangères (parents avant enfants)
INSERT_ORDER = [
    "teams",
    "players",
    "analyse_joueurs_une_equipe",
    "analyse_nbr_joueurs_et_points_par_equipe",
    "analyse_top_15_joueurs_nombre_points",
    "stats_joueurs_saison_reguliere",
]


# ---------------------------------------------------------------------------
# Conversion des float stockés comme chaînes avec virgule décimale
# ---------------------------------------------------------------------------
def convert_comma_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes textuelles au format décimal français ("51,9" → 51.9)."""
    converted_cols: list[str] = []
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str)
            # Détecte les valeurs du type "-?\d+,\d+"
            if sample.str.match(r"^-?\d+,\d+$").any():
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",  # valeurs non convertibles → NaN
                )
                converted_cols.append(col)
    if converted_cols:
        logfire.info("Conversion virgule→point", colonnes=converted_cols)
    return df


# ---------------------------------------------------------------------------
# Validation Pydantic d'un DataFrame
# ---------------------------------------------------------------------------
def validate_dataframe(table_name: str, df: pd.DataFrame) -> None:
    """Valide chaque ligne du DataFrame avec le modèle Pydantic correspondant.

    Raises:
        ValueError: Si au moins une ligne est invalide.
    """
    model_class = TABLE_MODELS[table_name]
    errors: list[str] = []

    with logfire.span("Validation Pydantic {table}", table=table_name):
        for i, record in enumerate(df.to_dict(orient="records")):
            # Remplace les NaN pandas par None (incompatibles avec les champs Optional Pydantic)
            clean = {k: (None if pd.isna(v) else v) for k, v in record.items()}
            try:
                model_class.model_validate(clean)
            except ValidationError as e:
                errors.append(f"    Ligne {i + 1} : {e.error_count()} erreur(s) → {e.errors(include_url=False)}")

        if errors:
            logfire.warning("{table} : {n} ligne(s) invalide(s)", table=table_name, n=len(errors), details=errors[:5])
            print(f"  ⚠️  {table_name} : {len(errors)} ligne(s) invalide(s) :")
            for msg in errors[:5]:
                print(msg)
            if len(errors) > 5:
                print(f"    … et {len(errors) - 5} autre(s) erreur(s).")
            raise ValueError(f"Validation Pydantic échouée pour la table '{table_name}'.")
        else:
            logfire.info("{table} : validation OK ({n} lignes)", table=table_name, n=len(df))
            print(f"  ✅  {table_name} : validation Pydantic OK ({len(df)} lignes).")


# ---------------------------------------------------------------------------
# Chargement d'un DataFrame dans une table PostgreSQL
# ---------------------------------------------------------------------------
def load_dataframe(cursor, table_name: str, df: pd.DataFrame) -> None:
    """Insère en masse les lignes d'un DataFrame dans une table PostgreSQL (bulk insert).

    Les noms de colonnes commençant par un chiffre sont entourés de guillemets doubles.
    Les doublons sont ignorés via `ON CONFLICT DO NOTHING`.
    """
    # Les colonnes dont le nom commence par un chiffre doivent être quotées en SQL
    cols_sql = ", ".join(
        f'"{c}"' if (not c.replace("_", "").isalpha() and c[0].isdigit()) else c
        for c in df.columns
    )

    # NaN pandas → None pour que psycopg2 les transmette comme NULL
    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False)
    ]

    with logfire.span("INSERT {table} ({n} lignes)", table=table_name, n=len(rows)):
        # Utilise execute_values de psycopg2 pour faire un bulk insert efficace
        sql = f"INSERT INTO {table_name} ({cols_sql}) VALUES %s ON CONFLICT DO NOTHING"
        execute_values(cursor, sql, rows)
    logfire.info("{table} insérée", table=table_name, lignes=len(rows))
    print(f"  ✅  {table_name} : {len(rows)} ligne(s) insérée(s).")



# ---------------------------------------------------------------------------
# Fonction principale : pipeline ETL (Extract → Transform → Load)
# ---------------------------------------------------------------------------
def main():
    """Orchestre la lecture du fichier Excel, la validation Pydantic et le chargement PostgreSQL."""
    with logfire.span("Chargement Excel → PostgreSQL"):
        print(f"📂  Lecture : {EXCEL_INPUTS_FOR_SQL}")
        sheet_data: dict[str, pd.DataFrame] = {}

        # --- Extract ---
        with logfire.span("Lecture fichier Excel"):
            xls = pd.ExcelFile(EXCEL_INPUTS_FOR_SQL)
            for sheet_name, table_name in SHEET_TABLE_MAP.items():
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df = convert_comma_floats(df)
                sheet_data[table_name] = df
                logfire.info("Onglet lu", onglet=sheet_name, table=table_name, lignes=len(df))
                print(f"  📄  '{sheet_name}' → '{table_name}' : {len(df)} lignes")

        # --- Transform / Validate ---
        # Validation avant connexion : inutile d'ouvrir une transaction si les données sont invalides
        print("\n🔍  Validation Pydantic…")
        with logfire.span("Validation Pydantic (toutes tables)"):
            for table_name in INSERT_ORDER:
                validate_dataframe(table_name, sheet_data[table_name])

        # --- Load ---
        print(f"\n🔌  Connexion PostgreSQL ({PG_ADMIN}@{PG_HOST}:{PG_PORT}/{PG_DB})…")
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_ADMIN, password=PG_ADMIN_PASSWORD)
        logfire.info("Connexion PostgreSQL établie", host=PG_HOST, db=PG_DB, user=PG_ADMIN)
        conn.autocommit = False  # transaction manuelle : TRUNCATE + INSERT sont atomiques

        try:
            with conn.cursor() as cur:
                # Diffère la vérification des FK jusqu'au COMMIT pour éviter
                # les violations temporaires entre TRUNCATE et INSERT
                cur.execute("SET CONSTRAINTS ALL DEFERRED;")

                print("\n🗑️   Vidage des tables…")
                with logfire.span("TRUNCATE toutes les tables"):
                    for table_name in reversed(INSERT_ORDER):  # enfants avant parents
                        cur.execute(f"TRUNCATE TABLE {table_name} CASCADE;")
                        logfire.info("Table vidée", table=table_name)
                        print(f"  🗑️   {table_name} vidée.")

                print("\n📥  Insertion des données…")
                with logfire.span("INSERT toutes les tables"):
                    for table_name in INSERT_ORDER:
                        load_dataframe(cur, table_name, sheet_data[table_name])

            conn.commit()  # les FK différées sont vérifiées ici
            logfire.info("Chargement terminé avec succès")
            print("\n🎉  Chargement terminé avec succès !")

        except Exception as e:
            conn.rollback()
            logfire.error("Échec du chargement", erreur=str(e))
            print(f"\n❌  Erreur : {e}")
            raise
        finally:
            conn.close()


if __name__ == "__main__":
    main()

