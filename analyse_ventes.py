"""
TP - Analyse de données avec Pandas et Matplotlib (CC-AIBD)
============================================================
Données : CC-AIBD-DataAnalyse-donnees.xlsx (7 feuilles : ventes d'une marque de vélos).

Prérequis (Python 3.11) :
    pip install pandas pandas-stubs openpyxl matplotlib scipy
    (scipy n'est utilisé que pour les tests statistiques de la partie 5)

Utilisation :
    python tp_analyse_ventes.py [chemin_du_fichier.xlsx] [--show]
        --show : affiche aussi les graphiques à l'écran (sinon : uniquement sauvegarde PNG)

Le fichier est découpé en cellules "# %%" (exécutables une par une dans VS Code / PyCharm).
Tous les graphiques sont sauvegardés en PNG dans le dossier ./figures
"""
# %% Imports et configuration générale
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SHOW = "--show" in sys.argv
if not SHOW:
    matplotlib.use("Agg")  # pas de fenêtre : on sauvegarde seulement les PNG

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

DATA_FILE = Path(ARGS[0]) if ARGS else Path("CC-AIBD-DataAnalyse-donnees.xlsx")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def titre(texte: str) -> None:
    print("\n" + "=" * 100 + f"\n{texte}\n" + "=" * 100)


def usd(x, pos=None) -> str:
    """Formate un montant en dollars (k / M) pour les axes et les étiquettes."""
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.1f} M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f} k"
    return f"${x:,.0f}"


def sauver(fig, nom: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / nom, dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)
    print(f"[figure sauvegardée] {OUT_DIR / nom}")


# %% ===========================================================================
# JOUR 1 - 1. BASES DE PANDAS
# ==============================================================================
titre("1. Bases de Pandas : configuration de l'affichage")

pd.set_option("display.max_columns", 12)      # au maximum 12 colonnes (>= 6 affichées dès qu'elles existent)
pd.set_option("display.max_rows", 100)        # au maximum 100 lignes
pd.set_option("display.min_rows", 20)         # si > 100 lignes : on en montre 20 (10 début + 10 fin)
pd.set_option("display.max_colwidth", 20)     # au maximum 20 caractères par colonne
pd.set_option("display.width", 240)           # au maximum 240 caractères par ligne

rng = np.random.default_rng(42)
df_alea = pd.DataFrame(
    rng.random((200, 20)),
    columns=[f"colonne_{i:02d}" for i in range(20)],
)
affichage = repr(df_alea)
lignes = affichage.splitlines()
print(df_alea)
print("\nVérification de l'affichage :")
print(f"  - largeur de la plus longue ligne affichée : {max(len(l) for l in lignes)} (<= 240)")
print(f"  - nombre de lignes affichées (en-tête + données + '...' + pied) : {len(lignes)} (<= 100 lignes de contenu)")
print(f"  - colonnes réellement affichées : 12 sur {df_alea.shape[1]} (max_columns = 12, donc >= 6)")

titre("1. Concepts de base de Pandas")
print(
    """
* Series  : tableau 1D étiqueté (une colonne). Valeurs homogènes (un dtype) + un Index.
* DataFrame : tableau 2D = ensemble de Series partageant le même Index (lignes) ;
             chaque colonne a son propre dtype. C'est l'objet central de Pandas.
* Index  : les étiquettes des lignes (ou des colonnes). Il permet l'alignement automatique
           des données lors des opérations/jointures, la sélection rapide (.loc) et peut
           être un index de dates (DatetimeIndex) pour les séries temporelles.
"""
)
s_demo = pd.Series([10, 20, 30], index=["a", "b", "c"], name="demo")
print("Exemple de Series :\n", s_demo)
print("Son index :", s_demo.index.tolist(), "| .loc['b'] =", s_demo.loc["b"])
print("Exemple de DataFrame :\n", df_alea[["colonne_00", "colonne_01"]].head(3))

# %% ===========================================================================
# 2. CHARGEMENT ET EXPLORATION
# ==============================================================================
titre("2. Chargement : une feuille Excel -> un DataFrame")

frames = pd.read_excel(DATA_FILE, sheet_name=None)  # dict {nom_feuille: DataFrame}
orders = frames["Sales Order_data"]
territories = frames["Sales Territory_data"]
sales = frames["Sales_data"]
resellers = frames["Reseller_data"]
dates = frames["Date_data"]
products = frames["Product_data"]
customers = frames["Customer_data"]

DONNEES = {
    "orders": orders,
    "territories": territories,
    "sales": sales,
    "resellers": resellers,
    "dates": dates,
    "products": products,
    "customers": customers,
}

for nom, df in DONNEES.items():
    titre(f"DataFrame '{nom}' - {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print("-> 5 premières lignes :")
    print(df.head())
    print("\n-> Types des colonnes :")
    print(df.dtypes)
    print("\n-> info() :")
    df.info()
    print("\n-> describe() :")
    print(df.describe())

titre("2. Colonnes dont le type semble incorrect vis-à-vis de l'usage")
print(
    """
* sales.OrderDateKey / DueDateKey (int64) et ShipDateKey (float64, à cause des NaN) :
  ce sont des dates encodées 'AAAAMMJJ' -> à convertir en datetime64.
* sales.ShipDateKey : float64 parce qu'il y a des valeurs manquantes (commandes non expédiées).
* sales.Unit Price Discount Pct : int64 alors qu'un pourcentage/ratio est un nombre décimal.
* products.Color : 56 valeurs manquantes.
* orders.Channel, products.Category / Subcategory / Model : texte répétitif -> dtype 'category'.
* Les clés (…Key) sont des identifiants : le type entier est correct, mais on ne fait pas de calcul dessus.
"""
)

titre("2. Dispersion du coût standard des produits du catalogue")
cs = products["Standard Cost"]
print(f"Médiane : {cs.median():,.4f} | Écart-type : {cs.std():,.4f} | Variance : {cs.var():,.4f}")
print("\nPar catégorie de produit :")
print(products.groupby("Category")["Standard Cost"].agg(mediane="median", ecart_type="std", variance="var"))

# %% ===========================================================================
# 3. NETTOYAGE DES DONNÉES
# ==============================================================================
titre("3. Cellules vides (avant nettoyage)")


def colonnes_vides() -> dict:
    return {nom: df.columns[df.isna().any()].tolist() for nom, df in DONNEES.items() if df.isna().any().any()}


print("Colonnes contenant des valeurs vides :", colonnes_vides())
print("Détail :", {nom: df.isna().sum()[df.isna().sum() > 0].to_dict() for nom, df in DONNEES.items() if df.isna().any().any()})

# Valeurs de repli :
#  - products.Color : 'Sans couleur' (produits pour lesquels la couleur n'a pas de sens)
#  - sales.ShipDateKey : -1 = convention "[Not Applicable]" déjà utilisée dans le jeu de données
#    pour les clés inconnues (commande pas encore expédiée). La date convertie restera NaT.
products["Color"] = products["Color"].fillna("Sans couleur")
sales["ShipDateKey"] = sales["ShipDateKey"].fillna(-1).astype("int64")
print("\nColonnes contenant des valeurs vides (après nettoyage) :", colonnes_vides() or "aucune")

# --- Conversion des types ----------------------------------------------------------
titre("3. Conversion des types")


def cle_vers_date(serie: pd.Series) -> pd.Series:
    return pd.to_datetime(serie.astype(str), format="%Y%m%d", errors="coerce")  # -1 -> NaT


sales["Order Date"] = cle_vers_date(sales["OrderDateKey"])
sales["Due Date"] = cle_vers_date(sales["DueDateKey"])
sales["Ship Date"] = cle_vers_date(sales["ShipDateKey"])
sales["Is Shipped"] = sales["ShipDateKey"] != -1
sales["Unit Price Discount Pct"] = sales["Unit Price Discount Pct"].astype("float64")
orders["Channel"] = orders["Channel"].astype("category")
for col in ["Category", "Subcategory", "Model"]:
    products[col] = products[col].astype("category")
print(sales[["Order Date", "Due Date", "Ship Date", "Is Shipped", "Unit Price Discount Pct"]].dtypes)
print(f"Commandes non expédiées (Ship Date = NaT) : {(~sales['Is Shipped']).sum()}")

# --- Vérification du total des ventes --------------------------------------------------
titre("3. Vérification de Sales Amount (remise comprise)")
check_ext = np.isclose(sales["Extended Amount"], sales["Unit Price"] * sales["Order Quantity"]).all()
check_cost = np.isclose(sales["Total Product Cost"], sales["Product Standard Cost"] * sales["Order Quantity"]).all()
print(f"Extended Amount = Unit Price x Order Quantity pour toutes les lignes : {check_ext}")
print(f"Total Product Cost = Standard Cost x Quantity pour toutes les lignes : {check_cost}")

sales["Expected Sales Amount"] = sales["Extended Amount"] * (1 - sales["Unit Price Discount Pct"])
sales["Amount Gap"] = sales["Sales Amount"] - sales["Expected Sales Amount"]
ecarts = sales[~np.isclose(sales["Amount Gap"], 0, atol=0.01)]
print(f"Lignes avec une remise non nulle (colonne Discount) : {(sales['Unit Price Discount Pct'] != 0).sum()}")
print(f"Lignes où Sales Amount != Extended Amount x (1 - remise) : {len(ecarts)} sur {len(sales)}")
print(f"Écart cumulé : {ecarts['Amount Gap'].sum():,.2f} $ (écart max : {ecarts['Amount Gap'].min():,.2f} $)")
sales["Implied Discount Pct"] = np.where(
    sales["Extended Amount"] != 0, 1 - sales["Sales Amount"] / sales["Extended Amount"], 0.0
)
print("Remise implicite sur les lignes en écart :")
print(sales.loc[ecarts.index, "Implied Discount Pct"].describe())
print(
    "-> CONCLUSION : la colonne de remise vaut 0 partout, mais "
    f"{len(ecarts)} lignes ont un Sales Amount inférieur au montant brut : une remise est donc appliquée "
    "sans être renseignée (écart toujours négatif, jamais de surfacturation). "
    "Pour les analyses on utilise Sales Amount, qui est le montant réellement facturé."
)

# --- Nouvelles colonnes -----------------------------------------------------------------
titre("3. Nouvelles colonnes : Price ratio, Line number, Margin")
sales["Price ratio"] = sales["Unit Price"] / sales["Product Standard Cost"].where(sales["Product Standard Cost"] > 0)
products["Price ratio"] = products["List Price"] / products["Standard Cost"].where(products["Standard Cost"] > 0)
orders["Line number"] = (
    orders["Sales Order Line"].str.extract(r"-\s*(\d+)\s*$", expand=False).astype("int64")
)
sales["Margin"] = sales["Sales Amount"] - sales["Total Product Cost"]
print(sales[["Unit Price", "Product Standard Cost", "Price ratio", "Sales Amount", "Total Product Cost", "Margin"]].head())
print(orders[["Sales Order Line", "Line number"]].head())
print(f"Lignes de vente à marge négative : {(sales['Margin'] < 0).sum()} ({(sales['Margin'] < 0).mean():.1%})")

# %% ===========================================================================
# 4. ANALYSE EXPLORATOIRE (EDA)
# ==============================================================================
# Thème graphique commun (police + palette de couleurs)
PALETTE = ["#1f4e79", "#2e86ab", "#f18f01", "#c73e1d", "#6a994e", "#8e6c8a", "#e0a458", "#4a4e69", "#99c1b9", "#d1495b"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.prop_cycle": cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

fy_map = dates.set_index("DateKey")["Fiscal Year"]
fq_map = dates.set_index("DateKey")["Fiscal Quarter"]


def enrichir(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute territoire, produit et période fiscale aux lignes de vente."""
    out = df.merge(
        territories[["SalesTerritoryKey", "Region", "Country", "Group"]],
        on="SalesTerritoryKey", how="left", validate="m:1",
    ).merge(
        products[["ProductKey", "Product", "Model", "Subcategory", "Category", "List Price"]],
        on="ProductKey", how="left", validate="m:1",
    )
    out["Territory"] = out["Country"].where(out["Country"] == out["Region"], out["Country"] + " - " + out["Region"])
    out["Territory label"] = out["Country"] + "\n" + out["Region"]
    out["Fiscal Year"] = out["OrderDateKey"].map(fy_map)
    out["Fiscal Quarter"] = out["OrderDateKey"].map(fq_map)
    return out


ventes = enrichir(sales)
assert len(ventes) == len(sales)

titre("4. Total des ventes par territoire (barres)")
par_terr = ventes.groupby("Territory label")["Sales Amount"].sum().sort_values(ascending=False)
print(par_terr.map(usd))
fig, ax = plt.subplots(figsize=(15, 6))
barres = ax.bar(par_terr.index, par_terr.values, color=PALETTE[0])
ax.bar_label(barres, labels=[usd(v) for v in par_terr.values], padding=3)
ax.tick_params(axis="x", labelrotation=0)  # intitulés horizontaux : Country / Region sur 2 lignes
ax.yaxis.set_major_formatter(usd)
ax.set_title("Total des ventes par territoire")
ax.set_xlabel("Territoire (Country / Region)")
ax.set_ylabel("Ventes ($)")
sauver(fig, "ventes_par_territoire.png")

titre("4. Total des ventes par modèle de produit")
par_modele = ventes.groupby("Model", observed=True)["Sales Amount"].sum().sort_values(ascending=False)
top_n = 15
top_modeles = par_modele.head(top_n)
autres = par_modele.iloc[top_n:].sum()
print(top_modeles.map(usd))
print(f"... + {len(par_modele) - top_n} autres modèles : {usd(autres)}")
affichage_modeles = pd.concat([top_modeles, pd.Series({f"Autres ({len(par_modele) - top_n} modèles)": autres})])
fig, ax = plt.subplots(figsize=(11, 8))
couleurs = [PALETTE[1]] * top_n + ["#9aa0a6"]
barres = ax.barh(affichage_modeles.index[::-1], affichage_modeles.values[::-1], color=couleurs[::-1])
ax.bar_label(barres, labels=[usd(v) for v in affichage_modeles.values[::-1]], padding=3)
ax.xaxis.set_major_formatter(usd)
ax.set_title(f"Ventes par modèle de produit (top {top_n} + autres)")
ax.set_xlabel("Ventes ($)")
ax.set_xlim(0, affichage_modeles.max() * 1.15)
sauver(fig, "ventes_par_modele.png")

titre("4. Ventes par jour, tendances saisonnières et moyennes lissées 7 / 30 jours")
jour = (
    ventes.groupby("Order Date")["Sales Amount"].sum()
    .reindex(pd.date_range(ventes["Order Date"].min(), ventes["Order Date"].max(), freq="D"), fill_value=0.0)
)
jour.index.name = "Date"
quotidien = jour.to_frame("Ventes")
quotidien["Moyenne 7j"] = quotidien["Ventes"].rolling(7).mean()
quotidien["Moyenne 30j"] = quotidien["Ventes"].rolling(30).mean()
print(quotidien.tail(8))
saison = quotidien["Ventes"].groupby(quotidien.index.month).mean()
print("\nVente moyenne par jour selon le mois de l'année :")
print(saison.map(usd))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1.6]})
ax1.plot(quotidien.index, quotidien["Ventes"], color="#9aa0a6", alpha=0.35, lw=0.8, label="Ventes quotidiennes")
ax1.plot(quotidien.index, quotidien["Moyenne 7j"], color=PALETTE[1], lw=1.4, label="Moyenne mobile 7 jours")
ax1.plot(quotidien.index, quotidien["Moyenne 30j"], color=PALETTE[3], lw=2.2, label="Moyenne mobile 30 jours")
ax1.yaxis.set_major_formatter(usd)
ax1.set_title("Ventes quotidiennes et moyennes lissées")
ax1.set_ylabel("Ventes ($)")
ax1.legend(loc="upper left")
noms_mois = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
barres = ax2.bar(noms_mois, saison.values, color=PALETTE[0])
ax2.bar_label(barres, labels=[usd(v) for v in saison.values], padding=2, fontsize=8)
ax2.yaxis.set_major_formatter(usd)
ax2.set_title("Saisonnalité : vente moyenne par jour selon le mois")
ax2.set_ylabel("Ventes / jour ($)")
sauver(fig, "tendances_journalieres.png")

titre("4. Tableaux croisés dynamiques")
pivot_prod_terr = ventes.pivot_table(
    index="Product", columns="Region", values="Sales Amount", aggfunc="sum", fill_value=0, margins=True, margins_name="TOTAL"
)
top_produits = pivot_prod_terr.drop(index="TOTAL").sort_values("TOTAL", ascending=False).head(10)
print("Somme des ventes par produit (top 10) et par région :")
print(top_produits.round(0))
pivot_cat_terr = ventes.pivot_table(
    index="Category", columns="Group", values="Sales Amount", aggfunc="sum", fill_value=0, margins=True, margins_name="TOTAL", observed=True
)
print("\nSomme des ventes par catégorie et par groupe géographique :")
print(pivot_cat_terr.round(0))
pivot_annee_cat = ventes.pivot_table(
    index="Fiscal Year", columns="Category", values="Sales Amount", aggfunc="sum", fill_value=0, observed=True
)
print("\nSomme des ventes par année fiscale et par catégorie :")
print(pivot_annee_cat.round(0))

# %% ===========================================================================
# 5. ANALYSE STATISTIQUE
# ==============================================================================
titre("5. Statistiques des ventes (Sales Amount) par territoire")
stats_terr = ventes.groupby("Territory")["Sales Amount"].agg(
    moyenne="mean",
    mediane="median",
    ecart_type="std",
    minimum="min",
    q1=lambda s: s.quantile(0.25),
    q3=lambda s: s.quantile(0.75),
    maximum="max",
)
print(stats_terr.round(2))

titre("5. Corrélations")
for a, b in [("Sales Amount", "Margin"), ("Total Product Cost", "Margin")]:
    r, p = stats.pearsonr(ventes[a], ventes[b])
    rho, _ = stats.spearmanr(ventes[a], ventes[b])
    print(f"{a} vs {b} : Pearson r = {r:.4f} (p = {p:.2e}) | Spearman rho = {rho:.4f}")
print(
    "\nLecture : la corrélation linéaire entre ventes et marge est positive mais faible (le taux de marge varie "
    "beaucoup d'un produit/territoire à l'autre, certaines grosses ventes sont à marge négative). "
    "Le rang (Spearman) est plus parlant que la valeur brute : les plus grosses lignes ont tendance à avoir "
    "la plus grande marge. Entre coût total et marge, la corrélation linéaire est quasi nulle (les lignes les plus coûteuses "
    "ne sont pas les plus rentables) alors que Spearman reste modérément positive : la relation n'est pas linéaire."
)

titre("5. Territoires et produits les plus performants")
perf_terr = ventes.groupby("Territory").agg(Ventes=("Sales Amount", "sum"), Marge=("Margin", "sum"))
perf_terr["Taux de marge"] = perf_terr["Marge"] / perf_terr["Ventes"]
print("Territoires :")
print(perf_terr.sort_values("Ventes", ascending=False).round(3))
perf_prod = ventes.groupby("Product").agg(Ventes=("Sales Amount", "sum"), Marge=("Margin", "sum"), Quantite=("Order Quantity", "sum"))
print("\nTop 10 produits par ventes :")
print(perf_prod.sort_values("Ventes", ascending=False).head(10).round(0))
print("\nTop 10 produits par marge :")
print(perf_prod.sort_values("Marge", ascending=False).head(10).round(0))

titre("5. Tests statistiques")
groupes = [g["Sales Amount"].to_numpy() for _, g in ventes.groupby("Territory")]
std_terr = stats_terr["ecart_type"]
print("Écart-type des ventes par ligne selon le territoire :")
print(std_terr.round(2).to_string())
lev = stats.levene(*groupes, center="median")
kw = stats.kruskal(*groupes)
an = stats.f_oneway(*groupes)
print(f"\nH0 : les variances des ventes sont égales entre territoires (test de Levene) : W={lev.statistic:.2f}, p={lev.pvalue:.3e}")
print(f"H0 : les distributions sont identiques entre territoires (Kruskal-Wallis)    : H={kw.statistic:.2f}, p={kw.pvalue:.3e}")
print(f"H0 : les moyennes sont égales entre territoires (ANOVA)                      : F={an.statistic:.2f}, p={an.pvalue:.3e}")
t1 = stats.ttest_1samp(ventes["Margin"], 0.0)
print(f"H0 : la marge moyenne par ligne est nulle (t-test)                           : t={t1.statistic:.2f}, p={t1.pvalue:.3e}")
for nom, res in [("Levene", lev), ("Kruskal-Wallis", kw), ("ANOVA", an)]:
    verdict = "rejetée" if res.pvalue < 0.05 else "non rejetée"
    print(f"  -> {nom} : H0 {verdict} au seuil de 5 %")

# %% ===========================================================================
# 6. VISUALISATION AVANCÉE
# ==============================================================================
titre("6. Nuage de points : ventes (x), marge (y), prix unitaire (couleur)")
fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter(
    ventes["Sales Amount"], ventes["Margin"], c=ventes["Unit Price"], s=8, alpha=0.5, cmap="viridis",
    norm=LogNorm(vmin=ventes["Unit Price"].quantile(0.01), vmax=ventes["Unit Price"].max()),
)
cb = fig.colorbar(sc, ax=ax)
cb.set_label("Prix unitaire ($, échelle logarithmique)")
ax.xaxis.set_major_formatter(usd)
ax.yaxis.set_major_formatter(usd)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Relation entre ventes et marge par ligne de commande")
ax.set_xlabel("Total des ventes de la ligne ($)")
ax.set_ylabel("Marge ($)")
sauver(fig, "scatter_ventes_marge.png")

titre("6. Box plot : une boîte par territoire, valeurs = ventes totales par modèle de produit")
par_terr_modele = ventes.groupby(["Territory label", "Model"], observed=True)["Sales Amount"].sum()
ordre = par_terr.index.tolist()
donnees_box = [par_terr_modele.loc[t].to_numpy() for t in ordre]
fig, ax = plt.subplots(figsize=(15, 7))
bp = ax.boxplot(donnees_box, patch_artist=True, medianprops={"color": "black", "lw": 1.8}, flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})
for patch, couleur in zip(bp["boxes"], PALETTE * 2):
    patch.set_facecolor(couleur)
    patch.set_alpha(0.75)
ax.set_xticklabels(ordre)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(usd)
ax.set_title("Distribution des ventes par modèle de produit, pour chaque territoire")
ax.set_xlabel("Territoire")
ax.set_ylabel("Ventes par modèle ($, échelle log)")
sauver(fig, "boxplot_territoires.png")

titre("6. Heat map : matrice de corrélation")
cols_corr = ["Sales Amount", "Margin", "Total Product Cost", "Order Quantity", "Unit Price", "Price ratio"]
corr = ventes[cols_corr].corr()
print(corr.round(3))
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols_corr)), labels=cols_corr, rotation=40, ha="right")
ax.set_yticks(range(len(cols_corr)), labels=cols_corr)
for i in range(len(cols_corr)):
    for j in range(len(cols_corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(corr.values[i, j]) > 0.6 else "black", fontsize=9)
ax.grid(False)
fig.colorbar(im, ax=ax, label="Coefficient de corrélation de Pearson")
ax.set_title("Corrélations (dont ventes totales vs marge)")
sauver(fig, "heatmap_correlations.png")


def camembert(serie: pd.Series, titre_graph: str, nom_fichier: str, max_parts: int = 8) -> None:
    serie = serie.sort_values(ascending=False)
    if len(serie) > max_parts:
        reste = serie.iloc[max_parts - 1:].sum()
        serie = pd.concat([serie.iloc[: max_parts - 1], pd.Series({"Autres": reste})])
    total = serie.sum()
    fig, ax = plt.subplots(figsize=(10, 7))
    parts, _, _ = ax.pie(
        serie.values, autopct=lambda p: f"{p:.1f} %\n{usd(p * total / 100)}", startangle=90, counterclock=False,
        colors=PALETTE[: len(serie)], pctdistance=0.74, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9, "color": "white", "weight": "bold"},
    )
    ax.legend(parts, serie.index, title="Légende", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(titre_graph)
    sauver(fig, nom_fichier)


titre("6. Graphiques en secteur : parts de marché")
camembert(ventes.groupby("Category", observed=True)["Sales Amount"].sum(), "Part des ventes par catégorie de produit", "pie_categories.png")
camembert(ventes.groupby("Territory")["Sales Amount"].sum(), "Part des ventes par territoire", "pie_territoires.png")

# %% ===========================================================================
# 7. ANALYSE TEMPORELLE
# ==============================================================================
titre("7. Séries temporelles : mois / trimestre / année")
serie_ventes = ventes.set_index("Order Date")["Sales Amount"].sort_index()
mensuel = serie_ventes.resample("MS").sum()          # index = DatetimeIndex (1er jour du mois)
trimestriel = serie_ventes.resample("QS").sum()
annuel = serie_ventes.resample("YS").sum()
print("Ventes mensuelles (index = dates) :")
print(mensuel.map(usd).head(12))
print("\nVentes trimestrielles (trimestres civils) :")
print(trimestriel.to_frame("Ventes").assign(Trimestre=trimestriel.index.to_period("Q")).set_index("Trimestre")["Ventes"].map(usd))
print("\nVentes annuelles (années civiles) :")
print(annuel.to_frame("Ventes").assign(Annee=annuel.index.year).set_index("Annee")["Ventes"].map(usd))
print(
    "ATTENTION : les données couvrent du 01/07/2017 au 15/06/2020 -> les années civiles 2017 et 2020 sont partielles ; "
    "juin 2020 aussi. On ajoute donc l'analyse par exercice fiscal (juillet -> juin)."
)
fiscal = ventes.groupby("Fiscal Year")["Sales Amount"].sum()
print("\nVentes par exercice fiscal :")
print(fiscal.map(usd))
print("\nVentes par trimestre fiscal :")
print(ventes.groupby("Fiscal Quarter")["Sales Amount"].sum().map(usd))

fig, axes = plt.subplots(3, 1, figsize=(13, 13))
axes[0].plot(mensuel.index, mensuel.values, marker="o", color=PALETTE[0], lw=2)
axes[0].yaxis.set_major_formatter(usd)
axes[0].set_title("Ventes mensuelles")
axes[0].set_ylabel("Ventes ($)")
etiq_trim = [str(p) for p in trimestriel.index.to_period("Q")]
barres = axes[1].bar(etiq_trim, trimestriel.values, color=PALETTE[1])
axes[1].bar_label(barres, labels=[usd(v) for v in trimestriel.values], padding=2, fontsize=8)
axes[1].yaxis.set_major_formatter(usd)
axes[1].set_title("Ventes trimestrielles (trimestres civils)")
axes[1].set_ylabel("Ventes ($)")
barres = axes[2].bar(fiscal.index, fiscal.values, color=PALETTE[2])
axes[2].bar_label(barres, labels=[usd(v) for v in fiscal.values], padding=3)
axes[2].yaxis.set_major_formatter(usd)
axes[2].set_title("Ventes par exercice fiscal (FY2020 : jusqu'au 15 juin 2020)")
axes[2].set_ylabel("Ventes ($)")
sauver(fig, "series_temporelles.png")

# Bougies japonaises : pour chaque mois, ouverture = ventes du 1er jour, clôture = dernier jour,
# plus haut / plus bas = meilleur / pire jour du mois.
ohlc = jour.resample("MS").ohlc()
fig, ax = plt.subplots(figsize=(14, 6))
largeur = 20
for date, (o, h, b, c) in zip(ohlc.index, ohlc[["open", "high", "low", "close"]].to_numpy()):
    x = mdates.date2num(date) + 15
    couleur = "#2a9d8f" if c >= o else "#c73e1d"
    ax.vlines(x, b, h, color=couleur, lw=1.2)
    ax.add_patch(Rectangle((x - largeur / 2, min(o, c)), largeur, max(abs(c - o), 50), facecolor=couleur, edgecolor=couleur))
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.autoscale_view()
ax.yaxis.set_major_formatter(usd)
ax.set_title("Bougies mensuelles des ventes quotidiennes (vert : clôture >= ouverture)")
ax.set_ylabel("Ventes quotidiennes ($)")
fig.autofmt_xdate()
sauver(fig, "bougies_mensuelles.png")

# %% ===========================================================================
# 8. FUSION DES DONNÉES
# ==============================================================================
titre("8. Fusion de toutes les tables")
clients = customers.rename(
    columns={c: f"Customer {c}" for c in ["City", "State-Province", "Country-Region", "Postal Code"]}
).rename(columns={"Customer": "Customer Name"})
revendeurs = resellers.rename(
    columns={c: f"Reseller {c}" for c in ["City", "State-Province", "Country-Region", "Postal Code"]}
).rename(columns={"Reseller": "Reseller Name"})

full = (
    sales.merge(orders, on="SalesOrderLineKey", how="left", validate="1:1")
    .merge(territories, on="SalesTerritoryKey", how="left", validate="m:1")
    .merge(
        products[["ProductKey", "Product", "SKU", "Color", "List Price", "Model", "Subcategory", "Category"]],
        on="ProductKey", how="left", validate="m:1",
    )
    .merge(
        dates[["DateKey", "Fiscal Year", "Fiscal Quarter", "MonthKey"]],
        left_on="OrderDateKey", right_on="DateKey", how="left", validate="m:1",
    )
    .merge(clients, on="CustomerKey", how="left", validate="m:1")
    .merge(revendeurs, on="ResellerKey", how="left", validate="m:1")
)
full["Territory"] = full["Country"].where(full["Country"] == full["Region"], full["Country"] + " - " + full["Region"])

# Vérifications : aucune ligne de vente perdue ni dupliquée, aucun montant altéré
assert len(full) == len(sales), "Des lignes de ventes ont été perdues ou dupliquées !"
assert full["SalesOrderLineKey"].is_unique
assert np.isclose(full["Sales Amount"].sum(), sales["Sales Amount"].sum())
print(f"Lignes dans sales : {len(sales):,} | lignes après fusion : {len(full):,} -> OK")
print(f"Total Sales Amount avant / après fusion : {sales['Sales Amount'].sum():,.2f} / {full['Sales Amount'].sum():,.2f} -> OK")
print("Valeurs manquantes sur les clés de jointure après fusion :")
print(full[["Channel", "Country", "Product", "Fiscal Year", "Customer Name", "Reseller Name"]].isna().sum().to_string())
print(f"Dimensions de la vue complète : {full.shape[0]:,} lignes x {full.shape[1]} colonnes")
print(full.head(3).T)

titre("8. Analyse croisée")
print("--- Ventes par produit (top 15) et par territoire ---")
pt = full.pivot_table(index="Product", columns="Territory", values="Sales Amount", aggfunc="sum", fill_value=0, margins=True, margins_name="TOTAL")
print(pt.drop(index="TOTAL").sort_values("TOTAL", ascending=False).head(15).round(0))

print("\n--- Nombre de produits par modèle (catalogue) - top 15 ---")
nb_par_modele = products.groupby("Model", observed=True)["ProductKey"].nunique().sort_values(ascending=False)
print(nb_par_modele.head(15))
print(f"Nombre de modèles : {len(nb_par_modele)} | produits : {nb_par_modele.sum()}")

ventes_clients = full[full["CustomerKey"] != -1]  # -1 = vente via revendeur (pas de client final)
print("\n--- Clients avec le plus de commandes (top 10) ---")
top_cmd = ventes_clients.groupby(["CustomerKey", "Customer Name"])["Sales Order"].nunique().sort_values(ascending=False).head(10)
print(top_cmd)
print("\n--- Clients ayant le plus dépensé (top 10) ---")
top_depense = ventes_clients.groupby(["CustomerKey", "Customer Name"])["Sales Amount"].sum().sort_values(ascending=False).head(10)
print(top_depense.map(usd))

print("\n--- Nombre de ventes (lignes) par catégorie de produit et par territoire ---")
croise = pd.crosstab(full["Territory"], full["Category"], margins=True, margins_name="TOTAL")
print(croise)
print("\n--- Nombre de commandes distinctes par catégorie et par territoire ---")
print(full.pivot_table(index="Territory", columns="Category", values="Sales Order", aggfunc="nunique", observed=True, fill_value=0))

titre("8. Visualisations globales (produit, territoire, période)")
cat_terr = full.pivot_table(index="Territory", columns="Category", values="Sales Amount", aggfunc="sum", fill_value=0, observed=True)
cat_terr = cat_terr.loc[cat_terr.sum(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(14, 7))
bas = np.zeros(len(cat_terr))
for couleur, cat in zip(PALETTE, cat_terr.columns):
    ax.bar(cat_terr.index, cat_terr[cat], bottom=bas, label=cat, color=couleur)
    bas += cat_terr[cat].to_numpy()
ax.tick_params(axis="x", labelrotation=20)
ax.yaxis.set_major_formatter(usd)
ax.set_title("Ventes par territoire et par catégorie de produit")
ax.set_ylabel("Ventes ($)")
ax.legend(title="Catégorie")
sauver(fig, "global_territoire_categorie.png")

an_cat = full.pivot_table(index="Fiscal Year", columns="Category", values="Sales Amount", aggfunc="sum", observed=True)
fig, ax = plt.subplots(figsize=(11, 6))
an_cat.plot(kind="bar", ax=ax, color=PALETTE[: an_cat.shape[1]], rot=0, width=0.8)
ax.yaxis.set_major_formatter(usd)
ax.set_title("Ventes par exercice fiscal et par catégorie de produit")
ax.set_ylabel("Ventes ($)")
ax.set_xlabel("Exercice fiscal")
sauver(fig, "global_annee_categorie.png")

an_grp = full.pivot_table(index="Fiscal Year", columns="Group", values="Sales Amount", aggfunc="sum")
fig, ax = plt.subplots(figsize=(11, 6))
an_grp.plot(kind="line", marker="o", lw=2.5, ax=ax)
ax.yaxis.set_major_formatter(usd)
ax.set_title("Évolution des ventes par groupe géographique")
ax.set_ylabel("Ventes ($)")
ax.set_xlabel("Exercice fiscal")
sauver(fig, "global_annee_groupe.png")

# %% ===========================================================================
# 9. PERFORMANCE DES REVENDEURS
# ==============================================================================
titre("9. Performance des revendeurs")
rs = full[full["ResellerKey"] != -1]  # on exclut les ventes Internet (pas de revendeur)
print(f"Lignes de vente via revendeurs : {len(rs):,} sur {len(full):,} ({len(rs) / len(full):.1%}) ; "
      f"{rs['ResellerKey'].nunique()} revendeurs actifs")
perf_rev = rs.groupby(["ResellerKey", "Reseller Name", "Business Type"]).agg(
    Ventes=("Sales Amount", "sum"), Marge=("Margin", "sum"), Commandes=("Sales Order", "nunique")
).reset_index()
perf_rev["Taux de marge"] = perf_rev["Marge"] / perf_rev["Ventes"]
top_rev = perf_rev.sort_values("Ventes", ascending=False).head(10)
print("Top 10 revendeurs par ventes :")
print(top_rev.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 7))
lab = [f"{n} ({k})" for n, k in zip(top_rev["Reseller Name"], top_rev["ResellerKey"])]
barres = ax.barh(lab[::-1], top_rev["Ventes"].to_numpy()[::-1], color=PALETTE[0])
ax.bar_label(barres, labels=[usd(v) for v in top_rev["Ventes"].to_numpy()[::-1]], padding=3)
ax.xaxis.set_major_formatter(usd)
ax.set_xlim(0, top_rev["Ventes"].max() * 1.15)
ax.set_title("Top 10 des revendeurs par ventes")
ax.set_xlabel("Ventes ($)")
sauver(fig, "revendeurs_top10_barres.png")

camembert(
    perf_rev.set_index("Reseller Name")["Ventes"].groupby(level=0).sum(),
    "Part des ventes revendeurs : top 7 + autres", "revendeurs_pie_top.png", max_parts=8,
)
camembert(rs.groupby("Business Type")["Sales Amount"].sum(), "Ventes par type de revendeur", "revendeurs_pie_type.png")

titre("9. Comparaison des revendeurs sur différentes périodes (exercices fiscaux)")
periodes = rs.pivot_table(index="Reseller Name", columns="Fiscal Year", values="Sales Amount", aggfunc="sum", fill_value=0)
top_noms = top_rev["Reseller Name"].tolist()
periodes_top = periodes.loc[periodes.index.isin(top_noms)].copy()
periodes_top = periodes_top.loc[periodes_top.sum(axis=1).sort_values(ascending=False).index]
croissance = periodes_top.pct_change(axis=1).replace([np.inf, -np.inf], np.nan)
print("Ventes par exercice (top 10 revendeurs) :")
print(periodes_top.round(0))
print("\nCroissance d'un exercice au suivant :")
print((croissance * 100).round(1).astype(str) + " %")

fig, ax = plt.subplots(figsize=(14, 7))
periodes_top.plot(kind="bar", ax=ax, color=PALETTE[: periodes_top.shape[1]], width=0.8, rot=30)
ax.yaxis.set_major_formatter(usd)
ax.set_title("Ventes des 10 meilleurs revendeurs par exercice fiscal")
ax.set_ylabel("Ventes ($)")
ax.set_xlabel("Revendeur")
ax.legend(title="Exercice")
sauver(fig, "revendeurs_periodes.png")

titre("FIN DU TP - tous les graphiques sont dans le dossier : " + str(OUT_DIR.resolve()))