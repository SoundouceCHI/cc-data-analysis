import pandas as pd

fichier = "CC-AIBD-DataAnalyse-donnees.xlsx"
ventes   = pd.read_excel(fichier, sheet_name="Sales_data")
commande = pd.read_excel(fichier, sheet_name="Sales Order_data")
produits = pd.read_excel(fichier, sheet_name="Product_data")

# Fusion : on ajoute le canal (Internet / Reseller) et la catégorie
df = (ventes
      .merge(commande[["SalesOrderLineKey", "Channel"]], on="SalesOrderLineKey")
      .merge(produits[["ProductKey", "Category"]], on="ProductKey"))

df["Margin"] = df["Sales Amount"] - df["Total Product Cost"]

# 1) Barres : part des ventes et part de la marge par canal (73/27 et 4/96)
canal = df.groupby("Channel")[["Sales Amount", "Margin"]].sum()
print((canal / canal.sum() * 100).round(1))

# 2) Taux de marge par canal (0,6 % revendeurs, 41 % Internet)
print((canal["Margin"] / canal["Sales Amount"] * 100).round(1))

# 3) Tableau : taux de marge par catégorie et par canal
cat = df.groupby(["Category", "Channel"])[["Sales Amount", "Margin"]].sum()
print((cat["Margin"] / cat["Sales Amount"] * 100).unstack().round(1))

# 4) Lignes vendues sous le coût standard (36 %, 21 821 lignes, -3,3 M$)
revendeurs = df[df["Channel"] == "Reseller"]
sous_cout = revendeurs[revendeurs["Unit Price"] < revendeurs["Product Standard Cost"]]
print(len(sous_cout), len(revendeurs), round(len(sous_cout) / len(revendeurs) * 100, 1))
print(round(sous_cout["Margin"].sum() / 1e6, 2), "M$")