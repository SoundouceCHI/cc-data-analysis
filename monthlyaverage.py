import matplotlib.pyplot as plt

# Monthly average daily sales data based on our previous calculation
months = ['Novembre', 'Février', 'Mai', 'Septembre', 'Août', 'Juin', 'Avril', 'Décembre', 'Juillet', 'Mars', 'Janvier', 'Octobre']
avg_daily_sales = [150310, 127481, 125652, 121285, 121181, 104047, 88087, 86796, 84689, 77668, 68931, 67324]

plt.figure(figsize=(10, 5))
colors = ['#E11D48' if m in ['Novembre', 'Février', 'Mai'] else '#1B559B' for m in months]
bars = plt.bar(months, [s/1000 for s in avg_daily_sales], color=colors)

plt.title('Moyenne des ventes quotidiennes par mois ($k / jour)', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Ventes moyennes ($k / jour)', fontsize=11)
plt.xlabel('Mois', fontsize=11)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add data labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{int(yval)}k$', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('moyenne_ventes_journalieres_mois.png', dpi=300)
plt.close()
print("Chart created successfully!")