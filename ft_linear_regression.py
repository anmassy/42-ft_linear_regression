# ---Stockage des données---
kilometrages = []  # Crée une liste vide pour stocker les kilométrages
prix = []          # Crée une liste vide pour stocker les prix

# Ouvre le fichier data.csv en mode lecture ('r' = read)
with open("data.csv", 'r') as file:
	next(file)  # Ignore la première ligne (en-têtes du CSV)
	# Parcourt chaque ligne restante du fichier
	for value in file:
		# .strip() enlève les espaces/retours à la ligne, .split(",") sépare par la virgule
		km, p = value.strip().split(",")
		# Convertit le texte en nombre décimal et l'ajoute à la liste
		kilometrages.append(float(km))
		prix.append(float(p))

# Affiche les données chargées pour vérification
print("    ---kilometrages---    \n", kilometrages, "\n")
print("    ---prix---    \n", prix)

# Normalisation des kilométrages pour éviter les problèmes numériques
# Divise chaque kilométrage par 1000 pour avoir des nombres plus petits
kilometrages = [km/1000 for km in kilometrages]  # List comprehension
print("Kilométrages normalisés (en milliers):", kilometrages[:5])  # Affiche les 5 premiers

# ---voir les donner sur un graphique---
import matplotlib.pyplot as plt  # Importe la bibliothèque pour faire des graphiques

# ---fonction de prédiction---
def estimate_price(x, theta0, theta1):
	"""Calcule la prédiction du prix pour un kilométrage x."""
	# Formule de la droite : y = theta0 + theta1 * x
	# theta0 = ordonnée à l'origine, theta1 = pente de la droite
	return theta0 + theta1 * x

# parametres initiaux du graphique
plt.scatter(kilometrages, prix)
plt.xlabel("Kilométrage")
plt.ylabel("Prix")
plt.title("Données voiture")

# --- Descente de gradient ---
# Initialisation des paramètres
theta0 = 0
theta1 = 0
learning_rate = 0.01  # À ajuster selon l’échelle de tes données
epochs = 5000  # Nombre d’itérations
m = len(kilometrages)

for epoch in range(epochs):
	sum_error0 = 0
	sum_error1 = 0
	for x, y in zip(kilometrages, prix):
		prediction = estimate_price(x, theta0, theta1)
		error = prediction - y
		sum_error0 += error
		sum_error1 += error * x
	theta0 -= learning_rate * (sum_error0 / m)
	theta1 -= learning_rate * (sum_error1 / m)

print(f"Theta0 final : {theta0}")
print(f"Theta1 final : {theta1}")

# Affichage de la droite ajustée
x_min = min(kilometrages)
x_max = max(kilometrages)
x_vals = [x_min, x_max]
y_vals_fit = [estimate_price(x, theta0, theta1) for x in x_vals]
plt.plot(x_vals, y_vals_fit, color='red', label='Ajustement')
plt.legend()

plt.savefig("graphique.png")