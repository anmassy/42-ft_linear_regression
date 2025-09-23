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

# Normalisation des données pour améliorer la convergence
# Normalisation des kilométrages : (x - moyenne) / écart-type
km_mean = sum(kilometrages) / len(kilometrages)
km_std = (sum((x - km_mean)**2 for x in kilometrages) / len(kilometrages))**0.5
kilometrages_norm = [(km - km_mean) / km_std for km in kilometrages]

# Normalisation des prix : (y - moyenne) / écart-type  
prix_mean = sum(prix) / len(prix)
prix_std = (sum((y - prix_mean)**2 for y in prix) / len(prix))**0.5
prix_norm = [(p - prix_mean) / prix_std for p in prix]

print("Kilométrages normalisés (5 premiers):", kilometrages_norm[:5])
print("Prix normalisés (5 premiers):", prix_norm[:5])
print(f"Statistiques km - Moyenne: {km_mean:.2f}, Écart-type: {km_std:.2f}")
print(f"Statistiques prix - Moyenne: {prix_mean:.2f}, Écart-type: {prix_std:.2f}")

# ---voir les donner sur un graphique---
import matplotlib.pyplot as plt  # Importe la bibliothèque pour faire des graphiques

# ---fonction de prédiction---
def estimate_price(x, theta0, theta1):
	"""Calcule la prédiction du prix pour un kilométrage x."""
	# Formule de la droite : y = theta0 + theta1 * x
	# theta0 = ordonnée à l'origine, theta1 = pente de la droite
	return theta0 + theta1 * x

# parametres initiaux du graphique
plt.figure(figsize=(10, 6))
plt.scatter(kilometrages, prix, alpha=0.7)
plt.xlabel("Kilométrage")
plt.ylabel("Prix")
plt.title("Données voiture - Kilométrage vs Prix")

# --- Descente de gradient ---
# Initialisation des paramètres
theta0 = 0
theta1 = 0
learning_rate = 0.1  # À ajuster selon l’échelle de tes données
epochs = 1000  # Nombre d’itérations
m = len(kilometrages)

print("Début de l'entraînement...")
for epoch in range(epochs):
	sum_error0 = 0
	sum_error1 = 0
	for x, y in zip(kilometrages_norm, prix_norm):
		prediction = estimate_price(x, theta0, theta1)
		error = prediction - y
		sum_error0 += error
		sum_error1 += error * x
	
	# Mise à jour des paramètres
	theta0 -= learning_rate * (sum_error0 / m)
	theta1 -= learning_rate * (sum_error1 / m)
	
	# Affichage du progrès toutes les 200 itérations
	if epoch % 200 == 0:
		# Calcul du coût (MSE)
		total_cost = sum((estimate_price(x, theta0, theta1) - y)**2 for x, y in zip(kilometrages_norm, prix_norm)) / (2 * m)
		print(f"Époque {epoch}: Coût = {total_cost:.6f}, θ0 = {theta0:.6f}, θ1 = {theta1:.6f}")

print(f"\nTheta0 final (normalisé) : {theta0}")
print(f"Theta1 final (normalisé) : {theta1}")

# Conve sion vers l'espace original
# Pour dénormaliser : y_original = y_norm * prix_std + prix_mean
# Pour x : x_original = x_norm * km_std + km_mean
# Donc : prix = theta0_norm * prix_std + prix_mean + theta1_norm * prix_std/km_std * (km - km_mean)
theta0_original = theta0 * prix_std + prix_mean - theta1 * prix_std * km_mean / km_std
theta1_original = theta1 * prix_std / km_std

print(f"\nTheta0 original : {theta0_original}")
print(f"Theta1 original : {theta1_original}")

# Affichage de la droite ajustée sur les données originales
x_min = min(kilometrages)
x_max = max(kilometrages)
x_vals = [x_min, x_max]
# Utilisation des paramètres originaux pour l'affichage
y_vals_fit = [theta0_original + theta1_original * x for x in x_vals]
plt.plot(x_vals, y_vals_fit, color='red', label='Régression linéaire', linewidth=2)
plt.legend()

# Ajout d'informations sur le graphique
plt.text(0.05, 0.95, f'θ₀ = {theta0_original:.2f}\nθ₁ = {theta1_original:.6f}', 
         transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graphique.png", dpi=150)
# plt.show() #afficher le graphique sur une fenetre independante

print(f"\nÉquation de la droite : Prix = {theta0_original:.2f} + {theta1_original:.6f} * Kilométrage")