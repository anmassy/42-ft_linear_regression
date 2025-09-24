import matplotlib.pyplot as plt

def load_data():
    kilometrages = []
    prix = []

    with open("data/data.csv", 'r') as file:
        next(file)
        for line in file:
            km, p = line.strip().split(",")
            kilometrages.append(float(km) / 1000)
            prix.append(float(p))

    return kilometrages, prix

def estimate_price(km, theta0, theta1):
    return theta0 + theta1 * km

def train_model():
    kilometrages, prix = load_data()
    
    km_mean = sum(kilometrages) / len(kilometrages)
    km_std = (sum((x - km_mean)**2 for x in kilometrages) / len(kilometrages))**0.5
    kilometrages_norm = [(km - km_mean) / km_std for km in kilometrages]
    
    prix_mean = sum(prix) / len(prix)
    prix_std = (sum((p - prix_mean)**2 for p in prix) / len(prix))**0.5
    prix_norm = [(p - prix_mean) / prix_std for p in prix]
    
    print("Données chargées:", len(kilometrages), "voitures")
    print("KM min:", min(kilometrages), "max:", max(kilometrages))
    print("Prix min:", min(prix), "max:", max(prix))
    print("\nDonnées normalisées:")
    print("KM norm min:", min(kilometrages_norm), "max:", max(kilometrages_norm))
    print("Prix norm min:", min(prix_norm), "max:", max(prix_norm))
    
    theta0, theta1 = 0.0, 0.0
    learning_rate = 0.01
    epochs = 2000
    m = len(kilometrages_norm)
    
    for epoch in range(epochs):
        sum_error = 0
        sum_error_km = 0
        
        for i in range(m):
            prediction = estimate_price(kilometrages_norm[i], theta0, theta1)
            error = prediction - prix_norm[i]
            sum_error += error
            sum_error_km += error * kilometrages_norm[i]
        
        gradient_theta0 = sum_error / m
        gradient_theta1 = sum_error_km / m
        
        theta0 = theta0 - learning_rate * gradient_theta0
        theta1 = theta1 - learning_rate * gradient_theta1
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: theta0={theta0:.4f}, theta1={theta1:.4f}")
    
    theta1_real = theta1 * prix_std / km_std
    theta0_real = prix_mean - theta1_real * km_mean
    
    return theta0_real, theta1_real, kilometrages, prix

def create_graphics(km_original, prix_original, theta0, theta1):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(km_original, prix_original, alpha=0.7, label="Données réelles")
    
    km_line = [min(km_original), max(km_original)]
    prix_line = [estimate_price(km, theta0, theta1) for km in km_line]
    plt.plot(km_line, prix_line, 'r-', label=f"Régression: prix = {theta0:.2f} + {theta1:.2f} * km")
    
    plt.xlabel("Kilométrage (en milliers)")
    plt.ylabel("Prix (€)")
    plt.title("Régression Linéaire - Kilométrage vs Prix")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("img/regression_graphique.png", dpi=300, bbox_inches='tight')
    print("Graphique sauvegardé : regression_graphique.png")

def save_theta(theta0, theta1):
    with open("data/theta.csv", 'w') as file:
        file.write(f"{theta0},{theta1}\n")
    print(f"\nModèle entraîné sauvegardé !")
    print(f"θ0 = {theta0:.6f}")
    print(f"θ1 = {theta1:.6f}")
    print("Paramètres sauvegardés dans theta.csv")

def main():
    print("=== ENTRAÎNEMENT DU MODÈLE ===\n")
    
    # Entraînement
    theta0, theta1, km_original, prix_original = train_model()
    
    # Sauvegarde
    save_theta(theta0, theta1)
    
    # Graphique
    create_graphics(km_original, prix_original, theta0, theta1)
    
    # # Tests de prédiction
    # print("\n=== TESTS DE PRÉDICTION ===")
    # test_km = [50, 100, 150, 200]  # En milliers de km
    # for km in test_km:
    #     prix_estime = estimate_price(km, theta0, theta1)
    #     print(f"{km*1000:,} km → {prix_estime:.2f}€")
    
    print("\nEntraînement terminé ! Vous pouvez maintenant utiliser predict.py")

if __name__ == "__main__":
    main()