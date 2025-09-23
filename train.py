def load_data():
    kilometrages = []
    prix = []

    with open("data.csv", 'r') as file:
        next(file)
        for line in file:
            km, p = line.strip().split(",")
            kilometrages.append(float(km) / 1000)
            prix.append(float(p))

    # print("Kilometrages:", kilometrages)
    # print("Prix:", prix)
    return kilometrages, prix

def estimate_price(km, theta0, theta1):
    return theta0 + theta1 * km

def train_model():
    kilometrages, prix = load_data()
    theta0, theta1 = 0.0, 0.0
    learning_rate = 0.000001
    epochs = 5000
    m = len(kilometrages)

    for epoch in range(epochs):
        # Calcul des gradients
        gradient0 = sum(estimate_price(kilometrages[i], theta0, theta1) - prix[i] for i in range(m)) / m
        gradient1 = sum((estimate_price(kilometrages[i], theta0, theta1) - prix[i]) * kilometrages[i] for i in range(m)) / m

        # Mise à jour des paramètres
        # print(gradient0, gradient1)
        theta0 -= learning_rate * gradient0
        theta1 -= learning_rate * gradient1
        print(theta0, theta1)
    return theta0, theta1

def save_theta(theta0, theta1):
    with open("theta.csv", 'w') as file:
        file.write(f"{theta0},{theta1}\n")
    print(f"Modèle entraîné: θ0 = {theta0}, θ1 = {theta1}")
    return

def main():
    theta0, theta1 = train_model()
    save_theta(theta0, theta1)
    return

if __name__ == "__main__":
    main()