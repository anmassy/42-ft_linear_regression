def load_theta():
    try:
        with open("data/theta.csv", 'r') as file:
            line = file.readline().strip()
            theta0, theta1 = map(float, line.split(','))
            return theta0, theta1
    except FileNotFoundError:
        print("Fichier theta.csv non trouvé. Veuillez d'abord exécuter train.py.")
    except Exception:
        print("Erreur lors du chargement des paramètres: theta0 et theta1.")
    return 0, 0

def estimate_price(km, theta0, theta1):
	return (theta0 + theta1 * km)

def main ():
    theta0, theta1 = load_theta()

    if theta0 == 0 and theta1 == 0:
        print("Le modèle n'est pas encore entraîné.")
        print("Lancez d'abord train.py pour entraîner le modèle.")
        return

    try:
        mileage = float(input("Entrez le kilométrage de la voiture : "))
        if mileage < 0:
            print("Le kilométrage ne peut pas être négatif.")
            return
        
        km = mileage / 1000
        prix = estimate_price(km, theta0, theta1)
        print(f"Prix estimé : {prix:.2f}€")
        
    except ValueError:
        print("Erreur : Veuillez entrer un nombre valide.")
    
    return

if __name__ == "__main__":
    main()