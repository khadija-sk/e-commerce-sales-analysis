import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)
n_transactions = 10000

products = {
    'Electronique': [
        ('Ecouteurs Bluetooth', 49.99, 25.00), ('Chargeur USB-C', 19.99, 8.00),
        ('Souris Gaming', 39.99, 18.00), ('Clavier Mecanique', 89.99, 45.00),
        ('Webcam HD', 59.99, 30.00), ('Cable HDMI', 14.99, 5.00),
        ('Support Telephone', 12.99, 4.00), ('Batterie Externe', 34.99, 15.00),
    ],
    'Vetements': [
        ('T-Shirt Coton', 24.99, 10.00), ('Jean Slim', 59.99, 28.00),
        ('Sweat a Capuche', 44.99, 20.00), ('Chaussettes Pack3', 14.99, 5.00),
        ('Casquette', 19.99, 8.00), ('Echarpe Hiver', 29.99, 12.00),
        ('Gants', 17.99, 7.00), ('Ceinture Cuir', 34.99, 15.00),
    ],
    'Maison': [
        ('Coussin Deco', 22.99, 9.00), ('Tasse Ceramique', 12.99, 5.00),
        ('Bougie Parfumee', 18.99, 7.00), ('Cadre Photo', 16.99, 6.00),
        ('Plaid Polaire', 29.99, 13.00), ('Miroir Mural', 39.99, 18.00),
        ('Vase Moderne', 24.99, 10.00), ('Lampe LED', 34.99, 16.00),
    ],
    'Sport': [
        ('Gourde Inox', 19.99, 8.00), ('Tapis Yoga', 34.99, 16.00),
        ('Bandes Resistance', 24.99, 10.00), ('Halteres 2kg', 29.99, 12.00),
        ('Corde a Sauter', 14.99, 5.00), ('Serviette Sport', 17.99, 7.00),
        ('Sac Sport', 39.99, 18.00), ('Gants Fitness', 22.99, 9.00),
    ],
    'Beaute': [
        ('Creme Hydratante', 27.99, 12.00), ('Shampoing Bio', 16.99, 7.00),
        ('Masque Visage', 21.99, 9.00), ('Huile Essentielle', 19.99, 8.00),
        ('Brosse Cheveux', 14.99, 6.00), ('Serum Anti-Age', 44.99, 20.00),
        ('Savon Artisanal', 9.99, 3.50), ('Baume a Levres', 7.99, 2.50),
    ]
}

countries = {'France': 0.45, 'Belgique': 0.15, 'Suisse': 0.12, 'Canada': 0.10,
             'Allemagne': 0.08, 'Espagne': 0.05, 'Italie': 0.05}
cities = {
    'France': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes', 'Bordeaux', 'Lille'],
    'Belgique': ['Bruxelles', 'Anvers', 'Gand', 'Liege'],
    'Suisse': ['Geneve', 'Zurich', 'Berne', 'Lausanne'],
    'Canada': ['Montreal', 'Quebec', 'Ottawa', 'Toronto'],
    'Allemagne': ['Berlin', 'Munich', 'Francfort', 'Hambourg'],
    'Espagne': ['Madrid', 'Barcelone', 'Valence', 'Seville'],
    'Italie': ['Rome', 'Milan', 'Naples', 'Turin']
}

data = []
start_date = datetime(2023, 1, 1)

for i in range(n_transactions):
    day_offset = np.random.randint(0, 365)
    if np.random.random() < 0.2:
        day_offset = np.random.randint(335, 365)
    transaction_date = start_date + timedelta(days=day_offset)
    hour_weights = [2,2,1,1,2,3,5,8,6,4,5,8,10,9,6,5,6,8,10,11,10,7,4,3]
    hour = np.random.choice(range(24), p=np.array(hour_weights)/sum(hour_weights))
    transaction_date = transaction_date.replace(hour=hour, minute=np.random.randint(0,60))

    if np.random.random() < 0.3:
        customer_id = f'CUST_{np.random.randint(1, 2000):05d}'
    else:
        customer_id = f'CUST_{np.random.randint(2000, 10000):05d}'

    country = np.random.choice(list(countries.keys()), p=list(countries.values()))
    city = np.random.choice(cities[country])
    category = np.random.choice(list(products.keys()))
    product_name, price, cost = products[category][np.random.randint(0, len(products[category]))]
    quantity = np.random.choice([1,2,3,4,5], p=[0.5,0.3,0.12,0.05,0.03])
    discount = np.random.choice([0,5,10,15,20], p=[0.8,0.05,0.05,0.05,0.05])
    base_pub = {'Electronique': 300, 'Vetements': 250, 'Maison': 200, 'Sport': 220, 'Beaute': 280}
    pub_budget = base_pub[category] + np.random.randint(-100, 150)
    if transaction_date.month == 12:
        pub_budget *= 1.5
    price_after_discount = price * (1 - discount/100)
    total_price = round(price_after_discount * quantity, 2)
    total_cost = round(cost * quantity, 2)
    profit = round(total_price - total_cost - pub_budget, 2)

    data.append({
        'Transaction_ID': f'TXN_{i+1:06d}', 'Date': transaction_date.strftime('%Y-%m-%d'),
        'Heure': transaction_date.strftime('%H:%M'), 'Customer_ID': customer_id,
        'Categorie': category, 'Produit': product_name, 'Quantite': quantity,
        'Prix_Unitaire': round(price, 2), 'Cout_Unitaire': round(cost, 2),
        'Reduction_%': discount, 'Prix_Final': total_price, 'Cout_Total': total_cost,
        'Budget_Pub': round(pub_budget, 2), 'Profit': profit, 'Pays': country, 'Ville': city
    })

df = pd.DataFrame(data).sort_values('Date').reset_index(drop=True)

# Sauvegarde dans le dossier data\ du projet
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"[OK] dataset.csv genere avec {len(df)} lignes")
print(f"[OK] Fichier sauvegarde dans : {os.path.abspath(output_path)}")