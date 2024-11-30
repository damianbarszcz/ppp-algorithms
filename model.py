from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd
import pickle

# Funkcja do wczytania danych o ocenach użytkowników dla mentorów
def load_data():
    # Przykładowe dane, które mogą pochodzić z bazy danych
    data = {
        'user_id': [1, 2, 3, 4, 5],
        'mentor_id': [101, 102, 103, 101, 104],
        'rating': [5, 4, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df[['user_id', 'mentor_id', 'rating']], reader)
    return dataset

# Funkcja do trenowania modelu Collaborative Filtering
def train_model():
    dataset = load_data()
    trainset, testset = train_test_split(dataset, test_size=0.2)

    # Wybór algorytmu (tu SVD)
    model = SVD()
    model.fit(trainset)

    # Testowanie modelu na zestawie testowym
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"RMSE: {rmse}")

    # Zapisz model, jeśli chcesz go później wykorzystać
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

# Funkcja do ładowania zapisanego modelu (jeśli już jest wytrenowany)
def load_trained_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Funkcja do generowania rekomendacji dla danego użytkownika
def get_recommendations(user_id, model):
    dataset = load_data()
    all_mentor_ids = set(dataset.df['mentor_id'])
    rated_mentor_ids = set(dataset.df[dataset.df['user_id'] == user_id]['mentor_id'])
    non_rated_mentor_ids = all_mentor_ids - rated_mentor_ids

    recommendations = []
    for mentor_id in non_rated_mentor_ids:
        pred = model.predict(user_id, mentor_id)
        recommendations.append((mentor_id, pred.est))  # Przewidywana ocena

    recommendations.sort(key=lambda x: x[1], reverse=True)  # Sortowanie rekomendacji po ocenie
    return recommendations

# Funkcja do generowania rekomendacji na podstawie ocenionych mentorów
def generate_user_recommendations(user_id):
    # Jeśli model jest już zapisany, załaduj go, w przeciwnym razie trenuj go na nowo
    try:
        model = load_trained_model()
    except FileNotFoundError:
        model = train_model()

    # Generowanie rekomendacji dla użytkownika
    recommendations = get_recommendations(user_id, model)

    # Zwróć 3 najlepsze rekomendacje
    return recommendations[:3]

# Przykład użycia
if __name__ == '__main__':
    user_id = 1
    recommendations = generate_user_recommendations(user_id)
    print(f"Recommendations for user {user_id}: {recommendations}")