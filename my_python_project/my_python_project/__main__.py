class BankAccount:
    def __init__(self, account_holder_name: str, balance: float = 0.0) -> None:
        self.account_holder_name = account_holder_name
        self.balance = balance

    def deposit(self, amount: float) -> None:
        if amount > 0:
            self.balance += amount

    def withdraw(self, amount: float) -> bool:
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False

    def get_balance(self) -> float:
        return self.balance

    def show_balance(self) -> None:
        print(f"Account holder: {self.account_holder_name}, Balance: {self.get_balance()}")


account = BankAccount("John Doe", 100.0)
account.show_balance()
account.deposit(50.0)
account.show_balance()
account.withdraw(30.0)
account.show_balance()

class User:
    def __init__(self, name, favorite_genre):
        self.name = name
        self.favorite_genre = favorite_genre
        self.movies_watched = 0
        self.watched_movies = []
    def watch_movie(self, movie_title: str):
        self.movies_watched += 1
        self.watched_movies.append(movie_title)

    def get_watched_movies(self):
        return self.watched_movies
    
    def recommend_movie(self):
        if self.favorite_genre == "Action":
            return "The Dark Knight"
        elif self.favorite_genre == "Comedy":
            return "The Grand Budapest Hotel"
        elif self.favorite_genre == "Sci-Fi":
            return "Interstellar"
        else:
            return "Finding Nemo"

    def display_profile(self):
        print(f"Name: {self.name}")
        print(f"Favorite Genre: {self.favorite_genre}")
        print(f"Movies Watched: {self.movies_watched}")
        print(f"Recommended Movie: {self.recommend_movie()}")
        print(f"Watched Movies Count: {self.movies_watched}")
        print("Watched Movies:")
        for movie in self.get_watched_movies():
            print(f"Watched Movie: {movie}")
    

class recomendation_system:
    def __init__(self, users: list[User]):  
        self.users = users

    def add_user(self, user: User) -> None:
        self.users.append(user)

    def remove_user(self, user: User) -> bool:
        if user in self.users:
            self.users.remove(user)
            return True
        return False

    def display_all_users(self) -> None:
        for user in self.users:
            user.display_profile()
            print()  # Add a blank line between user profiles for better readability

    def calculate_average_movies_watched(self) -> float:
        if not self.users:
            return 0.0
        total_movies = sum(user.movies_watched for user in self.users)
        return total_movies / len(self.users)

    def recommended_movies_for_all_users(self) -> dict[str, str]:
        recommendations = {}
        for user in self.users:
            recommendations[user.name] = user.recommend_movie()
        return recommendations
    
    def show_statistics(self) -> None:
        print(f"Average Movies Watched: {self.calculate_average_movies_watched()}")
        print("Recommended Movies for All Users:")
        recommendations = self.recommended_movies_for_all_users()
        for user_name, movie in recommendations.items():
            print(f"{user_name}: {movie}")
        for user in self.users:
            user.display_profile()
            print()  # Add a blank line between user profiles for better readability

def main() -> None:
   users = [
    User(name="Alex", favorite_genre="Action"),
    User(name="Sam", favorite_genre="Comedy"),
    User(name="Priya", favorite_genre="Sci-Fi"),
    User(name="Maya", favorite_genre="Animation")
]
   print("hello world")
   system = recomendation_system(users)
   system.show_statistics()


if __name__ == "__main__":
    main()
