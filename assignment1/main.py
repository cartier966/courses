

RECOMMENDATIONS = {
    "Action": "The Dark Knight",
    "Comedy": "Superbad",
    "Drama": "The Shawshank Redemption",
    "Sci-Fi": "Interstellar",
    "Animation": "Spirited Away",
    "Horror": "Get Out",
}


def header(title):
    print("")
    print("")
    print(f"{'=' * 60}")
    print("")
    print(f"{title}")
    print("")
    print(f"{'=' * 60}")
    print("")
        

# ---------------------------------------------------------------- Part 1

users = [
    ("Alex", "Action", 25, "Mad Max: Fury Road", True),
    ("Bella", "Comedy", 8, "Bridesmaids", False),
    ("Chris", "Sci-Fi", 42, "Blade Runner 2049", True),
    ("Dana", "Action", 12, "John Wick", True),
    ("Eli", "Drama", 5, "Whiplash", False),
    ("Fay", "Animation", 17, "Coco", True),
    ("Gus", "Sci-Fi", 3, "The Matrix", False),
]

header("PART 1: USER DATABASE")
for name, genre, watched, movie, subscribed in users:
    print(
        f"{name} | Genre: {genre} | Watched: {watched} | Favorite: {movie} | Subscribed: {subscribed}"
    )
print("")
print(f"Total users: {len(users)}")

# get active subscribers
active_subscribers = []
for u in users:

    # check if the user is an active subscriber
    if u[4]:
        active_subscribers.append(u[0])

print("Active subscribers:", ", ".join(active_subscribers))

# get users who watched more than 10 movies
more_than_10 = []
for u in users:

    # check if the user has watched more than 10 movies
    if u[2] > 10:
        more_than_10.append(u[0])

print("Watched more than 10 movies:", ", ".join(more_than_10))

# ---------------------------------------------------------------- Part 2
profiles = []
for name, genre, watched, movie, subscribed in users:
    profiles.append(
        {
            "name": name,
            "favorite_genre": genre,
            "favorite_movie": movie,
            "movies_watched": watched,
            "subscription_status": subscribed,
        }
    )


header("PART 2: USER PROFILES")
for p in profiles:
    status = "Active" if p["subscription_status"] else "Inactive"
    print(
        f"{p['name']}: loves {p['favorite_genre']} (favorite: {p['favorite_movie']}), {p['movies_watched']} movies watched, subscription {status}"
    )

# get users by genre
users_by_genre = {}
for p in profiles:
    users_by_genre.setdefault(p["favorite_genre"], []).append(p["name"])

# get the number of genres
genre_counts = {}
for genre, names in users_by_genre.items():
    genre_counts[genre] = len(names)

# get the most popular genre(s)
top_count = max(genre_counts.values())
top_genres = []
for genre, count in genre_counts.items():

    # check if this genre has the top count of users
    if count == top_count:
        top_genres.append(genre)

print("")
print("Users sharing a favorite genre:")

for genre, names in users_by_genre.items():

    # only print genres that have more than one user sharing them
    if len(names) > 1:
        print(f"  {genre}: {', '.join(names)}")

print("")
print("Users per genre:")
for genre, count in genre_counts.items():
    print(f"  {genre} {count}")

print("")
print(f"Most popular genre: {' & '.join(top_genres)} ({top_count} users each)")

print("")
print("Product manager report:")
print(f"  - {len(profiles)} users across {len(genre_counts)} genres.")
print(f"  - Most popular genre(s): {', '.join(top_genres)}.")
print("  - Single-user genres: " + ", ".join(g for g, c in genre_counts.items() if c == 1))

# ---------------------------------------------------------------- Part 3
header("PART 3: ENGAGEMENT ANALYSIS")
total_watched = 0
subscribers = 0
most_active_profile = profiles[0] # default startup value
least_active_profile = profiles[0] # default startup value

for p in profiles:

    # increment thhe total_watched counter with the number of movies this user has watched
    total_watched += p["movies_watched"]

    # check if the user is an active subscriber
    if p["subscription_status"]:

        # increment the subscribers counter for active subscribers
        subscribers += 1

    # check if this user has watched more movies than the current most active profile
    if p["movies_watched"] > most_active_profile["movies_watched"]:

        # update the most active profile
        most_active_profile = p
    if p["movies_watched"] < least_active_profile["movies_watched"]:

        # update the least active profile
        least_active_profile = p

print(f"Total movies watched:      {total_watched}")
print(f"Average per user:          {total_watched / len(profiles)}")
print(f"Most active user:          {most_active_profile['name']} ({most_active_profile['movies_watched']})")
print(f"Least active user:         {least_active_profile['name']} ({least_active_profile['movies_watched']})")
print(f"Active subscriptions:      {subscribers / len(profiles) * 100:.1f}%")

print("")
print("Leaderboard (most to least active):")

temporary_profiles_bag = profiles.copy()
rank = 1

while temporary_profiles_bag: # only if there are items left in the collection

    # set the first element for comparaison purposes in case it is the highest initially
    best = temporary_profiles_bag[0] 

    # iterate through the remaining profiles to find the most active one
    for p in temporary_profiles_bag: 

        #check for a possible new best profile
        if p["movies_watched"] > best["movies_watched"]:

            # if this profile has watched more movies than the current best, update best
            best = p

    print(f"  {rank}. {best['name']} {best['movies_watched']} movies")

    # remove the best profile from the temporary bag and increment the rank
    temporary_profiles_bag.remove(best)
    rank += 1


# ---------------------------------------------------------------- Part 4
def calculate_average(counts):
    # average of a list of movie counts (0 for an empty list)
    if not counts:
        return 0
    else:
        return sum(counts) / len(counts)


def recommend_for_genre(genre):
    # return a recommended movie for a genre
    # defaults to "Inception" if the genre is not found
    return RECOMMENDATIONS.get(genre, "Inception")


def personalized_message(profile):
    # print a personalized recommendation for a profile dictionary
    movie = recommend_for_genre(profile["favorite_genre"])
    print(
        f"Hi {profile['name']}! Since you enjoy {profile['favorite_genre']} movies, we recommend {movie}."
    )


header("PART 4: FUNCTIONS")
print("Average tests:")
print("  all users:       ", calculate_average([p["movies_watched"] for p in profiles]))
print("  subscribers only:", calculate_average([p["movies_watched"] for p in profiles if p["subscription_status"]]))
print("  empty list:      ", calculate_average([]))

print("")
print("Genre recommender tests:")
for genre in RECOMMENDATIONS.keys():
    print(f"  {genre} -> {recommend_for_genre(genre)}")

print("")
print("Personalized messages:")
for p in profiles:
    personalized_message(p)


# ---------------------------------------------------------------- Part 5


header("PART 5: OOP")

class User:
    def __init__(self, name, favorite_genre, favorite_movie, movies_watched, subscription_status, watch_history=[]):
        self.name = name
        self.favorite_genre = favorite_genre
        self.favorite_movie = favorite_movie
        self.movies_watched = movies_watched
        self.subscription_status = subscription_status
        self.watch_history = watch_history

    def add_movie_to_history(self, movie):
            self.movies_watched += 1
            self.watch_history.append(movie)
    
    def watch_movie(self):
            self.movies_watched += 1
    
    def change_genre(self, new_genre):
        self.favorite_genre = new_genre

    def display_profile(self):
        status = ""
        if self.subscription_status:
            status = "Active"
        else:
            status = "Inactive"
        print(f"  Name:            {self.name}")
        print(f"  Favorite genre:  {self.favorite_genre}")
        print(f"  Favorite movie:  {self.favorite_movie}")
        print(f"  Movies watched:  {self.movies_watched}")
        print(f"  Watch History:   {', '.join(self.watch_history)}")
        print(f"  Subscription:    {status}")

    def recommend_movie(self):
        return recommend_for_genre(self.favorite_genre)



user_objects = []
for profile in profiles:
    user_objects.append(User(
        profile["name"],
        profile["favorite_genre"],
        profile["favorite_movie"],
        profile["movies_watched"],
        profile["subscription_status"]
    ))

# set some mock watch history and watch count for testing
user_objects[0].watch_history = ["Inception", "The Matrix"]
user_objects[0].movies_watched = 2

user_objects[3].watch_history = ["The Godfather", "Pulp Fiction", "Inception"]
user_objects[3].movies_watched = 3


for user in user_objects:
    print("")
    print(f"--- {user.name} (initial) ---")
    user.display_profile()
    print(f"  Recommendation:  {user.recommend_movie()}")

    user.watch_movie()
    print("")
    print(f"  After watch_movie(): {user.movies_watched} movies watched")

    new_genre = "Drama" if user.favorite_genre != "Drama" else "Comedy"
    user.change_genre(new_genre)
    print(f"  After change_genre('{new_genre}'):")
    user.display_profile()
    print(f"  New recommendation: {user.recommend_movie()}")


class RecommendationSystem:

    __recommendations = {
        "Action": "The Dark Knight",
        "Comedy": "Superbad",
        "Drama": "The Shawshank Redemption",
        "Sci-Fi": "Interstellar",
        "Animation": "Spirited Away",
        "Horror": "Get Out",
    }
    def __init__(self, users: list[User] = []):
        self.users = users

    def load_users(self, users: list[User]):
        self.users = users

    def add_user(self, user: User):
        self.users.append(user)

    def remove_user(self, user: User):
        if user in self.users:
            self.users.remove(user)
        else:
            print(f"User {user.name} not found.")

    def list_users(self):
        for user in self.users:
            print(f"{', '.join([user.name])}")

    def list_user_statistics(self, user: User):
        print(f"Name: {user.name}, Favorite Genre: {user.favorite_genre}, Favorite Movie: {user.favorite_movie}, Movies Watched: {user.movies_watched}, Subscription: {'Active' if user.subscription_status else 'Inactive'}")
        print(f"Watch History: {', '.join(user.watch_history)}")
        print(f"Recommendation: {self.recommend_for_user(user)}")
        print("")
        print("--------------------------------------------------")
            
    def list_all_users_statistics(self):
        for user in self.users:
            self.list_user_statistics(user)

    def generate_recommendations_for_all_users(self):
        for user in self.users:
            print(f"Recommendation for {user.name}: {self.recommend_for_user(user)}")
            print("")
            print("--------------------------------------------------")

    def __get_user(self, name):
        for user in self.users:
            if user.name == name:
                return user
        return None

    def find_user(self, name):
        user = self.__get_user(name)
        if user:
            return user
        else:
            print(f"User with name {name} not found.")
    
    @staticmethod
    def recommend_for_user(user: User):
        return RecommendationSystem.__recommendations.get(user.favorite_genre, "Inception")

    def __recommend_for_user(self, user):
        return self.__recommendations.get(user.favorite_genre, "Inception")

    
# empty user list in constructor for testing purposes
recommendation_system = RecommendationSystem() 

print("should be empty")
print("")

recommendation_system.list_all_users_statistics()

print("should be empty")
print("")

recommendation_system.generate_recommendations_for_all_users()

print("should not find user Alex")
print("")

user1 = recommendation_system.find_user("Alex")
if user1:
    recommendation_system.list_user_statistics(user1)


print("Lets load some users")
print("")
recommendation_system.load_users(user_objects)


print("Lets rerun the statistics and recommendations after loading users")
print("")
print("should now have user statistics")
print("")

recommendation_system.list_all_users_statistics()

print("should now have recommendations for all users")
print("")

recommendation_system.generate_recommendations_for_all_users()

print("should find user Alex")
print("")

user1 = recommendation_system.find_user("Alex")
if user1:
    recommendation_system.list_user_statistics(user1)

print("should not find user Grace")
print("")
# should not find the user and print a not found message
user2 = recommendation_system.find_user("Grace")
if not user2:
    print("User Grace not found. As expected!")

print("Adding Grace")
print("")

new_user = User("Grace", "Comedy", "Superbad", 5, True)
recommendation_system.add_user(new_user)

print("should find user Grace")
print("")
# should find the user and print stats
user2 = recommendation_system.find_user("Grace")
if user2:
    recommendation_system.list_user_statistics(user2)
    user2.add_movie_to_history("Top Gun")
    user2.display_profile()
    recommendation_system.list_user_statistics(user2)



recommendation_system.generate_recommendations_for_all_users()

# static method example
print(RecommendationSystem.recommend_for_user(user_objects[0]))