import pandas as pd

# Create a dictionary of users and their anxiety scores
users = {1: 10, 2: 15, 3: 8, 4: 12}

# Create a dictionary of songs and their IDs
songs = {1: 'Killing in the name of', 2: 'Firestarter', 3: 'Fantasia', 4: 'BluesbyDad'}

# Create a list of tuples with user ID, song ID,song rating, anxiety score
data = [(1, 1, 4, users[1]), (1, 2, 5, users[1]),
        (1, 3, 3, users[1]), (1, 4, 2, users[1]),
        (2, 1, 5, users[2]), (2, 2, 4, users[2]),
        (2, 3, 2, users[2]), (2, 4, 3, users[2]),
        (3, 1, 3, users[3]), (3, 2, 4, users[3]),
        (3, 3, 5, users[3]), (3, 4, 2, users[3]),
        (4, 1, 2, users[4]), (4, 2, 3, users[4]),
        (4, 3, 4, users[4]), (4, 4, 5, users[4])]

# Create the dataframe
df = pd.DataFrame(data, columns=['user_id', 'song_id', 'rating', 'anxiety_score'])

# Export the dataframe as a csv file, where Index=False
df.to_csv('D:\\Code\\MRS\\user_ratings.csv', index=False)