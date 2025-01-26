from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import pandas as pd
import numpy as np
import ast
from tempfile import mkdtemp
import random
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
import math
import os
import glob
import re

from sklearn.metrics.pairwise import cosine_similarity  # For CF & content-based similarities

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = mkdtemp()
app.secret_key = 'your_secret_key'
Session(app)

# -------------------------------------------------------------------------------------
# 1) LOAD DATASETS
# -------------------------------------------------------------------------------------

# Main recipes dataset for nutritional info
df_recipes = pd.read_csv('./Dataset/medlineplus/cleaned_recipes.csv', sep=';', encoding='latin1')

# The user’s actual product/meal names from MyFitnessPal (we'll use user_id=31)
df_productnames = pd.read_csv('./Dataset/MyFitnessPal/user_product_names.csv')

# Example user history data
history_data = pd.read_csv('./Dataset/MyFitnessPal/user31_history.csv', sep=';')

# Collaborative filtering datasets
# a) Combined ratings from 15 sample users, plus a "user16" column that we fill in
df_combined = pd.read_csv('./Dataset/medlineplus/combined_ratings.csv', sep=';', encoding='latin1')
df_combined.drop('Link', axis=1, inplace=True)  # ignore the Link column

# b) This webapp user’s individual ratings => "user16"
df_user16 = pd.read_csv('./Dataset/medlineplus/recipes_ratings.csv', sep=';', encoding='latin1')
df_user16.drop('Link', axis=1, inplace=True)  # ignore Link column
df_user16.rename(columns={'Rating (0-5)': 'user16'}, inplace=True)

# Merge user16 ratings into df_combined
for idx, row in df_user16.iterrows():
    title = row['Title']
    rating_val = row['user16']
    df_combined.loc[df_combined['Title'] == title, 'user16'] = rating_val

# Create user-rating matrix from df_combined, using Title as index
df_combined.set_index('Title', inplace=True)
# columns: [user1, user2, ..., user15, user16]

# -------------------------------------------------------------------------------------
# 1b) LOAD PRECOMPUTED EMBEDDINGS (CBF)
# -------------------------------------------------------------------------------------
embedding_folder = './Dataset/embeddings/'
df_titles = pd.read_csv(os.path.join(embedding_folder, 'recipe_titles.csv'))
all_embeddings = np.load(os.path.join(embedding_folder, 'recipe_embeddings.npy'))

# Build a lookup from Title -> embedding vector
title_to_embedding = {}
for i, row in df_titles.iterrows():
    title_to_embedding[row['Title']] = all_embeddings[i]

# -------------------------------------------------------------------------------------
# 1c) BUILD USER 31's CONTENT EMBEDDING
# -------------------------------------------------------------------------------------
df_user31 = df_productnames[df_productnames['user_id'] == 31]
user31_history = []
for product_str in df_user31['product_name']:
    try:
        products_list = ast.literal_eval(product_str)
        user31_history.extend(products_list)
    except:
        user31_history.append(product_str)

# Filter out any None or empty strings
user31_history = [x for x in user31_history if isinstance(x, str) and x.strip()]

user31_vectors = []
for item in user31_history:
    if item in title_to_embedding:
        user31_vectors.append(title_to_embedding[item])

if len(user31_vectors) == 0:
    user31_embedding = None
else:
    user31_embedding = np.mean(user31_vectors, axis=0)

# -------------------------------------------------------------------------------------
# 2) CF MODEL (Return CF "score" for every item)
# -------------------------------------------------------------------------------------

def build_user_based_cf(df_ratings, target_user='user16'):
    """
    - If user16 rated it, use that rating as CF score.
    - Otherwise, predict via user-based CF.
    """
    user_item_matrix = df_ratings.T.fillna(0)
    sim_matrix = cosine_similarity(user_item_matrix)
    sim_df = pd.DataFrame(sim_matrix,
                          index=user_item_matrix.index,
                          columns=user_item_matrix.index)
    user_similarities = sim_df.loc[target_user]
    target_user_ratings = user_item_matrix.loc[target_user]

    predictions = {}
    for item in user_item_matrix.columns:
        user16_rating = target_user_ratings[item]
        if user16_rating > 0:
            predictions[item] = float(user16_rating)
        else:
            item_ratings = user_item_matrix[item]
            mask = (item_ratings > 0)
            relevant_users = item_ratings[mask].index
            if len(relevant_users) == 0:
                predictions[item] = 0.0
                continue

            num = 0.0
            den = 0.0
            for other_u in relevant_users:
                sim_val = user_similarities[other_u]
                r = item_ratings[other_u]
                num += sim_val * r
                den += abs(sim_val)
            if den == 0:
                predictions[item] = 0.0
            else:
                predictions[item] = num / den
    return predictions

def get_all_cf_scores():
    return build_user_based_cf(df_combined, 'user16')

# -------------------------------------------------------------------------------------
# 3) CONTENT-BASED SCORES (User31 embedding -> each Title)
# -------------------------------------------------------------------------------------

def get_all_cbf_scores(user_embedding):
    cbf_scores = {}
    if user_embedding is None:
        for t in title_to_embedding.keys():
            cbf_scores[t] = 0.0
        return cbf_scores

    for title, emb in title_to_embedding.items():
        dot = np.dot(user_embedding, emb)
        normA = np.linalg.norm(user_embedding)
        normB = np.linalg.norm(emb)
        sim = 0.0
        if normA != 0 and normB != 0:
            sim = dot / (normA * normB)
        cbf_scores[title] = float(sim)
    return cbf_scores

# -------------------------------------------------------------------------------------
# 4) HYBRID SCORES
# -------------------------------------------------------------------------------------

def get_hybrid_scores(alpha=0.5):
    cf_scores = get_all_cf_scores()
    cbf_scores = get_all_cbf_scores(user31_embedding)

    all_titles = set(cf_scores.keys()).union(cbf_scores.keys())
    final_scores = {}
    for t in all_titles:
        cf = cf_scores.get(t, 0.0)
        cbf = cbf_scores.get(t, 0.0)
        final_scores[t] = alpha * cf + (1 - alpha) * cbf
    return final_scores

def get_top_n_hybrid_recommendations(n=200, alpha=0.5):
    final_scores = get_hybrid_scores(alpha=alpha)
    sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_n = [title for (title, score) in sorted_items[:n]]
    return top_n

# -------------------------------------------------------------------------------------
# 5) USER & RDI SETUP, FLASK ROUTES
# -------------------------------------------------------------------------------------

user = {
    "weight": 70,
    "height": 175,
    "age": 30,
    "gender": "male",
    "activity_level": 1.55,
    "goal": "maintain",
    "allergies": [],   # e.g. ["chicken", "gluten"]
    "dislikes": [],    # e.g. ["broccoli", "fish"]
    "diet": ""         # e.g. "vegetarian" or ""
}

def calculate_bmr(weight, height, age, gender):
    if gender == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

bmr = calculate_bmr(user['weight'], user['height'], user['age'], user['gender'])
tdee = bmr * user['activity_level']
rdi = {
    "Calories": int(tdee),
    "Protein (g)": int(tdee * 0.25 / 4),
    "Total Fat (g)": int(tdee * 0.25 / 9),
    "Sodium (mg)": 1500,
    "Dietary Fiber (g)": 25,
    "Calcium (mg)": 1000,
    "Iron (mg)": 18,
    "Vitamin D(mcg)": 15,
    "Vitamin A (mcg)": 900,
    "Vitamin C (mg)": 90,
    "Potassium (mg)": 4700,
    "Total Sugar (g)": 36,
    "Cholesterol (mg)": 300,
    "Saturated Fat (g)": 20
}

user_intake = {k: 0 for k in rdi.keys()}
chosen_meals = []
rejected_meals = []

def plot_nutrients_in_subplots(user_intake, rdi, nutrient_cols, cols=4):
    n = len(nutrient_cols)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), sharey=False)

    if n == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for i, nutrient in enumerate(nutrient_cols):
        ax = axes[i]
        intake_val = user_intake.get(nutrient, 0)
        rdi_val = rdi.get(nutrient, 0)

        x_positions = [0, 1]
        heights = [intake_val, rdi_val]

        ax.bar(x_positions, heights, color=['steelblue', 'orange'], width=0.6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['Intake', 'RDI'])
        ax.set_title(nutrient, fontsize=10)
        max_val = max(intake_val, rdi_val)
        ax.set_ylim(0, max_val * 1.2 if max_val > 0 else 1)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.before_request
def initialize_session():
    if 'chosen_meals' not in session:
        session['chosen_meals'] = []

@app.route('/')
def home():
    nutrient_plot = plot_nutrients_in_subplots(user_intake, rdi, list(rdi.keys()))
    return render_template('index.html',
                           meal=None,
                           user_intake=user_intake,
                           rdi=rdi,
                           nutrient_plot=nutrient_plot,
                           chosen_meals=session.get('chosen_meals', []))

# -------------------------------------------------------------------------------------
# 6) FILTERING LOGIC: ALLERGIES, DISLIKES, DIET
# -------------------------------------------------------------------------------------

def filter_by_allergies_dislikes_diet(df, user):
    """
    1) Filter out any row with an allergy or dislike word (exact word match in Ingredients).
    2) If user['diet'] == 'vegetarian', keep only Vegetarian==1 if that column exists,
       or fallback to removing typical meats from Ingredients.
    """

    # Clean up allergies/dislikes to remove empty strings
    allergies_clean = [a.strip().lower() for a in user['allergies'] if a.strip()]
    dislikes_clean = [d.strip().lower() for d in user['dislikes'] if d.strip()]

    # 1) Filter allergies
    for allergen in allergies_clean:
        if df.empty:
            break
        pattern = rf"\b{re.escape(allergen)}\b"
        # Keep rows that do NOT contain that allergen
        mask = ~df['Ingredients'].fillna('').str.lower().str.contains(pattern, regex=True)
        df = df[mask]

    # 2) Filter dislikes
    for dislike in dislikes_clean:
        if df.empty:
            break
        pattern = rf"\b{re.escape(dislike)}\b"
        mask = ~df['Ingredients'].fillna('').str.lower().str.contains(pattern, regex=True)
        df = df[mask]


    # 3) Filter diet
    diet_lower = user['diet'].strip().lower()
    if not df.empty and diet_lower == 'vegetarian':
        if 'Vegetarian' in df.columns:
            # Only keep rows where Vegetarian == 1 (or 1.0)
            df_veg = df[df['Vegetarian'] == 1]
            if not df_veg.empty:
                df = df_veg
            else:
                # if df_veg is empty, everything is 0 => user has no vegetarian items
                pass
                # optionally fallback to removing "chicken","beef", etc. if you wish

    return df

# -------------------------------------------------------------------------------------
# 7) HYBRID RECOMMENDATION + FILTERS
# -------------------------------------------------------------------------------------

def recommend_with_category(category, alpha=0.5):
    top_hybrid_titles = get_top_n_hybrid_recommendations(n=50, alpha=alpha)

    df_category = df_recipes[df_recipes[category] == 1]
    df_category = filter_by_allergies_dislikes_diet(df_category, user)
    recommended_in_cat = df_category[df_category['Title'].isin(top_hybrid_titles)]

    already_used = [m['title'] for m in session.get('chosen_meals', [])] + rejected_meals
    filtered = recommended_in_cat[~recommended_in_cat['Title'].isin(already_used)]

    if not filtered.empty:
        for t in top_hybrid_titles:
            if t in filtered['Title'].values:
                chosen_row = filtered[filtered['Title'] == t].iloc[0]
                break

        nutrient_plot = plot_nutrients_in_subplots(user_intake, rdi, list(rdi.keys()))
        return render_template(
            'index.html',
            meal={
                'title': chosen_row['Title'],
                'nutrients': chosen_row[rdi.keys()].to_dict(),
                'image_url': chosen_row['Image-URL'],
                'serving_size': chosen_row.get('Serving Size (g)', 'N/A')
            },
            category=category,  # <--- pass category here!
            user_intake=user_intake,
            rdi=rdi,
            nutrient_plot=nutrient_plot,
            chosen_meals=session.get('chosen_meals', [])
        )
    else:
        nutrient_plot = plot_nutrients_in_subplots(user_intake, rdi, list(rdi.keys()))
        return render_template(
            'index.html',
            category=category,  # <--- pass category here too
            error="No more Hybrid recommendations in this category.",
            user_intake=user_intake,
            rdi=rdi,
            nutrient_plot=nutrient_plot,
            chosen_meals=session.get('chosen_meals', [])
        )


@app.route('/recommend', methods=['POST'])
def recommend():
    category = request.form.get('category', 'Breakfast')
    return recommend_with_category(category, alpha=0.5)

@app.route('/accept', methods=['POST'])
def accept():
    meal_title = request.form.get('title')
    servings = float(request.form.get('servings', 1))

    df_meal = df_recipes[df_recipes['Title'] == meal_title]
    if not df_meal.empty:
        meal_data = df_meal.iloc[0]
        for key in rdi.keys():
            user_intake[key] += meal_data[key] * servings
        
        chosen_meals = session.get('chosen_meals', [])
        chosen_meals.append({'title': meal_title, 'servings': servings})
        session['chosen_meals'] = chosen_meals

    nutrient_plot = plot_nutrients_in_subplots(user_intake, rdi, list(rdi.keys()))
    return render_template('index.html',
                           success=f"Added {servings} serving(s) of {meal_title}.",
                           user_intake=user_intake,
                           rdi=rdi,
                           nutrient_plot=nutrient_plot,
                           chosen_meals=session.get('chosen_meals', []))

@app.route('/reject', methods=['POST'])
def reject():
    meal_title = request.form.get('title')
    rejected_meals.append(meal_title)

    category = request.form.get('category', 'Breakfast')
    return recommend_with_category(category, alpha=0.5)

# -------------------------------------------------------------------------------------
# 8) PROFILE, HISTORY, RATING
# -------------------------------------------------------------------------------------

@app.route('/profile')
def profile():
    return render_template('profile.html', user=user)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    global user

    user['weight'] = int(request.form.get('weight'))
    user['height'] = int(request.form.get('height'))
    user['age'] = int(request.form.get('age'))
    user['gender'] = request.form.get('gender')
    user['activity_level'] = float(request.form.get('activity_level'))
    user['goal'] = request.form.get('goal')
    user['allergies'] = [x.strip() for x in request.form.get('allergies', '').split(',')]
    user['dislikes'] = [x.strip() for x in request.form.get('dislikes', '').split(',')]
    user['diet'] = request.form.get('diet')

    # Recalc RDI
    global rdi
    bmr = calculate_bmr(user['weight'], user['height'], user['age'], user['gender'])
    tdee = bmr * user['activity_level']
    rdi = {
        "Calories": int(tdee),
        "Protein (g)": int(tdee * 0.25 / 4),
        "Total Fat (g)": int(tdee * 0.25 / 9),
        "Sodium (mg)": 1500,
        "Dietary Fiber (g)": 25,
        "Calcium (mg)": 1000,
        "Iron (mg)": 18,
        "Vitamin D(mcg)": 15,
        "Vitamin A (mcg)": 900,
        "Vitamin C (mg)": 90,
        "Potassium (mg)": 4700,
        "Total Sugar (g)": 36,
        "Cholesterol (mg)": 300,
        "Saturated Fat (g)": 20,
    }

    return render_template('profile.html', user=user, success="Profile updated successfully!")

@app.route('/history', methods=['GET', 'POST'])
def history():
    timeframe = request.form.get('timeframe', '1')
    months = int(timeframe)

    history_data['date'] = pd.to_datetime(history_data['date'])
    cutoff_date = history_data['date'].max() - pd.DateOffset(months=months)
    filtered_data = history_data[history_data['date'] >= cutoff_date]

    days_in_timeframe = (filtered_data['date'].max() - filtered_data['date'].min()).days + 1
    nutrient_totals = filtered_data.drop(columns=['user_id', 'date']).sum().to_dict()
    adjusted_rdi = {nutrient: rdi_value * days_in_timeframe for nutrient, rdi_value in rdi.items()}

    intake_vs_rdi = {
        nutrient: {'intake': nutrient_totals.get(nutrient, 0), 'rdi': adjusted_rdi.get(nutrient, 0)}
        for nutrient in rdi.keys()
    }

    nutrient_plot = plot_nutrients_in_subplots(nutrient_totals, adjusted_rdi, list(rdi.keys()))
    return render_template('history.html',
                           intake_vs_rdi=intake_vs_rdi,
                           nutrient_plot=nutrient_plot,
                           timeframe=timeframe)

@app.route('/rating', methods=['GET', 'POST'])
def rating():
    recipes_path = './Dataset/medlineplus/recipes_ratings.csv'
    recipes_data = pd.read_csv(recipes_path, sep=';', encoding='latin1')

    if 'Rating (0-5)' not in recipes_data.columns:
        recipes_data['Rating (0-5)'] = 0
    else:
        recipes_data['Rating (0-5)'] = recipes_data['Rating (0-5)'].fillna(0).astype(int)

    search_query = request.form.get('search_query', '').lower()
    if search_query:
        filtered_recipes = recipes_data[recipes_data['Title'].str.contains(search_query, case=False)]
    else:
        filtered_recipes = recipes_data

    recipes = filtered_recipes[['Title', 'Rating (0-5)']].rename(columns={'Rating (0-5)': 'Rating'}).to_dict(orient='records')
    return render_template('rating.html', recipes=recipes)

@app.route('/rate', methods=['POST'])
def rate():
    recipes_path = './Dataset/medlineplus/recipes_ratings.csv'
    recipes_data = pd.read_csv(recipes_path, sep=';', encoding='latin1')
    recipes_data = recipes_data.loc[:, ~recipes_data.columns.str.contains('^Unnamed')]

    if 'Rating (0-5)' not in recipes_data.columns:
        recipes_data['Rating (0-5)'] = 0

    title = request.form.get('title')
    rating = int(request.form.get('rating', 0))

    recipes_data.loc[recipes_data['Title'] == title, 'Rating (0-5)'] = rating
    recipes_data.to_csv(recipes_path, sep=';', index=False, encoding='latin1')

    recipes_data = pd.read_csv(recipes_path, sep=';', encoding='latin1')
    recipes = recipes_data[['Title', 'Rating (0-5)']].rename(columns={'Rating (0-5)': 'Rating'}).to_dict(orient='records')
    return render_template('rating.html', recipes=recipes)

# -------------------------------------------------------------------------------------
# 9) MAIN
# -------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)