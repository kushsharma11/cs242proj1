import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

books = {}
total_bytes = 0

file_names = os.listdir("data")
file_names.sort()

for file_name in file_names:

    if file_name.endswith(".txt"):
        file_path = "data/" + file_name

        with open(file_path, "r", encoding="utf-8-sig") as book_file:
            book_text = book_file.read()

        book_name = file_name[:-4]

        books[book_name] = book_text

        file_size = os.path.getsize(file_path)
        total_bytes = total_bytes + file_size

        character_count = len(book_text)
        print(file_name, "-", character_count, "characters loaded")

total_kilobytes = total_bytes / 1000

print()
print("books loaded:", len(books))
print("file size:", round(total_kilobytes, 1), "KB")


start_markers = {
    "cap": "On an exceptionally hot evening early in July",
    "fta": "In the late summer of that year",
    "prpr": "It is a truth universally acknowledged",
    "sh": "To Sherlock Holmes she is always"
}

end_markers = {
    "cap": "*** END OF THE PROJECT GUTENBERG",
    "fta": "THE END",
    "prpr": "CHISWICK PRESS:",
    "sh": "*** END OF THE PROJECT GUTENBERG"
}

tokens_by_book = {}

for book_name in books:
    text = books[book_name]

    start_position = text.index(start_markers[book_name])
    text = text[start_position:]

    end_position = text.index(end_markers[book_name])
    text = text[:end_position]

    text = re.sub(r"\[_Copyright.*?_\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[Illustration.*?\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\n +\[\*\].*?\n\n", "\n\n", text, flags=re.DOTALL)

    lines = text.splitlines()
    kept_lines = []

    for line in lines:
        line = line.strip()
        upper_line = line.upper()

        if upper_line.startswith("CHAPTER "):
            continue
        if upper_line.startswith("PART "):
            continue
        if upper_line.startswith("BOOK "):
            continue
        if upper_line == "EPILOGUE":
            continue
        if re.fullmatch(r"[IVXLCDM]+\.?", line):
            continue

        if book_name == "sh" and line.isupper():
            if re.match(r"[IVXLCDM]+\.", line):
                continue

        kept_lines.append(line)

    text = " ".join(kept_lines)

    text = text.lower()
    text = text.replace("_", "")

    text = re.sub(r"['’]", "", text)

    text = re.sub(r"[^\w\s]", " ", text)

    all_words = text.split()
    words = []

    for word in all_words:
        if word.isalpha():
            words.append(word)

    tokens_by_book[book_name] = words

    print()
    print(book_name, "-", len(words), "words after cleaning")
    print("First 10 words:", words[:10])


all_word_counts = {}

for book_name in tokens_by_book:
    book_word_counts = {}

    for word in tokens_by_book[book_name]:
        if word in book_word_counts:
            book_word_counts[word] = book_word_counts[word] + 1
        else:
            book_word_counts[word] = 1

    all_word_counts[book_name] = book_word_counts

word_document_table = pd.DataFrame(all_word_counts)
word_document_table = word_document_table.fillna(0)
word_document_table = word_document_table.astype(int)
word_document_table = word_document_table.sort_index()

print()
print("Word-document table:")
print(word_document_table.head(10))

print()
print("Table shape:", word_document_table.shape)

print()
print("Total words in each book:")
print(word_document_table.sum())

tf_table = word_document_table.copy()

for book_name in word_document_table.columns:
    total_words = word_document_table[book_name].sum()

    tf_table[book_name] = word_document_table[book_name] / total_words

for book_name in tf_table.columns:
    sorted_words = tf_table[book_name].sort_values(ascending=False)
    top_words = sorted_words.head(10)

    print()
    print(book_name, "- highest term frequencies:")
    print(top_words)

print()
print("TF column totals:")
print(tf_table.sum())


number_of_books = len(word_document_table.columns)

# True means the word appears in that book.
# False means its count is zero.
word_appears = word_document_table > 0

# Add across each row to count the books containing that word.
books_with_word = word_appears.sum(axis=1)

# Calculate IDF using the assignment's formula.
denominator = books_with_word + 1
ratio = number_of_books / denominator
idf_values = np.log(ratio)

# Put the document counts and IDF values together.
idf_table = pd.DataFrame(index=word_document_table.index)
idf_table["books_with_word"] = books_with_word
idf_table["idf"] = idf_values

# Sort from highest IDF to lowest.
sorted_idf_table = idf_table.sort_values(by="idf", ascending=False)

print()
print("Words with the highest IDF:")
print(sorted_idf_table.head(10))

print()
print("Words with the lowest IDF:")
print(sorted_idf_table.tail(10))

# Make a separate table for TF-IDF scores.
tfidf_table = tf_table.copy()

for book_name in tf_table.columns:
    tfidf_table[book_name] = tf_table[book_name] * idf_values

# Show the 10 highest TF-IDF scores for each book.
for book_name in tfidf_table.columns:
    sorted_words = tfidf_table[book_name].sort_values(ascending=False)
    top_words = sorted_words.head(10)

    print()
    print(book_name, "- highest TF-IDF scores:")
    print(top_words)

def cosine_similarity(first_vector, second_vector):
    # Multiply matching values, then add the products.
    products = first_vector * second_vector
    dot_product = products.sum()

    # Calculate the length of each vector.
    first_squares = first_vector ** 2
    first_length = np.sqrt(first_squares.sum())

    second_squares = second_vector ** 2
    second_length = np.sqrt(second_squares.sum())

    denominator = first_length * second_length

    # Similarity is undefined if either vector has zero length.
    if denominator == 0:
        return np.nan

    similarity = dot_product / denominator
    return similarity


book_names = list(tfidf_table.columns)

similarity_table = pd.DataFrame(
    0.0,
    index=book_names,
    columns=book_names
)

# Compare every book with every book.
for first_book in book_names:
    first_vector = tfidf_table[first_book]

    for second_book in book_names:
        second_vector = tfidf_table[second_book]

        score = cosine_similarity(first_vector, second_vector)
        similarity_table.loc[first_book, second_book] = score

print()
print("Cosine similarity between books:")
print(similarity_table.round(3))

os.makedirs("output", exist_ok=True)

# Save the numerical results.
word_document_table.to_csv("output/word_counts.csv", index_label="word")
tf_table.to_csv("output/tf.csv", index_label="word")
idf_table.to_csv("output/idf.csv", index_label="word")
tfidf_table.to_csv("output/tfidf.csv", index_label="word")
similarity_table.to_csv("output/similarity.csv", index_label="book")

book_titles = {
    "cap": "Crime and Punishment",
    "fta": "A Farewell to Arms",
    "prpr": "Pride and Prejudice",
    "sh": "The Adventures of Sherlock Holmes"
}

# Use the same horizontal scale for all four bar charts.
highest_scores = tfidf_table.max()
maximum_score = highest_scores.max()

for book_name in tfidf_table.columns:
    sorted_words = tfidf_table[book_name].sort_values(ascending=False)
    top_words = sorted_words.head(10)
    plot_words = top_words.sort_values()

    plt.figure(figsize=(8, 5))
    plt.barh(plot_words.index, plot_words.values)
    plt.xlim(0, maximum_score * 1.1)
    plt.xlabel("TF-IDF score")
    plt.title("Top 10 TF-IDF words: " + book_titles[book_name])
    plt.tight_layout()

    file_name = "output/" + book_name + "_tfidf.png"
    plt.savefig(file_name, dpi=300)
    plt.close()

# Use full book titles on the similarity heatmap.
labels = []

for book_name in book_names:
    labels.append(book_titles[book_name])

plt.figure(figsize=(9, 7))
plt.imshow(similarity_table, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(label="Cosine similarity")

plt.xticks(range(len(book_names)), labels, rotation=45, ha="right")
plt.yticks(range(len(book_names)), labels)
plt.title("Similarity between books")

# Put the numerical score inside each square.
for row in range(len(book_names)):
    for column in range(len(book_names)):
        score = similarity_table.iloc[row, column]
        label = str(round(score, 3))

        plt.text(
            column, row, label,
            ha="center", va="center", color="white"
        )

plt.tight_layout()
plt.savefig("output/book_similarity.png", dpi=300)
plt.close()

print()
print("Saved the tables and five plots in the output folder.")
