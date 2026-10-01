import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


book_names = ["cap", "fta", "prpr", "sh"]

book_titles = {
    "cap": "Crime and Punishment",
    "fta": "A Farewell to Arms",
    "prpr": "Pride and Prejudice",
    "sh": "The Adventures of Sherlock Holmes",
}

start_markers = {
    "cap": "On an exceptionally hot evening early in July",
    "fta": "In the late summer of that year",
    "prpr": "It is a truth universally acknowledged",
    "sh": "To Sherlock Holmes she is always",
}

end_markers = {
    "cap": "*** END OF THE PROJECT GUTENBERG",
    "fta": "THE END",
    "prpr": "CHISWICK PRESS:",
    "sh": "*** END OF THE PROJECT GUTENBERG",
}


tokens_by_book = {}
total_bytes = 0

for book_name in book_names:
    file_path = "data/" + book_name + ".txt"
    total_bytes += os.path.getsize(file_path)

    with open(file_path, "r", encoding="utf-8-sig") as book_file:
        text = book_file.read()

    start_position = text.index(start_markers[book_name])
    text = text[start_position:]

    end_position = text.index(end_markers[book_name])
    text = text[:end_position]

    text = re.sub(r"\[_Copyright.*?_\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[Illustration.*?\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\n +\[\*\].*?\n\n", "\n\n", text, flags=re.DOTALL)

    kept_lines = []

    for line in text.splitlines():
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

    text = " ".join(kept_lines).lower()
    text = re.sub(r"[_'’]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)

    words = []

    for word in text.split():
        if word.isalpha():
            words.append(word)

    tokens_by_book[book_name] = words

if total_bytes < 250000:
    raise ValueError("The selected book files must total at least 250 KB.")

print("Books loaded:", len(book_names))
print("Total file size:", round(total_bytes / 1000, 1), "KB")

for book_name in book_names:
    words = tokens_by_book[book_name]

    print(
        book_titles[book_name],
        "-", len(words), "words,",
        len(set(words)), "unique words"
    )


all_word_counts = {}

for book_name in book_names:
    all_word_counts[book_name] = pd.Series(
        tokens_by_book[book_name]
    ).value_counts()

word_document_table = pd.DataFrame(all_word_counts)
word_document_table = word_document_table.fillna(0).astype(int)
word_document_table = word_document_table.sort_index()

print("\nVocabulary size:", len(word_document_table))


tf_table = word_document_table.copy()

for book_name in book_names:
    total_words = word_document_table[book_name].sum()
    tf_table[book_name] = word_document_table[book_name] / total_words

for book_name in book_names:
    top_words = tf_table[book_name].sort_values(
        ascending=False
    ).head(10)

    print("\n" + book_titles[book_name] + " - highest term frequencies:")
    print(top_words.round(6))


number_of_books = len(book_names)
books_with_word = (word_document_table > 0).sum(axis=1)
idf_values = np.log(number_of_books / (1 + books_with_word))

print("\nIDF summary:")

for count in range(1, number_of_books + 1):
    matching_words = books_with_word[books_with_word == count]

    if len(matching_words) > 0:
        idf = idf_values.loc[matching_words.index[0]]

        print(
            count, "book(s):",
            len(matching_words), "words, IDF =",
            round(idf, 3)
        )


tfidf_table = tf_table.copy()

for book_name in book_names:
    tfidf_table[book_name] = tf_table[book_name] * idf_values

for book_name in book_names:
    top_words = tfidf_table[book_name].sort_values(
        ascending=False
    ).head(10)

    print("\n" + book_titles[book_name] + " - highest TF-IDF scores:")
    print(top_words.round(6))


def cosine_similarity(first_vector, second_vector):
    dot_product = (first_vector * second_vector).sum()

    first_length = np.sqrt((first_vector ** 2).sum())
    second_length = np.sqrt((second_vector ** 2).sum())

    return dot_product / (first_length * second_length)


similarity_table = pd.DataFrame(
    0.0,
    index=book_names,
    columns=book_names
)

for first_book in book_names:
    for second_book in book_names:
        similarity_table.loc[first_book, second_book] = cosine_similarity(
            tfidf_table[first_book],
            tfidf_table[second_book]
        )

print("\nCosine similarity between books:")
print(similarity_table.round(3))


os.makedirs("output", exist_ok=True)

maximum_score = tfidf_table.max().max()

for book_name in book_names:
    top_words = tfidf_table[book_name].sort_values(
        ascending=False
    ).head(10)

    plot_words = top_words.sort_values()

    plt.figure(figsize=(8, 5))
    plt.barh(plot_words.index, plot_words.values)
    plt.xlim(0, maximum_score * 1.1)
    plt.xlabel("TF-IDF score")
    plt.title("Top 10 TF-IDF words: " + book_titles[book_name])
    plt.tight_layout()
    plt.savefig("output/" + book_name + "_tfidf.png", dpi=300)
    plt.close()


labels = []

for book_name in book_names:
    labels.append(book_titles[book_name])


plt.figure(figsize=(9, 7))
plt.imshow(similarity_table, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(label="Cosine similarity")

plt.xticks(
    range(len(book_names)),
    labels,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(book_names)),
    labels
)

plt.title("Similarity between books")

for row in range(len(book_names)):
    for column in range(len(book_names)):
        score = similarity_table.iloc[row, column]

        if score > 0.5:
            text_color = "white"
        else:
            text_color = "black"

        plt.text(
            column,
            row,
            str(round(score, 3)),
            ha="center",
            va="center",
            color=text_color
        )

plt.tight_layout()
plt.savefig("output/book_similarity.png", dpi=300)
plt.close()

print("\nSaved five plots in the output folder.")