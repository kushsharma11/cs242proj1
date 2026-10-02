# cs242proj1
CS242 Project 1 - README
Kush Sharma
This project compares four books using TF-IDF and cosine similarity. It cleans the text,
counts words, calculates scores, and saves plots.
Files
Keep main.py in the project folder and the four text files in a folder named data.
File Description
main.py Python program
data/cap.txt Crime and Punishment
data/fta.txt A Farewell to Arms
data/prpr.txt Pride and Prejudice
data/sh.txt The Adventures of Sherlock Holmes
Setup and Run
Requires Python 3, NumPy, pandas, and Matplotlib. From the project folder on macOS or
Linux, run:
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pandas matplotlib
python3 main.py
Output
The terminal shows word counts, the highest TF and TF-IDF scores, an IDF summary, and
cosine similarities.
The program creates an output folder and saves five plots: cap_tfidf.png, fta_tfidf.png,
prpr_tfidf.png, sh_tfidf.png, and book_similarity.png.
Use the original text files for this project because the cleaning code looks for specific start
and end phrases in each book.
