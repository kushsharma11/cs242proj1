CS242 Project 1 - TF-IDF
Kush Sharma

This project compares four books using TF-IDF and cosine similarity.

- Requires Python 3, NumPy, pandas, and Matplotlib.
- Keep main.py in the project folder.
- Put the four text files in a folder named data:
  - cap.txt: Crime and Punishment
  - fta.txt: A Farewell to Arms
  - prpr.txt: Pride and Prejudice
  - sh.txt: The Adventures of Sherlock Holmes
- Use the original text files because the code looks for specific start and end phrases.

Run these commands from the project folder on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pandas matplotlib
python3 main.py
```

- The terminal prints word counts, TF scores, an IDF summary, TF-IDF scores, and cosine similarities.
- The program creates an output folder containing four TF-IDF bar charts and one similarity heatmap.
