"""
process_exams.py
----------------
Batch-processes all PDF exam files in a given folder.
For each PDF it:
  1. Extracts the questions
  2. Extracts the answer options (A–D)
  3. Detects the correct answer (yellow-highlighted option)
  4. Saves the result as a CSV with the same base name as the PDF

Usage:
    python process_exams.py <folder_path>

    If no folder is provided the current working directory is used.
"""

import sys
import os

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend – suppresses plot windows

import fitz                    # PyMuPDF
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Functions – copied verbatim from the notebook (no logic changes)
# ---------------------------------------------------------------------------

def extract_questions(pdf_path):
    doc = fitz.open(pdf_path)
    questions_data = []  # List to store questions

    # Updated regular expression: handle questions with or without a '?'
    # ^  + MULTILINE  → question numbers must start a line (avoids matching decimals like "1.2")
    # lookahead breakdown:
    #   \nA\.[^\n]*                     – newline + "A." + rest of first A line (may be empty)
    #   (?:\n(?!A\.|B\.|C\.|D\.)[^\n]*)*  – zero or more continuation lines for option A that do
    #                                       NOT start with an option letter; this handles PDFs where
    #                                       option A wraps to the next line. The moment a line starts
    #                                       with A./B./C./D. the continuation stops, which also
    #                                       preserves the fix for "HTA" word-wrap artefacts (those
    #                                       fake "A." lines are followed by more question text that
    #                                       eventually hits the real "A." – at that point the negative
    #                                       lookahead blocks further continuation and "B." is not
    #                                       found on the next line, so the fake match is rejected).
    #   \nB\.                           – the real option B must follow
    question_pattern = re.compile(
        r"^(\d+)\.\s(.*?\??)\s*(?=\nA\.[^\n]*(?:\n(?!A\.|B\.|C\.|D\.)[^\n]*)*\nB\.)",
        re.DOTALL | re.MULTILINE
    )

    for page in doc:
        text = page.get_text("text")  # Extract full page text

        # Find all questions in the text using findall()
        questions = question_pattern.findall(text)

        # Append all the questions to the questions_data list
        for _, question in questions:
            questions_data.append(question.strip())  # Store the cleaned question text

    # Convert extracted questions to DataFrame
    questions_df = pd.DataFrame(questions_data, columns=["question"])
    return questions_df


def extract_answers(pdf_path):
    doc = fitz.open(pdf_path)
    answers_data = []  # List to store answers for each question

    # Regular expression for extracting answers
    answer_pattern = re.compile(r"^(A|B|C|D)\.\s(.+)", re.MULTILINE)

    next_expected = {"A": "B", "B": "C", "C": "D"}

    for page in doc:
        text = page.get_text("text")  # Extract full page text
        lines = text.split("\n")  # Split into lines for structured parsing

        current_answers = []  # List to store answers for a given question
        expected_option = "A"   # Enforce strict A → B → C → D ordering

        for line in lines:
            answer_match = answer_pattern.match(line)
            if answer_match:
                opt, ans_text = answer_match.groups()

                if opt == expected_option:
                    current_answers.append(ans_text.strip())
                    if opt == "D":
                        # All four options collected – store and reset
                        answers_data.append(current_answers)
                        current_answers = []
                        expected_option = "A"
                    else:
                        expected_option = next_expected[opt]
                elif opt == "A":
                    # A second "A" when we expected B/C/D means the previous "A" was
                    # a false match (e.g. word-wrap artefact like "HTA" → "A."). Reset.
                    current_answers = [ans_text.strip()]
                    expected_option = "B"
                # Unexpected B / C / D → skip (out-of-order option, ignore it)

    # Convert extracted answers to DataFrame with four columns
    answers_df = pd.DataFrame(answers_data, columns=["option_A", "option_B", "option_C", "option_D"])
    return answers_df


def get_largest_rectangle(rects):
    """Return the largest rectangle from a list of rectangles."""
    if not rects:
        return None

    # Calculate area for each rectangle and store with index
    areas = [(abs((rect.br.x - rect.tl.x) * (rect.br.y - rect.tl.y)), idx)
             for idx, rect in enumerate(rects)]

    # Sort by area (first element of tuple)
    areas.sort(key=lambda x: x[0], reverse=True)

    # Return the rectangle with largest area
    return rects[areas[0][1]]


def extract_correct_answers(pdf_path):
    doc = fitz.open(pdf_path)
    correct_answers = []
    missing_questions = []  # Track questions that were skipped

    # Flexible regex for answer choices (handles missing spaces like "A." and "A. ")
    answer_pattern = re.compile(r"^(A|B|C|D)\.\s(.+)", re.MULTILINE)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # Extract full page text
        lines = text.split("\n")  # Split into lines

        # Extract valid answer positions using regex
        answer_positions = {key: [] for key in ["A", "B", "C", "D"]}

        for line in lines:
            match = answer_pattern.match(line)
            if match:
                answer_letter, answer_text = match.groups()

                # Try different search variations to find the best match
                search_texts = [
                    match.group(0),                           # Original matched text
                    match.group(0).strip(),                   # Stripped version
                    answer_letter + '.' + answer_text,        # Without space after period
                    answer_letter.upper() + '. ' + answer_text,  # Different case
                ]

                rects = None
                used_search_text = ""

                # Find the first successful search
                for search_text in search_texts:
                    rects = page.search_for(search_text)
                    if rects:
                        used_search_text = search_text
                        break

                if rects:
                    # Select the largest rectangle
                    largest_rect = get_largest_rectangle(rects)
                    if largest_rect:
                        answer_positions[answer_letter].append(largest_rect)

        # Find the maximum number of questions on the page
        num_questions = max(len(answer_positions["A"]), len(answer_positions["B"]),
                            len(answer_positions["C"]), len(answer_positions["D"]))

        # Debugging: Show how many answers were found
        print(f"\nPage {page_num}: Found {num_questions} questions")

        for idx in range(num_questions):
            best_answer = None
            best_yellow_pixels = 0  # Track the highest number of yellow pixels
            has_all_options = True  # Flag to check if we have all 4 options

            for answer_letter in ["A", "B", "C", "D"]:
                rects = answer_positions[answer_letter]
                if idx >= len(rects):
                    has_all_options = False  # Missing one or more options
                    continue

                rect = rects[idx]

                # Debugging: Print detected answer positions
                print(f"Page {page_num}, Q{idx+1} - {answer_letter}: Detected box: "
                      f"x0={rect.x0:.1f}, y0={rect.y0:.1f}, x1={rect.x1:.1f}, y1={rect.y1:.1f}")

                # Expand bounding box only to the right
                expanded_rect = fitz.Rect(rect.x0, rect.y0, rect.x1 + 50, rect.y1)

                # Extract pixels from the expanded region
                pixmap = page.get_pixmap(clip=expanded_rect)
                img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.h, pixmap.w, pixmap.n)

                # Count yellow pixels (pixels with high red and green, low blue)
                yellow_mask = (
                    (img[:, :, 0] > 200) &  # High red
                    (img[:, :, 1] > 200) &  # High green
                    (img[:, :, 2] < 150)    # Low blue
                )
                yellow_pixels = np.sum(yellow_mask)

                # Count total non-white pixels
                nonwhite_mask = ~np.all(img > 250, axis=2)
                total_pixels = np.sum(nonwhite_mask)

                # Debug information
                print(f"Page {page_num}, Q{idx+1}, {answer_letter}: Yellow pixels: {yellow_pixels}")
                print(f"Page {page_num}, Q{idx+1}, {answer_letter}: Total non-white pixels: {total_pixels}")

                if total_pixels > 0:
                    yellow_percentage = (yellow_pixels / total_pixels) * 100
                    print(f"Page {page_num}, Q{idx+1}, {answer_letter}: Yellow percentage: {yellow_percentage:.1f}%")

                # Display the box in the notebook
                plt.figure(figsize=(8, 4))
                plt.imshow(img)
                plt.gca().add_patch(plt.Rectangle((0, 0), expanded_rect.width, expanded_rect.height,
                                                  fill=False, color='red', linewidth=2))
                plt.title(f'Page {page_num}, Q{idx+1}, Answer {answer_letter}\nYellow pixels: {yellow_pixels}')
                plt.axis('off')
                plt.show()
                plt.close()   # Release figure memory when running in batch mode

                # Store the best answer based on the highest number of yellow pixels
                if yellow_pixels > best_yellow_pixels:
                    best_yellow_pixels = yellow_pixels
                    best_answer = answer_letter

            # If missing options, log missing questions
            if not has_all_options:
                missing_questions.append(f"Page {page_num}, Q{idx+1}")

            if best_answer:
                correct_answers.append(best_answer)
                print(f"\nPage {page_num}, Q{idx+1} - Selected Answer: {best_answer} (based on yellow pixel count)")
                print("=" * 100)
                print("\n")
                print("\n")

    # Debugging: Print missing questions
    if missing_questions:
        print("\n⚠️ The following questions were skipped due to missing options:")
        for q in missing_questions:
            print(q)

    return pd.DataFrame(correct_answers, columns=["correct_answer"])


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_pdf(pdf_file):
    """Run the full extraction pipeline for a single PDF and save a CSV."""
    print(f"\n{'=' * 80}")
    print(f"Processing: {os.path.basename(pdf_file)}")
    print(f"{'=' * 80}")

    questions_df     = extract_questions(pdf_path=pdf_file)
    answers_df       = extract_answers(pdf_path=pdf_file)
    correct_answers_df = extract_correct_answers(pdf_path=pdf_file)

    total_df = pd.concat([questions_df, answers_df, correct_answers_df],
                         ignore_index=True, axis=1)
    total_df.columns = ["questions", "option_A", "option_B", "option_C", "option_D", "correct_answer"]

    # Extract folder path
    folder_path = os.path.dirname(pdf_file)   # Gets the directory path

    # Extract filename without extension
    file_name = os.path.splitext(os.path.basename(pdf_file))[0]  # Removes the .pdf extension

    # Create CSV file path
    csv_file = os.path.join(folder_path, f"{file_name}.csv")

    # Save a sample DataFrame
    total_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"CSV saved at: {csv_file}")

    return csv_file


def main(folder_path):
    pdf_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print(f"No PDF files found in: {folder_path}")
        return

    print(f"Found {len(pdf_files)} PDF file(s) in: {folder_path}")

    saved = []
    failed = []
    for pdf_file in pdf_files:
        try:
            csv_path = process_pdf(pdf_file)
            saved.append(csv_path)
        except Exception as e:
            print(f"\n❌ Failed to process {os.path.basename(pdf_file)}: {e}")
            failed.append(pdf_file)

    print(f"\n{'=' * 80}")
    print(f"Done. {len(saved)} CSV(s) saved, {len(failed)} failed.")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    folder = os.path.abspath(folder)
    main(folder)
