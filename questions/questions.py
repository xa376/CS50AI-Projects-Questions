import nltk
import sys
import os
import string
import math
nltk.download('stopwords')
nltk.download('punkt')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    filesDict = {}

    # Reads files from corpus into dict, with filename as key and contents as value
    for file in os.listdir(directory):
        filePath = os.path.join(directory, file)
        with open(filePath, "r", encoding="utf-8") as openFile:
            filesDict[file] = openFile.read()
        print(f"{file} read successfully.")

    return filesDict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    wordsList = []

    # For word in the document, lowercase all words, remove all punctuation, append to list
    for word in nltk.word_tokenize(document.lower()):
        for character in word:
            if character in string.punctuation:
                word = word.replace(character, '')

        if word not in nltk.corpus.stopwords.words("english") and word != "":
            wordsList.append(word)
    
    return wordsList


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    allWords = {}
    idfDict = {}
    totalDocuments = len(documents)

    # Counts how many documents each word appears in
    for words in documents.values():
        # Removes duplicate words from text
        wordsSet = set(words)

        # Adds word to allWords dict or adds 1 if already there (counts documents word is present in)
        for word in wordsSet:
            if word not in allWords.keys():
                allWords[word] = 1
            else:
                allWords[word] += 1

    # Populates the idfDict with each word as a key and its idf score as its value
    for word, timesFound in allWords.items():
        idfDict[word] = math.log(totalDocuments / timesFound)

    return idfDict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    scoresList = []
    topMatches = []
    wordScores = {}

    # Calculates each document score in scoresList
    for file, words in files.items():
        wordScores = {}
        totalScore = 0
        for word in words:
            if word in query:
                if word not in wordScores.keys():
                    wordScores[word] = 1
                else:
                    wordScores[word] += 1
        
        # Calculates tfidf for each query word in document and adds to total
        for word in wordScores:
            totalScore += wordScores[word] * idfs[word]

        scoresList.append((file, totalScore))

    # Sorts scores in descending order
    scoresList.sort(key=lambda tf: tf[1], reverse = True)

    # Appends to list each document by score in ascending order
    for score in scoresList:
        topMatches.append(score[0])

    # Returns only n top matches
    return topMatches[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentenceScores = []
    topMatches = []

    # Calculates each sentences tf-idf score and quality term density, then appends the results to sentenceScores
    for sentence, words in sentences.items():
        wordsFromQuery = 0
        score = 0
        wordsSet = set(words)
        
        # Counts amount of query words found in sentence
        for word in words:
            if word in query:
                wordsFromQuery += 1

        # Adds idf value of each individual query word found in sentence
        for word in wordsSet:
            if word in query:
                score += idfs[word]

        # Calculates quality term density of sentence
        qTermDensity = wordsFromQuery / len(words)

        # Appends to list the sentence, total idf score, and quality term density
        sentenceScores.append((sentence, score, qTermDensity))

    # Sorts sentences descending by total idf score followed by quality term density
    sentenceScores.sort(key=lambda sc: (sc[1], sc[2]), reverse = True)

    # Appends only the sentence to topMatches, retaining the score order
    for sentence in sentenceScores:
        topMatches.append(sentence[0])
    
    # Returns only n amount of top matches
    return topMatches[:n]


if __name__ == "__main__":
    main()
