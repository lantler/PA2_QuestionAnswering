### Authors
# - Anna Fenn
# - June Huck
# - Leah Antler
# - Vitaliy Kishchenko

#Submitted 06/23/2024

#Course #: AIT-526-002 (202440.42417)

## QA Program Explanation

### The Problem to be solved
# A Question Answer System (QA) is a program meant to answer simple Who, What, When, Where Questions provided by user input.
# It does this from any domain and provides complete sentences as answers. Our program does this by striping down what the user is tying to ask
# finding both the subject and type of question being asked, finding the wiki page of subject and searching the summary for particular information
# based on the type of question asked.



### Examples, usage instructions, and bonus functionality
# To use our program, you can simply run the python file or run it in the terminal with an argument, for a logfile name if wanted, as such:
#> python qa-system.py example_log_file.txt
#> python qa-system.py
#If you don't specify a name, it automatically names it logfile.txt
# Once the program runs you will begin to see output and be prompted by the algorithm for input.
# Make sure your input is a question of who, what, when, or where, and is both spelled correctly and has proper grammar.
# An example of a proper question could be:
#> When was George Washington Born?
#> Who is Obama?
#> Where is China?
# You will then recieve a curated answer, if it is easy to gather that, or prompt indicating that the algorithm cannot answer that question
#Ex: When was George Washington Born?
#Ans: George Washington was born February 22, 1732
# After which you can continue to input more questions as above or simply reply "exit" to quit the program at any time

#Bonus functionality potentially includes the handling of born related questions or rather specific variants of when based questions, 
# where it specifically gathers dates from the standard parenthesis format.


### Algorithms and program logic
# Our programs main logic is as follows:
# 1. Take the users input and strip it down into its fundemental components, what is the subject of the question, what type of question,
#    does the question make sense, etc. And does this using a variety of Entity Recognition techniques, word spotting, POS tagging, matching, and regex
# 2. The program, if the question is usable, grabs the summary entry from the named entities wikipedia page, using Wikipediaapi tools
# 3. The program then tries to find a specific pattern of sentence based on the type of question inputted, such as sentences surrounding location
#    Sentences that might have the words born, is located, includes the name of the entity, etc.
# 4. Once the probable sentence with that information is housed within the summary, the program then extracts the information from the 
#    that sentence and concatenates the information if the information needs to be more concise or returns the sentence in most who or what sentences

#
import pandas as pd
import re
import sys
import nltk
import wikipediaapi
from nltk.tokenize import sent_tokenize, RegexpTokenizer, word_tokenize
import sys
import logging
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy.matcher import Matcher
nltk.download('punkt', quiet=True)


def get_user_input():
    userInput = input("> ")
    return userInput

def end_condition(text):
    tmptxt = text
    pattern = r'[^A-Za-z]' # Remove non-letter characters (I nuke the punctuation)
    tmptxt = re.sub(pattern, '', tmptxt)
    tmptxt = tmptxt.lower()
    if tmptxt == 'exit':
        return False
    else:
        return True
    
#Leah's scraping code from here
def fetch_wikipedia_summary(topic):
    user_agent = 'PA2/1.0 (lantler@gmu.edu)'
    wiki_api = wikipediaapi.Wikipedia('en', headers={'User-Agent': user_agent})
    page = wiki_api.page(topic)
    if page.exists():
        return page.summary
    else:
        return ""
    
def correct_tags(key_words):
    corrected_tags = []
    pos_tokens = nltk.word_tokenize(key_words)
    tags = nltk.pos_tag(pos_tokens)

    # Rule-based disambiguation
    for i, (word, tag) in enumerate(tags):
        if word.lower() == "fall" and i > 0 and tags[i-1][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            corrected_tags.append((word, 'VB'))
        elif word.lower() == "die" and i > 0 and tags[i-1][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            corrected_tags.append((word, 'VB'))
        elif word.lower() == "born" and i > 0 and tags[i-1][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            corrected_tags.append((word, 'VB'))
        elif word.lower() == "start" and i > 0 and tags[i-1][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            corrected_tags.append((word, 'VB'))
        else:
            corrected_tags.append((word, tag))
    return corrected_tags

def find_key_words(query):
    unimportant_words = r"\b(([Ww]here|[Ww]hat|[Ww]ho|[Ww]hen) (is|was|did))( (a|the))?\b"
    key_words = re.sub(unimportant_words, "", query)
    return key_words

def find_noun(corrected_tags):
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'IN']

    # Filter out only the nouns
    nouns = [word for word, pos in corrected_tags if pos in noun_tags]
    noun = " ".join(nouns)

    return noun

def gen_search_terms(query):
    # Find the key words
    key_words = find_key_words(query)
    
    # Then, Create tags on these key words
    corrected_tags = correct_tags(key_words)

    # Then, find nouns
    noun = find_noun(corrected_tags)

    # Join nouns
    search_term = "".join(noun)
    
    return search_term

#Leah's scraping code from here
def fetch_wikipedia_summary(topic):
    user_agent = 'PA2/1.0 (lantler@gmu.edu)'
    wiki_api = wikipediaapi.Wikipedia('en', headers={'User-Agent': user_agent})
    page = wiki_api.page(topic)
    if page.exists():
        return page.summary
    else:
        return ""
    
def entity_search(text):
    # Perform NER on the text
    doc = nlp(text)
    # Get the named entities
    if doc.ents:
        # Assign the first named entity to the global variable entity
        entity = doc.ents[0].text
    else:
        # If no named entities are found, set entity to an empty string or other default value
        entity = ""
    return entity

def get_first_date(text):
    doc = nlp(text)

    first_date = None
    # Iterate through entities to find the first date
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            first_date = ent.text
            break  # Stop as soon as the first date is found
    
    return first_date

def find_sentence_with_date(text):
    doc = nlp(text)
    
    matcher = Matcher(nlp.vocab)
    matcher.add("DATE", [[{"ENT_TYPE": "DATE"}]])
    
    for sentence in doc.sents:
        sentence_doc = nlp(sentence.text)
        matches = matcher(sentence_doc)
        if matches:
            return sentence.text # Match sentence if it has a date
        
    return None 

def find_sentence_with_location(text):
    doc = nlp(text)
    
    matcher = Matcher(nlp.vocab)
    matcher.add("DATE", [[{"ENT_TYPE": "GPE"}]])
    
    for sentence in doc.sents:
        sentence_doc = nlp(sentence.text)
        matches = matcher(sentence_doc)
        if matches:
            return sentence.text # Match sentence if it has GPE
    
    return None 

def find_sentence_with_regex(text, regex_pattern):
    doc = nlp(text)
    
    for sentence in doc.sents:
        if re.search(regex_pattern, sentence.text):
            return sentence.text
    
    return None 

def words_after_given_word(text, target_word):
    if text == None:
        return ''
    words = text.split()
    if target_word in words:
        target_index = words.index(target_word)
        return ' '.join(words[target_index + 1:])
    else:
        return ''

def find_last_gpe_in_sentence(sentence):
    doc = nlp(sentence)
    
    last_gpe = None

    for ent in doc.ents:
        if ent.label_ == "GPE":
            last_gpe = ent.text
    
    return last_gpe

def gen_who_response(user_input, wikipedia_summary):
    
    user_input= nltk.word_tokenize(user_input)
    doc = nlp(wikipedia_summary)
    
    # If they're not just asking about who a person is/was
    if user_input[1] != r'\b(is|was)\b':
        matcher = Matcher(nlp.vocab)
        matcher.add("ENTITY", [[{"ENT_TYPE": "PERSON"}]])
    
        for sentence in doc.sents:
            sentence_doc = nlp(sentence.text)
            matches = matcher(sentence_doc)
            if matches:
                cleaned_sentence = re.sub(r'\([^)]*\)', '', sentence.text).strip()
                return cleaned_sentence
    
    # Or if they're just asking about a person, return the first sentence of their bio
    else:    
    # Return the first sentence
        for sent in doc.sents:
            cleaned_sentence = re.sub(r'\([^)]*\)', '', sent.text).strip()
            return cleaned_sentence

def gen_what_response(user_input, wikipedia_summary):
    # Return the first sentence of the wikipedia summary
    doc = nlp(wikipedia_summary)
    
    # Return the first sentence
    for sent in doc.sents:
        return sent.text.strip()

def gen_when_response(user_input, wikipedia_summary):
    born_matches = re.findall(r'\bborn\b', user_input)
    born_question = False
    if len(born_matches) != 0:
        born_question = True
    
    if born_question:
        # Get the entity in the question
        named_entity = entity_search(user_input)
        
        sentences = sent_tokenize(wikipedia_summary)
        sentence_with_born = ''
        counted = False
        
        for sentence in sentences:
            if 'born' in sentence.lower() and counted == False:
                sentence_with_born = sentence
                counted = True

        # Grab date from sentence that contains the word born
        date = get_first_date(sentence_with_born)

        # If not in sentence, search entire text
        if date == None:
            date = get_first_date(wikipedia_summary)

        if date != None:
            return "{} was born on {}.".format(named_entity, date)
        else:
            return "I'm sorry, I don't know the answer."
    else:
        # Return the first sentence with a date in it
        sent_with_date = find_sentence_with_date(wikipedia_summary)

        if sent_with_date != None:
            return sent_with_date
        else:
            return "I'm sorry, I don't know the answer."

def gen_where_response(user_input, wikipedia_summary):
    location_patterns = r"\b(located)\b"
    response_sent = find_sentence_with_regex(wikipedia_summary, location_patterns)
    object_name = gen_search_terms(user_input)
    sent_part_2 = words_after_given_word(response_sent, "located")
    user_words = user_input.split()
    tense = 'was'
    if user_words[1] == 'is':
        tense = 'is'
    if sent_part_2 != '' and response_sent != None:
        return object_name + " " + tense + " located " + sent_part_2

    doc = nlp(wikipedia_summary)
    
    # Return the first sentence
    for sent in doc.sents:
        return sent.text.strip()
    
    return "I'm sorry, I don't know the answer."
    
def generate_response(user_input, wikipedia_summary):
    question_patterns = {
        'what': r'\bwhat\b',
        'who': r'\bwho\b',
        'where': r'\bwhere\b',
        'when': r'\bwhen\b',
    }

    # Make user input lowercase
    user_input_lower = user_input.lower()

    # Identify question type
    question_type = None
    for qtype, pattern in question_patterns.items():
        if re.search(pattern, user_input_lower):
            question_type = qtype
            break

    response = "I'm sorry, I don't know the answer."
    # Now we have question type. Let's handle each case seperately
    if question_type == 'what':
        response = gen_what_response(user_input, wikipedia_summary)
    if question_type == 'who':
        response = gen_who_response(user_input, wikipedia_summary)
    if question_type == 'where':
        response = gen_where_response(user_input, wikipedia_summary)
    if question_type == 'when':
        response = gen_when_response(user_input, wikipedia_summary)
    
    if response == None or response == "None":
        response = "I'm sorry, I don't know the answer."

    return response

def respond_to_input(user_input):
    # First, check if the user said 'quit'
    continue_conv = end_condition(user_input)
    
    # Next, generate a wikipedia query based on the user input
    search_term = gen_search_terms(user_input)
    logging.info("WIKIPEDIA QUERY: {}".format(search_term))
    #print("WIKIPEDIA QUERY: {}".format(search_term))

    # Next, get a wikipedia summary based on this query
    wikipedia_summary = fetch_wikipedia_summary(search_term)
    logging.info("WIKIPEDIA SUMMARY: {}".format(wikipedia_summary))
    #print("----\nWIKIPEDIA SUMMARY: {}\n-----".format(wikipedia_summary))

    # Last, generate a response based on user input and the wikipedia summary
    response = generate_response(user_input, wikipedia_summary)
    
    return response, continue_conv

def main():
    conversating = True

    # Start of conversation
    print("This is a QA system by Leah Antler, Anna Fenn, June Huck, and Vitaliy Kishchenko. ", end="")
    print("It will try to answer questions that start with Who, What, When or Where. ", end="")
    print("Enter \"exit\" to leave the program.")
    print("Please ask a question. Type 'exit' to exit.")
    logging.info("Please ask a question. Type 'exit' to exit.")
    # Speech loop
    while conversating:
        # Grab user input
        user_input = get_user_input()
        
        # Generate response, decide whether to continue the conversation
        response, continue_conv = respond_to_input(user_input)

        # If we're not continuing our conversation, break out of the loop.
        if not continue_conv:
            break

        # Print our response
        print(response)
        logging.info("RESPONSE: {}".format(response))
    
    print("Thank you! Goodbye.")
    logging.info("Thank you! Goodbye.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        logfile = str(sys.argv[1])
    else:
        logfile = 'logfile.txt'

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        filename=logfile,
        filemode='a'
    )
    main()
