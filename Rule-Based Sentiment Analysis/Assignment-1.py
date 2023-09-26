import string
import numpy as np

def import_data(train_file):
    f = open(train_file, 'r')
    train_lines = f.readlines()
    f.close()

    train_data, train_label = [], []
    for i in train_lines:
        [text, label] = i.split('|')
        train_data.append(text)
        train_label.append(int(label))
    return train_data, train_label

def word_frequency(data, label):
    words = dict()
    for i in range(len(data)):
        phrase_words = data[i].split()
        for j in phrase_words:
            j = j.lower()
            if j.isalpha() and j not in words:
                words[j] = {'total': 0, 'pos': 0, 'neg': 0}
            if j.isalpha() and label[i] == 1:
                words[j]['pos'] += 1
                words[j]['total'] += 1
            elif j.isalpha():
                words[j]['neg'] += 1
                words[j]['total'] += 1

    return words

def sentiment_analysis_model(phrase):
    score = 0
    for i in phrase.split():
        if i in words:
            if words[i]['pos']>words[i]['neg']:
                score += abs(words[i]['pos'] - words[i]['neg'])/words[i]['total'] 
            elif words[i]['pos']<words[i]['neg']:
                score -= abs(words[i]['neg'] - words[i]['pos'])/words[i]['total'] 
    if score >= 0:
        return 1
    else:
        return -1
    
def train(func, data, label):
    for key in ['i','my','me','he','if','when','how','john','q','them','their','they','into']:
        if key in words:
            words.pop(key) 

    total, correct = 0, 0
    for i in range(len(data)):
        if func(data[i]) == label[i]:
            correct += 1
        total += 1

    accuracy = correct/total
    
    return accuracy

def test(test_file, func):
    f = open(test_file, 'r')
    test_lines = f.readlines()
    f.close()

    test_data, test_label = [], []
    for i in test_lines:
        [text, label] = i.split('|')
        test_data.append(text)
        test_label.append(int(label))

    score = 0
    for idx in range(len(test_data)):
        if int(func(test_data[idx])) == int(test_label[idx]):
            score += 1

    return score / len(test_data)

if __name__ == '__main__':
    train_file = 'train.txt'
    test_file = 'test.npy'

    train_data, train_label = import_data(train_file)
    words =  word_frequency(train_data, train_label)

    train_acc = train(sentiment_analysis_model, train_data, train_label)
    print(f"Your method has the training accuracy of {train_acc*100}%")

    test_acc = test('test.txt', sentiment_analysis_model)
    print(f"Your method has the test accuracy of {test_acc*100}%")