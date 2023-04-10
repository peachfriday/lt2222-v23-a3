import os
import sys
import argparse
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle

# path = "./data/enron_sample"

def stripdata(directory):
    content = []
    authors1 = []
    for author in os.listdir(directory):
        author_dir = os.path.join(directory, author)
        if os.path.isdir(author_dir):
                for filename in os.listdir(author_dir):
                    filepath = os.path.join(author_dir, filename)
                    with open(filepath, 'r') as f:
                        text = f.read()
                        text = re.sub(r'Cordially,','',text)
                        text = re.sub(r'Susan S. Bailey\n','',text)
                        text = re.sub(r'Enron North America Corp.','',text)
                        text = re.sub(r'1400 Smith Street, Suite 3803A','',text)
                        text = re.sub(r'Phone: (713) 853-4737','',text)
                        text = re.sub(r'Fax: (713) 646-3490','',text)
                        text = re.sub(r'Email: Susan.Bailey@enron.com','',text)
                        text = re.sub(r'From:','',text)
                        text = re.sub(r'To:','',text)
                        text = re.sub(r'X-From:','',text)
                        text = re.sub(r'X-To:','',text)
                        text = re.sub(r'X-cc:','',text)
                        text = re.sub(r'X-Origin:','',text)
                        text = re.sub(r'Sent from my BlackBerry Wireless Handheld (www.BlackBerry.net)','',text)
                        text = re.sub(r'X-Origin:','',text)
                        text = re.sub(r'Rick\n','',text)
                        text = re.sub(r'Rick \n','',text)
                        text = re.sub(r'Shelley Corman\n','',text)
                        text = re.sub(r'Shelley Corman \n','',text)
                        text = re.sub(r'Stacy\n','',text)
                        text = re.sub(r'Stacy \n','',text)
                        text = re.sub(r'Craig Dean  	(503) 880.5303','',text)
                        text = re.sub(r'Geir Solberg	(503) 772.0515','',text)
                        text = re.sub(r'Geir Solberg	(503) 772.0515','',text)
                        text = re.sub(r'Sincerely,','',text)
                        text = re.sub(r'Thanks,','',text)
                        text = re.sub(r'Thank you,','',text)
                        text = re.sub(r'Tom Donohoe \n','',text)
                        text = re.sub(r'Tom Donohoe\n','',text)
                        text = re.sub(r'713-853-7151','',text)
                        text = re.sub(r'Thanks','',text)
                        text = re.sub(r'Kam','',text)
                        text = re.sub(r'Tom Donohoe \n','',text)
                        text = re.sub(r'KK \n','',text)
                        text = re.sub(r'KK\n','',text)
                        text = re.sub(r'See ya','',text)
                        text = re.sub(r'Diane \n','',text)
                        text = re.sub(r'Diane\n','',text)
                        text = re.sub(r'Mandi\n','',text)
                        text = re.sub(r'Mandi \n','',text)
                        text = re.sub(r'Sincerely,','',text)
                        text = re.sub(r'Andrew\n','',text)
                        text = re.sub(r'Andrew \n','',text)
                        text = re.sub(r'x57534','',text)
                        text = re.sub(r'Carl Tricoli\n','',text)
                        text = re.sub(r'Carl Tricoli \n','',text)
                        text = re.sub(r'x3-5781','',text)
                        text = re.sub(r'Frank\n','',text)
                        text = re.sub(r'Rosie\n','',text)
                        text = re.sub(r'Rosie \n','',text)
                        text = re.sub(r'Lawrence J. May \n','',text)
                        text = re.sub(r'Lawrence J. May\n','',text)
                        text = re.sub(r'Wk. 713 853-6731 email larry.may@enron.com','',text)
                        text = re.sub(r'Hm. 281 379-1525 email ljnmay@aol.com','',text)
                        text = re.sub(r'Dan\n','',text)
                        text = re.sub(r'Dan \n','',text)
                        text = re.sub(r'Stephanie Panus\n','',text)
                        text = re.sub(r'Senior Legal Specialist\n','',text)
                        text = re.sub(r'Enron Wholesale Services\n','',text)
                        text = re.sub(r'1400 Smith Street, EB3803C\n','',text)
                        text = re.sub(r'Houston, Texas 77002\n','',text)
                        text = re.sub(r'ph:  713.345.3249\n','',text)
                        text = re.sub(r'fax:  713.646.3490\n','',text)
                        text = re.sub(r'email:  stephanie.panus@enron.com\n','',text)
                        text = re.sub(r'Dutch\n','',text)
                        text = re.sub(r'Dutch \n','',text)
                        text = re.sub(r'Eric \n','',text)
                        text = re.sub(r'Eric\n','',text)
                        text = re.sub(r'Holden\n','',text)
                        text = re.sub(r'Holden \n','',text)
                        text = re.sub(r'Jim \n','',text)
                        text = re.sub(r'Jim\n','',text)
                        text = re.sub(r'Love\n','',text)
                        text = re.sub(r'Love,\n','',text)
                        text = re.sub(r'Shelley\n','',text)
                        text = re.sub(r'Shelley \n','',text)
                        text = re.sub(r'Message-ID','',text)
                        text = re.sub(r'Sent:','',text)
                        text = re.sub(r'Subject:','',text)
                        text = re.sub(r'Sent:','',text)
                        text = re.sub(r'-----Original Message-----','',text)
                        content.append(text)
                        authors1.append(author)
        
    return content, authors1

mails, authors = stripdata(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    
    mails, authors =stripdata(args.inputdir)

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(mails)
    print(matrix.toarray())
    
    x_train, x_test, y_train, y_test = train_test_split(matrix, authors, train_size=0.8, test_size=0.2, shuffle=False)

    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.

    data = [x_train, y_train, x_test, y_test]
    train_df = pd.DataFrame(x_train.toarray())
    train_df['author'] = y_train
    train_df['set'] = 'train'
    test_df = pd.DataFrame(x_test.toarray())
    test_df['author'] = y_test
    test_df['set'] = 'test'
    df = pd.concat([train_df, test_df], ignore_index=True)
    df.to_csv(args.outputfile, index=False, columns=list(range(args.dims)) + ['author'] + ['set'])

    print("Done!")
