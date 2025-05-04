# # EXAMPLE 1
text = "X-DSPAM-Confidence:    0.8475"
a1 = text.find('0')
number = float(text[a1:])
print(number)

line = " abc  "
line.strip()

# EXAMPLE 2
# Write a program that prompts for a file name, then opens that file and reads through the file, 
# looking for lines of the form:
# X-DSPAM-Confidence:    0.8475
# Count these lines and extract the floating point values from each of the lines and 
# compute the average of those values and produce an output as shown below. 
# Do not use the sum() function or a variable named sum in your solution.

try:
    fh = open(fname)  # Open the file
except FileNotFoundError:
    print("Error: File not found.")
    quit()

count = 0
total = 0.0  # Instead of sum(), we use total

# Read through the file line by line
for line in fh:
    if line.startswith("X-DSPAM-Confidence:"):  # Look for the target lines
        start_pos = line.find("0")  # Find position of number
        number = float(line[start_pos:])  # Extract and convert to float
        total += number  # Accumulate total
        count += 1  # Count occurrences

fh.close()  # Close the file

# Compute and print the average
if count > 0:
    average = total / count
    print("Average spam confidence:", average)
else:
    print("No matching lines found.")



# using sum funtion
values = []  # List to store float values
for line in fh:
    if line.startswith("X-DSPAM-Confidence:"):
        start_pos = line.find("0")
        number = float(line[start_pos:])
        values.append(number)

if values:
    average = sum(values) / len(values)
    print("Average spam confidence:", average)
else:
    print("No matching lines found.")



#8.4 Open the file romeo.txt and read it line by line. 
# For each line, split the line into a list of words using the split() method. 
# The program should build a list of words. 
# For each word on each line check to see if the word is already in the list and 
# if not append it to the list. 
# When the program completes, sort and print the resulting words in python sort() order 
# as shown in the desired output.

# Prompt for file name
fname = input("Enter file name: ")

# Open the file
fh = open(fname)

# Initialize an empty list
unique_words = []

# Read file line by line
for line in fh:
    words = line.split()  # Split line into words
    for word in words:
        if word not in unique_words:  # Check if word is already in list
            unique_words.append(word)

fh.close()  # Close the file

# Sort the list in alphabetical order
unique_words.sort()

# Print the sorted list of words
print(unique_words)





# 8.5 Open the file mbox-short.txt and read it line by line. 
# When you find a line that starts with 'From ' like the following line:
# From stephen.marquard@uct.ac.za Sat Jan  5 09:14:16 2008

fname = input("Enter file name: ")
if len(fname) < 1:
    fname = "mbox-short.txt"

fh = open(fname)
count = 0

for line in fh:
    if line.startswith("From "):   
        count = count + 1
        email_1 = line.split()
        print(email_1[1])
print("There were", count, "lines in the file with From as the first word")



data = 'From stephen.marquard@uct.ac.za Sat Jan 5 09:14:16 2008'
atpos = data.find('@') # indexing?
print(atpos)
21
sppos = data.find(' ',atpos)
print(sppos)
31
host = data[atpos+1:sppos]
print(host)
uct.ac.za



# Write a program to read through the mbox-short.txt and 
# figure out who has sent the greatest number of mail messages. 
# The program looks for 'From ' lines and takes the second word of those lines 
# as the person who sent the mail. 
# The program creates a Python dictionary that maps the sender's mail address to 
# a count of the number of times they appear in the file. 
# After the dictionary is produced, the program reads through the dictionary 
# using a maximum loop to find the most prolific committer.

# Open file
fname = input("Enter file name: ")
if len(fname) < 1:
    fname = "mbox-short.txt"  # Default file
fh = open(fname)

# Create dictionary to store email counts
email_counts = {}

# Read through the file line by line
for line in fh:     # sentence by sentence
    if line.startswith("From "):  # Only process lines that start with 'From '
        words = line.split()
        email = words[1]  # Extract email address
        email_counts[email] = email_counts.get(email, 0) + 1  # Update count

# Find the most prolific sender
max_count = 0
max_sender = None

for sender, count in email_counts.items():
    if count > max_count:
        max_count = count
        max_sender = sender
    print(sender, count)  # This prints all senders and their counts
# Print result, only the max and its count
print(max_sender, max_count)




# USING PANDA
import pandas as pd
emails = []

with open('file.txt') as fh:
    for line in fh:
        if line.startswith("From "):
            words = line.split()
            emails.append(words[1])

# Use pandas to count and get the top
df = pd.Series(emails).value_counts()
print(df)  # All senders and their counts

# Most prolific sender
print(df.idxmax(), df.max())
#idxmax Return index of first occurrence of maximum over requested axis.




# f-string / formatted string literal
>>> camels = 42
>>> f'{camels}'
'42'


x = {'chuck': 1, 'fred': 42, 'jan': 100}  # A dictionary with key-value pairs
y = x.items()  # Retrieves the key-value pairs as a view object
print(y)






# From stephen.marquard@uct.ac.za Sat Jan  5 09:14:16 2008
# counting the hour, splitting and indexing

# Open the file
fname = input("Enter file name: ")
fh = open(fname)

# Create an empty dictionary to store hour counts
hour_counts = {}

# Read through the file line by line
for line in fh:
    if line.startswith("From "):  # Filter only lines starting with "From "
        words = line.split()  # Split line into words
        time_part = words[5]  # Get the time part (e.g., '09:14:16')
        hour = time_part.split(':')[0]  # Extract the hour (e.g., '09')

        # Update dictionary count for the hour
        hour_counts[hour] = hour_counts.get(hour, 0) + 1

# Sort dictionary by hour (key) and print
for hour, count in sorted(hour_counts.items()):
    print(hour, count)


