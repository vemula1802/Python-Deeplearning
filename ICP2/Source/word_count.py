input_file = open("doc.txt","r+")
count = {} #creating a dictionary to save the words and their counts
for word in input_file.read().split():
    if word in count:
        count[word] += 1
    else:
        count[word] = 1

# For writing data  into file
for word, count[word] in count.items():
    z = "\n" + word + " : " + str(count[word])
    input_file.write(z)

input_file.close()