in1 = input("Enter the string ")

# Defining function to be called
def string_alternative(in1):
    out = ""
    # Checking for even index and copy data only for even index's
    for i in range(len(in1)):
        if i%2 == 0:
            out = out + in1[i]
    # printing the input and processed output
    print("input is : ", in1)
    print("Modified output is : ", out)
    #return out

#Function call
string_alternative(in1)





