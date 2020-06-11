list1 = []
list2 = []
list3 = []
N= int(input("Enter the total number of students "))
I= 0

while I < N:
    I = I + 1
    print("enter the %d weight in lbs" %I)
    input_value = int(input())
    list1.append(input_value)
print("The entered weights in lbs are : ", list1)

#using Loops
for n in list1:
    list2.append(n * 0.453592)
print("The entered weights in kgs using loops is : ", list2)

#Using Comprehensions
list3 = [n * 0.453592 for n in list1]
print("The entered weights in kgs using comp is : ", list3)
