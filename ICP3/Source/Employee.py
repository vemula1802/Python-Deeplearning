class Employee:
    count = 0
    total_sal = 0
    def __init__(self,n,f,s,d):
        self.name = n
        self.family = f
        self.salary = s
        self.department = d
        # Keeping a count of number of employees
        Employee.count += 1
        # Keeping count of total salary
        Employee.total_sal += self.salary

    def average_sal(self):
        average_sal = self.total_sal / self.count
        print("The average salary is: " + str(average_sal))

    def info(self):
        print("Name:"+self.name, "Family:"+self.family, "Salary:" + str(self.salary), "Department:"+self.department)

#Sub class for the employee class and inheriting the fields from Employee class
class FullTimeEmployee(Employee):
    pass
fullTimeEmp1 = FullTimeEmployee("Raj", "Vemula", 7000, "Developer")
fullTimeEmp2 = FullTimeEmployee("Dev", "Vemula", 9000, "Manager")
fullTimeEmp3 = FullTimeEmployee("Reddy", "vemula", 3500, "Testing")

# Inheriting the methods of main class in inherited class
fullTimeEmp1.info()
print("Total number of employees are : " ,Employee.count)
Employee.average_sal(Employee)