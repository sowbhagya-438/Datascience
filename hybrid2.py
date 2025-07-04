# Base Class
class Person:
    def __init__(self, name):
        self.name = name
        print(f"[Person] Initialized: {self.name}")

# Parent Class 1
class Employee:
    def Work(self):
        print("[Employee] Working...")

# Parent Class 2
class Student:
    def Study(self):
        print("[Student] Studying...")

# Derived Class (Hybrid)
class Intern_worker(Person, Employee, Student):
    def __init__(self, name):
        Person.__init__(self, name)
        print("[Intern] Initialized!")

    def Internship(self):
        print(f"[Intern] {self.name} is interning!")

# Object creation and method calls
intern = Intern_worker("Alice")
intern.Work()
intern.Study()
intern.Internship()
##############EX:3############
# Base Class
class Account:
    def __init__(self, id):
        self.id = id
        print(f"[Account] ID: {self.id}")

# Parent Class 1
class Buyer:
    def buy(self):
        print("[Buyer] Buying...")

# Parent Class 2
class Seller:
    def sell(self):
        print("[Seller] Selling...")

# Derived Class (Hybrid)
class MarketplaceUser(Account, Buyer, Seller):
    def __init__(self, id):
        Account.__init__(self, id)
        print("[MarketplaceUser] Initialized!")

    def is_active(self):
        print(f"[MarketplaceUser] User {self.id} is active!")

# Object creation and method calls
user = MarketplaceUser(101)
user.buy()
user.sell()
user.is_active()

################EX:3##################
# Base Class
# Base Class
class Device:
    def __init__(self, name):
        self.name = name
        print(f"[Device] Name: {self.name}")

# Parent Class 1
class Camera:
    def Take(self):
        print("[Camera] Taking photo...")

# Parent Class 2
class Phone:
    def Make(self):
        print("[Phone] Making call...")

# Derived Class (Hybrid)
class Smartphone(Device,Camera,Phone):
    def __init__(self, name):
        Device.__init__(self, name)
        print("[Smartphone] Ready!")

    def Smart(self):
        print(f"[Smartphone] {self.name} is smart!")

# Object creation and method calls
sp = Smartphone("Pixel")
sp.Take()
sp.Make()
sp.Smart() 
     