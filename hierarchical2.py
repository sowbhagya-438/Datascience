# Base Class
class Shape:
    def __init__(self, name):
        self.name = name
        print(f"[Shape] Initialized: {self.name}")

    def Draw(self):
        print(f"[Shape] Drawing {self.name}")

# Derived Class 1
class Circle(Shape):
    def __init__(self, name, radius):
        super().__init__(name)
        self.radius = radius
        print(f"[Circle] Radius: {self.radius}")

    def Area1(self):
        print(f"[Circle] Area: {3.14 * self.radius * self.radius}")

# Derived Class 2
class Square(Shape):
    def __init__(self, name, side):
        super().__init__(name)
        self.side = side
        print(f"[Square] Side: {self.side}")

    def Area2(self):
        print(f"[Square] Area: {self.side * self.side}")

# Object creation and method calls
c = Circle("Circle", 5)
s = Square("Square", 4)
c.Draw()
c.Area1()
s.Draw()
s.Area2() 

##############EX:2############

# Base Class
class Employee:
    def __init__(self, name):
        self.name = name
        print(f"[Employee] Name: {self.name}")

    def work(self):
        print(f"[Employee] Working: {self.name}")

# Derived Class 1
class Manager(Employee):
    def __init__(self, name, department):
        super().__init__(name)
        self.department = department
        print(f"[Manager] Department: {self.department}")

    def manage(self):
        print(f"[Manager] Managing {self.department}")

# Derived Class 2
class Developer(Employee):
    def __init__(self, name, project):
        super().__init__(name)
        self.project = project
        print(f"[Developer] Project: {self.project}")

    def code(self):
        print(f"[Developer] Coding for {self.project}")

# Object creation and method calls
mgr = Manager("Alice", "HR")
dev = Developer("Bob", "AI")
mgr.manage()
mgr.work()
dev.code()
dev.work() 

############EX:3############

# Base Class
class Vehicle:
    def __init__(self, name):
        self.name = name
        print(f"[Vehicle] Name: {self.name}")

    def Move(self):
        print(f"[Vehicle] Moving: {self.name}")

# Derived Class 1
class Bike(Vehicle):
    def __init__(self, name, wheels):
        super().__init__(name)
        self.wheels = wheels
        print(f"[Bike] Wheels: {self.wheels}")

    def Ride(self):
        print(f"[Bike] Riding with {self.wheels} wheels")

# Derived Class 2
class Bus(Vehicle):
    def __init__(self, name, seats):
        super().__init__(name)
        self.seats = seats
        print(f"[Bus] Seats: {self.seats}")

    def Carry(self):
        print(f"[Bus] Carrying {self.seats} passengers")

# Object creation and method calls
bike = Bike("MountainBike", 2)
bus = Bus("CityBus", 40)
bike.Move()
bike.Ride()
bus.Move()
bus.Carry() 