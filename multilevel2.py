# Base Class (Level 1)
class Vehicle:
    def __init__(self, brand):
        self.brand = brand
        print(f"[Vehicle] Initialized for brand: {self.brand}")

    def Starting_engine(self):
        print(f"[Vehicle] Starting engine of {self.brand}")

# Derived Class (Level 2)
class Car(Vehicle):
    def __init__(self, brand, model):
        Vehicle.__init__(self, brand)  # Call base class constructor
        self.model = model
        print(f"[Car] Initialized for model: {self.model}")

    def Drive(self):
        print(f"[Car] Driving {self.brand} {self.model}")

# Derived Class (Level 3)
class ElectricCar(Car):
    def __init__(self, brand, model, battery_range):
        super().__init__(brand, model)
        self.battery_range = battery_range
        print(f"[ElectricCar] Initialized with range: {self.battery_range} km")

    def Charge(self):
        print(f"[ElectricCar] Charging {self.brand} {self.model}... Range: {self.battery_range} km")


# Object creation and method calls
my_electric_car = ElectricCar("Tesla", "Model 3", 500)

my_electric_car.Starting_engine()
my_electric_car.Drive()
my_electric_car.Charge() 

# Base Class (Level 1)
class Vehicle:
    def __init__(self, brand):
        self.brand = brand
        print(f"[Vehicle] Initialized for brand: {self.brand}")

    def Starting_engine(self):
        print(f"[Vehicle] Starting engine of {self.brand}")

# Derived Class (Level 2)
class Car(Vehicle):
    def __init__(self, brand, model):
        Vehicle.__init__(self, brand)  # Call base class constructor
        self.model = model
        print(f"[Car] Initialized for model: {self.model}")

    def Drive(self):
        print(f"[Car] Driving {self.brand} {self.model}")

# Derived Class (Level 3)
class ElectricCar(Car):
    def __init__(self, brand, model, battery_range):
        super().__init__(brand, model)
        self.battery_range = battery_range
        print(f"[ElectricCar] Initialized with range: {self.battery_range} km")

    def Charge(self):
        print(f"[ElectricCar] Charging {self.brand} {self.model}... Range: {self.battery_range} km")

    def Starting_engine(self):  # Override base method
        print(f"[ElectricCar] No engine to start. {self.brand} {self.model} is electric!")
# Object creation and method calls
my_electric_car = ElectricCar("Tesla", "Model 3", 500)

my_electric_car.Starting_engine()
my_electric_car.Drive()
my_electric_car.Charge()

#########################EX:2#####################

# Base Class
class Device:
    def __init__(self, name):
        self.name = name
        print(f"[Device] Name: {self.name}")

    def power_on(self):
        print(f"[Device] Powering on {self.name}")

# Derived Class 1
class Computer(Device):
    def __init__(self, name, os):
        super().__init__(name)
        self.os = os
        print(f"[Computer] OS: {self.os}")

    def boot(self):
        print(f"[Computer] Booting {self.name} with {self.os}")

# Derived Class 2
class Laptop(Computer):
    def __init__(self, name, os, version):
        super().__init__(name, os)
        self.version = version
        print(f"[Laptop] Version: {self.version}")

    def run(self):
        print(f"[Laptop] {self.name} running {self.os} v{self.version}")

# Object creation and method calls
my_laptop = Laptop("ThinkPad", "Windows", "10")
my_laptop.power_on()
my_laptop.boot()
my_laptop.run() 

#######################EX:3############################

