# Base Class
class Animal:
    def __init__(self, name):
        self.name = name
        print(f"[Animal] Initialized: {self.name}")

    def make_sound(self):
        print(f"[Animal] {self.name} is making a sound.")

# Derived Class
class Dog(Animal):
    def __init__(self, name, breed):
        Animal.__init__(self, name)
        self.breed = breed
        print(f"[Dog] Initialized: {self.breed}")

    def Bark(self):
        print(f"[Dog] {self.name} the {self.breed} is barking!")

# Object creation and method calls
my_dog = Dog("Buddy", "Labrador")

my_dog.make_sound()
my_dog.Bark() 

##############EX:2###############

# Base Class
class Fruit:
    def __init__(self, color):
        self.color = color
        print(f"[Fruit] Color: {self.color}")

    def describe(self):
        print(f"[Fruit] This fruit is {self.color}.")

# Derived Class
class Apple(Fruit):
    def __init__(self, color, taste):
        super().__init__(color)
        self.taste = taste
        print(f"[Apple] Taste: {self.taste}")

    def describe(self):
        print(f"[Apple] This apple tastes {self.taste}.")

# Object creation and method calls
my_apple = Apple("Red", "Sweet")
my_apple.describe()
my_apple.describe() 

###########EX:3##########

# Base Class
class Book:
    def __init__(self, title):
        self.title = title
        print(f"[Book] Title: {self.title}")

    def read(self):
        print(f"[Book] Reading {self.title}")

# Derived Class
class Novel(Book):
    def __init__(self, title, author):
        super().__init__(title)
        self.author = author
        print(f"[Novel] Author: {self.author}")

    def read(self):
        print(f"[Novel] {self.title} by {self.author}")

# Object creation and method calls
my_novel = Novel("1984", "George Orwell")
my_novel.read()
my_novel.read() 

# Base Class
class Book:
    def __init__(self, title):
        self.title = title
        print(f"[Book] Title: {self.title}")

    def read(self):
        print(f"[Book] Reading {self.title}")

# Derived Class
class Novel(Book):
    def __init__(self, title, author):
        super().__init__(title)
        self.author = author
        print(f"[Novel] Author: {self.author}")

    def Write(self):
        print(f"[Novel] {self.title} by {self.author}")

# Object creation and method calls
my_novel = Novel("1984", "George Orwell")
my_novel.read()
my_novel.Write() 