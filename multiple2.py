# Base Class 1
class Birds:
    def __init__(self):
        print("[Flyer] Can fly!")
    def pigeon(self):
        print("[Flyer] Flying...")

# Base Class 2
class Fish:
    def __init__(self):
        print("[Swimmer] Can swim!")
    def dolphin(self):
        print("[Swimmer] Swimming...")

# Derived Class
class Animal(Birds,Fish):
    def __init__(self):
        Birds.__init__(self)
        Fish.__init__(self)
        print("[Duck] Initialized!")

    def sound(self):
        print("[Duck] Quacking!")

# Object creation and method calls
donald = Animal()
print(donald.pigeon())
print(donald.dolphin())
print(donald.sound())
###################################EX;2########################################
# Base Class 1
class Human:
    def Men(self):
        print("[Walker] Walking...")

# Base Class 2
class Racer:
    def Women(self):
        print("[Runner] Running...")

# Derived Class
class Gamer(Human,Racer):
    def great(self):
        print("[Athlete] Competing!")

# Object creation and method calls
speaker= Gamer()
print(speaker.Men())
print(speaker.Women())
print(speaker.great())
##################################EX:3#################################
# Base Class 1
class Paint:
    def painter(self):
        print("[Painter] Painting...")

# Base Class 2
class Song:
    def singer(self):
        print("[Singer] Singing...")

# Derived Class
class Actor(Paint,Song):
    def Artist(self):
        print("[Artist] Performing!")

# Object creation and method calls
Creator= Actor()
print(Creator.painter())
print(Creator.singer())
print(Creator.Artist())

# Base Class 1
class Painter:
    def paint(self):
        print("[Painter] Painting...")

# Base Class 2
class Singer:
    def song(self):
        print("[Singer] Singing...")

# Derived Class
class Artist(Painter,Singer):
    def actor(self):
        print("[Artist] Performing!")

# Object creation and method calls
Creator= Artist()
print(Creator.paint())
print(Creator.song())
print(Creator.actor())

# Base Class 1
class Hema:
    def paint(self):
        print("[Painter] Painting...")

# Base Class 2
class Bhagya:
    def song(self):
        print("[Singer] Singing...")

# Derived Class
class Methu(Hema,Bhagya):
    def actor(self):
        print("[Artist] Performing!")

# Object creation and method calls
Creator= Methu()
print(Creator.paint())
print(Creator.song())
print(Creator.actor())