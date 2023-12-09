import matplotlib.pyplot as plt
# %matplotlib inline  

# class Circle(object):
    
#     # Constructor
#     def __init__(self, radius=3, color='blue'):
#         self.radius = radius
#         self.color = color 
    
#     # Method
#     def add_radius(self, r):
#         self.radius = self.radius + r
#         return(self.radius)
    
#     # Method
#     def drawCircle(self):
#         plt.gca().add_patch(plt.Circle((0, 0), radius=self.radius, fc=self.color))
#         plt.axis('scaled')
#         plt.show()  

# RedCircle = Circle(10, 'red')
# RedCircle = Circle(radius = 100)
# dir(redCircle)
# RedCircle.radius
# redCircle
# print('Radius of object:',RedCircle.radius)
# RedCircle.add_radius(2)
# print('Radius of object of after applying the method add_radius(2):',RedCircle.radius)
# RedCircle.add_radius(5)
# print('Radius of object of after applying the method add_radius(5):',RedCircle.radius)
# RedCircle.drawCircle()

# class Rectangle(object):
    
#     # Constructor
#     def __init__(self, width=2, height=3, color='r'):
#         self.height = height 
#         self.width = width
#         self.color = color
    
#     # Method
#     def drawRectangle(self):
#         plt.gca().add_patch(plt.Rectangle((0, 0), self.width, self.height ,fc=self.color))
#         plt.axis('scaled')
#         plt.show()
#         print("Circle formulated")
        
# # SkinnyBlueRectangle = Rectangle(2, 3, 'blue')
# # print(SkinnyBlueRectangle.height) 
# # SkinnyBlueRectangle.drawRectangle()

# FatYellowRectangle = Rectangle(20, 5, 'yellow')
# print("Height:", FatYellowRectangle.height) 
# print("Width:", FatYellowRectangle.width)
# print("Color:", FatYellowRectangle.width)
# FatYellowRectangle.drawRectangle()

# class Car(object):
    
#     # color = "White"

#     def __init__(self, max_speed, mileage, color):
#         self.max_speed = max_speed
#         self.mileage = mileage
#         self.color = color
#         self.capacity = None

#     def seating_capacity(self, capacity):
#         self.capacity = capacity

#     def properties(self):
#         print("Properties of car:")
#         print("Color:", self.color)
#         print("Maximum Speed:", self.max_speed)
#         print("Mileage:", self.mileage)
#         print("Seating Capacity:", self.capacity)

# Car1 = Car(200, 50000, "White")
# Car1.seating_capacity(5)
# print(Car1.properties())

# Car2 = Car(180, 75000, "Black")
# Car2.seating_capacity(4)
# print(Car2.properties())

givenstring="Lorem ipsum dolor! diam amet, consetetur Lorem magna. sed diam nonumy eirmod tempor. diam et labore? et diam magna. et diam amet."

class TextAnalyzer(object):
    
    def __init__ (self, text):
        # remove punctuation
        formattedText = text.replace('.','').replace('!','').replace('?','').replace(',','')
        
        # make text lowercase
        formattedText = formattedText.lower()
        
        self.fmtText = formattedText
        
    def freqAll(self):        
        # split text into words
        wordList = self.fmtText.split(' ')
        
        # Create dictionary
        freqMap = {}
        for word in set(wordList): # use set to remove duplicates in list
            freqMap[word] = wordList.count(word)
        
        return freqMap
    
    def freqOf(self,word):
        # get frequency map
        freqDict = self.freqAll()
        
        if word in freqDict:
            return freqDict[word]
        else:
            return 0
        
analyzed = TextAnalyzer(givenstring)
freqMap = analyzed.freqAll()
word = "lorem"
frequency = analyzed.freqOf(word)


print("Formatted Text:", analyzed.fmtText)
print(freqMap)
print("The word",word,"appears",frequency,"times.")