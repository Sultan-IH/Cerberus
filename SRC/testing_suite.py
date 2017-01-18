class Color:
    def __init__(self, color):
        self.color = color

    def getcolor(self):
        print(self.color)
        return self.color


class Color2(Color):
    def getcolor2(self):
        print(self.getcolor() + " extended!")


col = Color2("pink+white")
col.getcolor()
