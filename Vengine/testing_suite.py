class Color:
    def __init__(self):
        self.mes = 'hello'

    def method(self, mes2):
        print(self.mes + mes2)


class Color2(Color):
    def call(self):
        self.method(" world")


col = Color2()
col.call()
