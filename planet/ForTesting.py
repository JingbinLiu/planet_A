
try:
    aa = 1.0/0
except Exception as e:
    print(e)
    print(aa)


class Animal():
    def __init__(self):
        self.age = 1

class Cat(Animal):
    def __init__(self):
        super().__init__()
        self.age = 2
        self.name = 'Cat'


c = Cat()
print(c.age)
print(c.name)


class Student():
    def func(self,x=1):
        print('hello',x)
    def func(self):
        print('hello')

stu = Student()
stu.func()
stu.func(3)


