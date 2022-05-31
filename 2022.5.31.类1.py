class Student():
    def __init__(self, name, stu_id, score):
        self.name = name
        self.stu_id = stu_id
        self.score = score
    def ave(self,score):
        print((score[0]+score[1]+score[2])/3.0)
    def nam(self,name):
        print(name)
    def stu(self,stu_id):
        print(stu_id)

student1 = Student('mutao',120,[70,80,90])
student1.nam("mutao")
student1.stu("21313")
student1.ave([70,80,90])


