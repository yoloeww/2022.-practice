class Model:
    
  def run(self, x):
   return x ** 2
  def __call__(self, x):
   return x + 2
model = Model()
model(3)
