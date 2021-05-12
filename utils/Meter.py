class Meter(object):
	def __init__(self, name, display, fmt=':f', end=','):
		self.name = name
		self.display = display
		self.fmt = fmt
		self.start_time = 0
		self.end = end
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.start_time = 0
		self.time = 0

	def set_start_time(self, start_time):
		self.start_time = start_time

	def update(self, val, n=1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum/self.count
		self.time = val-self.start_time

	def __str__(self):
		fmtstr = '{name}:{'+self.display+self.fmt+'}'+self.end
		return fmtstr.format(**self.__dict__)
