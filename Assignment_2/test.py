import sys

a = sys.argv[1]
b = sys.argv[2]

total = 0
count = 0

with open(a,"r") as a, open(b, "r") as b:
	for x, y in zip(a,b):
		total += 1
		if x == y:
			count += 1
print(total)
print(count*100 / total)