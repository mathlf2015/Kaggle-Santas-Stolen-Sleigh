#import pandas as pd
import numpy as np
import math
import time
import pickle
import copy
import random
import csv
from haversine import haversine

class Tabu:
	def __init__(self, clu1, clu2, gift1, gift2) :
		self.clu1 = clu1
		self.clu2 = clu2
		self.gift1 = gift1
		self.gift2 = gift2
		self.level = 100
	def kill(self) :
		self.level -= 1
		if self.level==0 :
			return True
		return False
	def compare(self, clu1, clu2, gift1, gift2) :
		if self.clu1 == clu1 and self.clu2 == clu2 and self.gift1 == gift1 and self.gift2 == gift2 : return True
		else : return False

def dist_north(id) :
	# calculates the haversine distance between north pole and gift[id]
	return haversine((gift_array[id,0]*180.0/math.pi, gift_array[id,1]*180.0/math.pi), (90.0, 0.0))

def dist(id1, id2) :
	# calculates the haversine distance between gift[id1] and gift[id2]
	return haversine((gift_array[id1,0]*180.0/math.pi, gift_array[id1,1]*180.0/math.pi), (gift_array[id2,0]*180.0/math.pi, gift_array[id2,1]*180.0/math.pi))

# (1) weight big (2) located in upper region (high latitude)
def compare(idx1, idx2) :
	"""
	if gift_weight_list[idx1] != gift_weight_list[idx2] :
		if (gift_weight_list[idx1] < gift_weight_list[idx2]) : return 1
		else : return -1
	if gift_array[idx1,0] < gift_array[idx2,0] : return 1
	elif gift_array[idx1,0] == gift_array[idx2,0] : return 0
	else : return -1
	"""
	if idx1==idx2 : return 0

	if gift_array[idx1,0] < gift_array[idx2,0] : return 1
	elif gift_array[idx1,0] > gift_array[idx2,0] : return -1
	else :
		if gift_array[idx1,1] < gift_array[idx2,1] : return 1
		elif gift_array[idx1,1] == gift_array[idx2,1] : return 0
		else : return -1

def bin_search(list, idx) :
	#return random.randint(0, len(list)-1)
	if compare(list[-1], idx)<0 : return len(list)
	left = 0
	right = len(list)-1
	mid = (left+right)/2
	while left<=right :
		mid = (left+right)/2
		if compare(list[mid], idx)<0 :
			left = mid+1
		elif compare(list[mid], idx)>0 : 
			right = mid-1
		else :
			return mid
	if right<mid : return mid
	else : return mid+1

def sofar(gift1, clu2) :
	"""
	if len(cluster_list[clu2])==0 : return False
	gift2 = cluster_list[clu2][0]
	difflat = abs(gift_array[gift1,0]-gift_array[gift2,0])*180.0/math.pi
	temp = abs(gift_array[gift1,1]-gift_array[gift2,1])*180.0/math.pi
	difflong = min(temp, 360.0-temp)
	if difflat + difflong > 60.0 : return True
	else : return False
	"""
	if len(cluster_list[clu2])==0 : return True
	gift2 = cluster_list[clu2][0]
	temp = abs(gift_array[gift1,1]-gift_array[gift2,1])*180.0/math.pi
	difflong = min(temp, 360.0-temp)
	if difflong > 20.0 : return True
	else : return False

def delta_delete(clu1, gift_idx, x) :
	delta_WRW = 0.0
	nowlist = cluster_list[clu1]
	if gift_idx==0 :
		if len(nowlist)==1 : delta_WRW = -WRW_list[clu1]
		else :
			delta_WRW = -weight_list[clu1]*dist_north(x) + (weight_list[clu1]-gift_weight_list[x])*(dist_north(nowlist[gift_idx+1])-dist(nowlist[gift_idx], nowlist[gift_idx+1]))
	elif gift_idx==len(nowlist)-1 :
		delta_WRW = -10.0*dist_north(x) -(10.0+gift_weight_list[x])*(dist(nowlist[gift_idx], nowlist[gift_idx-1])) + 10.0*dist_north(nowlist[gift_idx-1]) - gift_weight_list[x]*accum_dist_list[clu1][gift_idx-1]
	else :
		#weight_sum = 10.0
		#for i in range(gift_idx+1, len(nowlist)) : weight_sum += gift_weight_list[nowlist[i]]
		weight_sum = accum_weight_list[clu1][gift_idx]
		delta_WRW = -gift_weight_list[x]*accum_dist_list[clu1][gift_idx] - weight_sum * (dist(nowlist[gift_idx-1], nowlist[gift_idx])+dist(nowlist[gift_idx], nowlist[gift_idx+1])) + weight_sum * dist(nowlist[gift_idx-1], nowlist[gift_idx+1])
	return delta_WRW

def delta_insert(clu2, idx, x) :
	delta_WRW = 0.0
	if len(cluster_list[clu2])==0 :
		delta_WRW += (20.0+gift_weight_list[x])*dist_north(x)
	else :
		if idx==0 : 
			delta_WRW += (weight_list[clu2]+gift_weight_list[x])*dist_north(x) + weight_list[clu2]*dist(x, cluster_list[clu2][0]) - weight_list[clu2]*dist_north(cluster_list[clu2][0])
		elif idx==len(cluster_list[clu2]) :
			delta_WRW += 10.0*dist_north(x) + (10.0+gift_weight_list[x])*dist(cluster_list[clu2][-1], x) - 10.0*dist_north(cluster_list[clu2][-1]) + gift_weight_list[x]*accum_dist_list[clu2][idx-1]
		else :
			delta_WRW += accum_dist_list[clu2][idx-1] * gift_weight_list[x] + (gift_weight_list[x] + accum_weight_list[clu2][idx-1]) * dist(cluster_list[clu2][idx-1], x) +	accum_weight_list[clu2][idx-1]*(dist(x, cluster_list[clu2][idx])-dist(cluster_list[clu2][idx-1], cluster_list[clu2][idx]))
	return delta_WRW

csv_file = open("gifts.csv", 'rb') # 0~99999
reader = csv.reader(csv_file)

# save values from dataframe
gift_array = np.zeros((100000, 2))
gift_weight_list = np.zeros((100000,))
for row in reader :
	if row[0] == 'GiftId' : continue
	# convert lat/lon into radian
	gift_array[int(row[0])-1, 0] = float(row[1])*math.pi/180.0
	gift_array[int(row[0])-1, 1] = float(row[2])*math.pi/180.0
	gift_weight_list[int(row[0])-1] = float(row[3])

## naive k-means for initial trip
"""
random_idx = np.random.permutation(gifts.shape[0])
k = 1800
center_array = np.zeros((k, 2))
count_list = np.zeros((k,))
weight_list = np.zeros((k,))
assign_list = np.zeros((gifts.shape[0]))
dead_list = np.zeros((k,)) # is this cluster dead? true if >0

# random center assign
for i in range(k) :
	center_array[i,:] = gift_array[random_idx[i], :]

for epoch in range(100) :
	print epoch
	# assign cluster
	count_list = np.zeros((k,))
	weight_list = np.zeros((k,))
	random_idx = np.random.permutation(gifts.shape[0])
	for temp in range(gifts.shape[0]) :
		i = random_idx[temp]
		best_dist = 999999999999.99
		best_cluster = -1
		dist_array = np.linalg.norm(center_array - np.tile(gift_array[i, :], (k, 1)), axis=1)
		arg_sort = np.argsort(dist_array)
		for j in range(k) :
			best_cluster = arg_sort[j]
			if weight_list[best_cluster] + gift_weight_list[i] > 1000.0 : continue
			if dead_list[best_cluster]==0 : continue
			assign_list[i] = best_cluster
			weight_list[best_cluster] += gift_weight_list[i]
			count_list[best_cluster] += 1
			break

	# move cluster center
	center_array = np.zeros((k,2))
	for i in range(gifts.shape[0]) : 
		center_array[assign_list[i],:] += gift_array[i,:]
	for i in range(k) :
		if count_list[i]==0 : 
			dead_list[i]=1
			continue
		center_array[i,:] /= count_list[i]

cluster_list = []
for i in range(k) : cluster_list.append([]);
for i in range(gifts.shape[0]) :
	cluster_list[int(assign_list[i])].append(i)	

total_sum = 0.0
for i in range(k) :
	cluster_list[i] = sorted(cluster_list[i], cmp=compare)
	nowsum = 0.0
	nowweight = 10.0
	for x in cluster_list[i] : nowweight+=gift_weight_list[x]
	if nowweight>1010.0 : 
		print "damn....... %d" % i
	nowposition = (90.0, 0.0)
	for x in cluster_list[i] :
		nowdist = haversine(nowposition, (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi))
		nowsum += nowweight*nowdist
		nowweight -= gift_weight_list[x]
		nowposition = (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi)

	nowsum += nowweight*haversine(nowposition, (90.0, 0.0))
	total_sum += nowsum;
print total_sum	

f = open('cluster.txt', 'rb')
[assign_list, dead_list, center_array, weight_list] = pickle.load(f)
f.close()
"""

csv_file = open("start.csv", 'rb')
reader = csv.reader(csv_file)

k=1523
cluster_list = []
assign_list = np.zeros(100000,)
accum_dist_list = [] # total travel distance to get to 'certain' gift place
accum_weight_list = [] # weight after giving gift to 'certain' gift place
WRW_list = np.zeros((k, ))
weight_list = np.zeros((k, ))
for i in range(k) : cluster_list.append([]); accum_dist_list.append([]); accum_weight_list.append([])
for row in reader :
	if row[0] == 'GiftId' or row[0] == 'TripId' : continue 
	cluster_list[int(row[1])-1].append(int(row[0])-1)
	assign_list[int(row[0])-1] = int(row[1])-1

total_sum = 0.0
for i in range(k) :
	cluster_list[i] = sorted(cluster_list[i], cmp=compare)
	nowsum = 0.0
	nowweight = 10.0
	for x in cluster_list[i] : nowweight+=gift_weight_list[x]
	weight_list[i] = nowweight
	nowposition = (90.0, 0.0)
	for x in cluster_list[i] :
		nowdist = haversine(nowposition, (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi))
		nowsum += nowweight*nowdist
		nowweight -= gift_weight_list[x]
		nowposition = (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi)
		accum_dist_list[i].append(nowdist)
		accum_weight_list[i].append(nowweight)
	for j in range(len(cluster_list[i])) : 
		if j==0 : continue
		accum_dist_list[i][j] = accum_dist_list[i][j] + accum_dist_list[i][j-1]

	nowsum += nowweight*haversine(nowposition, (90.0, 0.0))
	WRW_list[i] = nowsum;
	total_sum += nowsum;
print total_sum

# for making close-arrays
"""
close_array = np.zeros((100000, 100))
for i in range(100000) :
	if i%1000==0 : print i
	dist_array = np.linalg.norm(np.tile(gift_array[i,:], (100000, 1)) - gift_array, axis=1)
	close_array[i,:] = np.argsort(dist_array)[1:101]
"""

f = open('close_array.txt', 'rb')
close_array = pickle.load(f)
f.close()

# tabu search for trip maker
print "Tabu Search Begins!"
total_best_score = total_sum
total_best_cluster_list = []
now_score = total_sum
tabu_list = []
nondec = 0
block = 999
clus_look = 3
gone_num = 0
for epoch in range(1000001) :

	best_result = 98765432123456789.87
	best_clu1 = -1
	best_clu2 = -1
	best_gift1 = -1
	best_gift2 = -1
	best_idx = -1
	best_idx2 = -1
	best_test = -1
	best_delta = -1.0

	# neighbor 1 : choose one gift & move to other cluster
	random_idx = np.random.permutation(k)
	success_num = 0
	cnt = 0
	while success_num < clus_look :

		clu1 = random_idx[cnt]
		nowlist = cluster_list[clu1]
		while len(nowlist)==0 : 
			cnt += 1; clu1 = random_idx[cnt]; nowlist = cluster_list[clu1]
		success_num += 1

		for gift_idx in range(min(block, len(nowlist))) :
			x = nowlist[gift_idx]
			# calculate the effect of deleting this gift
			delta_WRW = 0.0
			if gift_idx==0 :
				if len(nowlist)==1 : delta_WRW = -WRW_list[clu1]
				else :
					delta_WRW = -weight_list[clu1]*dist_north(x) + (weight_list[clu1]-gift_weight_list[x])*(dist_north(nowlist[gift_idx+1])-dist(nowlist[gift_idx], nowlist[gift_idx+1]))
			elif gift_idx==len(nowlist)-1 :
				delta_WRW = -10.0*dist_north(x) -(10.0+gift_weight_list[x])*(dist(nowlist[gift_idx], nowlist[gift_idx-1])) + 10.0*dist_north(nowlist[gift_idx-1]) - gift_weight_list[x]*accum_dist_list[clu1][gift_idx-1]
			else :
				weight_sum = 10.0
				for i in range(gift_idx+1, len(nowlist)) : weight_sum += gift_weight_list[nowlist[i]]
				delta_WRW = -gift_weight_list[x]*accum_dist_list[clu1][gift_idx] - weight_sum * (dist(nowlist[gift_idx-1], nowlist[gift_idx])+dist(nowlist[gift_idx], nowlist[gift_idx+1])) + weight_sum * dist(nowlist[gift_idx-1], nowlist[gift_idx+1])
			for clu2 in range(k) :
				if clu1==clu2 : continue
				if weight_list[clu2]+gift_weight_list[x] > 1010.0 : continue
				if sofar(x, clu2) : continue
				#bonus = -10000000.0/(len(nowlist))**2 + 10000000.0 /(len(cluster_list[clu2]))**2
				bonus = 0.0
				now_delta = delta_WRW
				idx = 0
				if len(cluster_list[clu2])==0 :
					now_delta += (20.0+gift_weight_list[x])*dist_north(x)
				else :
					idx = bin_search(cluster_list[clu2], x) # put in list[idx]
					if idx==0 : 
						now_delta += (weight_list[clu2]+gift_weight_list[x])*dist_north(x) + weight_list[clu2]*dist(x, cluster_list[clu2][0]) - weight_list[clu2]*dist_north(cluster_list[clu2][0])
					elif idx==len(cluster_list[clu2]) :
						now_delta += 10.0*dist_north(x) + (10.0+gift_weight_list[x])*dist(cluster_list[clu2][-1], x) - 10.0*dist_north(cluster_list[clu2][-1]) + gift_weight_list[x]*accum_dist_list[clu2][idx-1]
					else :
						now_delta += accum_dist_list[clu2][idx-1] * gift_weight_list[x] + (gift_weight_list[x] + accum_weight_list[clu2][idx-1]) * dist(cluster_list[clu2][idx-1], x) +	accum_weight_list[clu2][idx-1]*(dist(x, cluster_list[clu2][idx])-dist(cluster_list[clu2][idx-1], cluster_list[clu2][idx]))
				if now_delta + bonus < best_result :
					tocontinue = False
					for tabu in tabu_list :
						if tabu.compare(clu1, clu2, x, -1) == True : tocontinue = True; break
					if tocontinue : continue
					best_result = now_delta+bonus
					best_clu1 = clu1
					best_clu2 = clu2
					best_gift1 = x
					best_gift2 = -1
					best_idx = idx
					best_test = -1
					best_delta = now_delta
		
	
	# neighbor 2 : choose some close-pair gifts and change their trip
	random_idx = np.random.permutation(100000)
	for i in range(10) :
		gift1 = random_idx[i]
		for j in range(50) : 
			gift2 = int(close_array[gift1, j])
			clu1 = int(assign_list[gift1])
			clu2 = int(assign_list[gift2])
			if clu1==clu2 : continue
			if weight_list[clu1]-gift_weight_list[gift1]+gift_weight_list[gift2] > 1010.0 : continue
			if weight_list[clu2]-gift_weight_list[gift2]+gift_weight_list[gift1] > 1010.0 : continue
			now_delta = 0.0
			idx1 = bin_search(cluster_list[clu1], gift1)
			idx2 = bin_search(cluster_list[clu2], gift2)
			cluster_list[clu1].remove(gift1)
			cluster_list[clu2].remove(gift2)
			
			idx1_t = bin_search(cluster_list[clu2], gift1)
			idx2_t = bin_search(cluster_list[clu1], gift2)
			cluster_list[clu1].insert(idx2_t, gift2)
			cluster_list[clu2].insert(idx1_t, gift1)

			nowweight = 10.0
			nowsum = 0.0
			nowposition = (90.0, 0.0)
			for x in cluster_list[clu1] : nowweight+=gift_weight_list[x]
			for x in cluster_list[clu1] :
				nowdist = haversine(nowposition, (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi))
				nowsum += nowweight*nowdist
				nowweight -= gift_weight_list[x]
				nowposition = (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi)
			nowsum += nowweight*haversine(nowposition, (90.0, 0.0))
			now_delta += nowsum - WRW_list[clu1]
			
			nowweight = 10.0
			nowsum = 0.0
			nowposition = (90.0, 0.0)
			for x in cluster_list[clu2] : nowweight+=gift_weight_list[x]
			for x in cluster_list[clu2] :
				nowdist = haversine(nowposition, (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi))
				nowsum += nowweight*nowdist
				nowweight -= gift_weight_list[x]
				nowposition = (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi)
			nowsum += nowweight*haversine(nowposition, (90.0, 0.0))
			now_delta += nowsum - WRW_list[clu2]

			cluster_list[clu1].remove(gift2)
			cluster_list[clu2].remove(gift1)
			cluster_list[clu1].insert(idx1, gift1)
			cluster_list[clu2].insert(idx2, gift2)

			if now_delta < best_result : 
				tocontinue = False
				for tabu in tabu_list :
					if tabu.compare(clu1, clu2, gift1, gift2) == True : tocontinue = True; break
				if tocontinue : continue
				best_result = now_delta
				best_clu1 = clu1
				best_clu2 = clu2
				best_gift1 = gift1
				best_gift2 = gift2
				best_idx = idx1_t
				best_idx2 = idx2_t
				best_test = gift2
				best_delta = now_delta
	
	# really moving
	if best_delta > 5000.0 : continue # don't move if so high
	now_score += best_delta
	if best_gift2 <0 : # just move one
		cluster_list[best_clu1].remove(best_gift1)
		cluster_list[best_clu2].insert(best_idx, best_gift1)
		assign_list[best_gift1] = best_clu2
	else :
		cluster_list[best_clu1].remove(best_gift1)
		cluster_list[best_clu2].remove(best_gift2)
		cluster_list[best_clu1].insert(best_idx2, best_gift2)
		cluster_list[best_clu2].insert(best_idx, best_gift1)
		assign_list[best_gift1] = best_clu2
		assign_list[best_gift2] = best_clu1

	tabu_list.append(Tabu(best_clu1, best_clu2, best_gift1, best_gift2))
	tabu_list.append(Tabu(best_clu2, best_clu1, best_gift1, best_gift2))

	# for debug
	if len(cluster_list[best_clu1])==0 :
		gone_num += 1
		print "GONE!!!!!!!!!!!!! %d" % gone_num

	# update accum dist, accum weight, weight, WRW
	accum_weight_list[best_clu1] = []
	accum_weight_list[best_clu2] = []
	accum_dist_list[best_clu1] = []
	accum_dist_list[best_clu2]= [] 

	testdelta = - WRW_list[best_clu1] - WRW_list[best_clu2]

	nowweight = 10.0
	nowsum = 0.0
	for x in cluster_list[best_clu1] : nowweight+=gift_weight_list[x]
	weight_list[best_clu1] = nowweight
	nowposition = (90.0, 0.0)
	for x in cluster_list[best_clu1] :
		nowdist = haversine(nowposition, (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi))
		nowsum += nowweight*nowdist
		nowweight -= gift_weight_list[x]
		nowposition = (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi)
		accum_dist_list[best_clu1].append(nowdist)
		accum_weight_list[best_clu1].append(nowweight)
	for j in range(len(cluster_list[best_clu1])) : 
		if j==0 : continue
		accum_dist_list[best_clu1][j] = accum_dist_list[best_clu1][j] + accum_dist_list[best_clu1][j-1]		
	nowsum += nowweight*haversine(nowposition, (90.0, 0.0))
	WRW_list[best_clu1] = nowsum

	nowweight = 10.0
	nowsum = 0.0
	for x in cluster_list[best_clu2] : nowweight+=gift_weight_list[x]
	weight_list[best_clu2] = nowweight
	nowposition = (90.0, 0.0)
	for x in cluster_list[best_clu2] :
		nowdist = haversine(nowposition, (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi))
		nowsum += nowweight*nowdist
		nowweight -= gift_weight_list[x]
		nowposition = (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi)
		accum_dist_list[best_clu2].append(nowdist)
		accum_weight_list[best_clu2].append(nowweight)
	for j in range(len(cluster_list[best_clu2])) : 
		if j==0 : continue
		accum_dist_list[best_clu2][j] = accum_dist_list[best_clu2][j] + accum_dist_list[best_clu2][j-1]
	nowsum += nowweight*haversine(nowposition, (90.0, 0.0))
	WRW_list[best_clu2] = nowsum

	# for debug
	testdelta += (WRW_list[best_clu1]+WRW_list[best_clu2])
	if abs(testdelta-best_delta) > 0.000001 :
		print "Prediction / Real", testdelta, best_delta
		print "WRONG"

	# save best list
	if now_score < total_best_score : 
		total_best_score = now_score
		total_best_cluster_list = copy.deepcopy(cluster_list)
	# nondecrease?
	if best_result > 0 :
		nondec += 1
		if (nondec>25) :
			block += 1
			clus_look += 1
			nondec = 0
	else : nondec = 0

	if (epoch+1)%1==0 :
		print "Epoch %d / Score : %lf, Delta : %lf, Test : %d" % (epoch+1, now_score, best_delta, best_test)
	# tabu list operation
	tabu_list = [x for x in tabu_list if not x.kill()]

	if epoch>15000 : block=999

	# save data
	save_period = 2000
	if epoch%save_period!=0 or epoch==0 : continue
	csv_file = open("result_new_martini_"+str(18+epoch/save_period)+".csv", "wb")
	cw = csv.writer(csv_file, delimiter=',', quotechar='|')
	cw.writerow(["GiftId", "TripId"])
	for i in range(k) :
		for x in total_best_cluster_list[i] :
			cw.writerow([x+1, i+1])
	csv_file.close()
	print "Saved result of epoch %d, best score : %lf" % (epoch, total_best_score)
